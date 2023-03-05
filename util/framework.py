import os
from tqdm import tqdm
import torch
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertConfig

from model.word_encoder import BERTWordEncoder
from model.copner import COPNER
from model.viterbi import ViterbiDecoder
from util.data_loader import SingleDatasetwithRamdonSample

def get_abstract_transitions(train_fname, args):
    """
    Compute abstract transitions on the training dataset for StructShot
    """
    samples = SingleDatasetwithRamdonSample(train_fname, None, None, word_map=args.train_word_map, args=args).samples
    tag_lists = [sample.tags for sample in samples]

    s_o, s_i = 0., 0.
    o_o, o_i = 0., 0.
    i_o, i_i, x_y = 0., 0., 0.
    for tags in tag_lists:
        if tags[0] == 'O': s_o += 1
        else: s_i += 1
        for i in range(len(tags)-1):
            p, n = tags[i], tags[i+1]
            if p == 'O':
                if n == 'O': o_o += 1
                else: o_i += 1
            else:
                if n == 'O':
                    i_o += 1
                elif p != n:
                    x_y += 1
                else:
                    i_i += 1

    trans = []
    trans.append(s_o / (s_o + s_i))
    trans.append(s_i / (s_o + s_i))
    trans.append(o_o / (o_o + o_i))
    trans.append(o_i / (o_o + o_i))
    trans.append(i_o / (i_o + i_i + x_y))
    trans.append(i_i / (i_o + i_i + x_y))
    trans.append(x_y / (i_o + i_i + x_y))
    return trans

class FewShotNERFramework:

    def __init__(self, args, train_data_loader, val_data_loader, test_data_loader, viterbi=False, train_fname=None):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        viterbi: Whether to use Viterbi decoding.
        train_fname: Path of the data file to get abstract transitions.
        '''
        self.args = args
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.viterbi = viterbi
        if viterbi:
            abstract_transitions = get_abstract_transitions(train_fname, args)
            self.viterbi_decoder = ViterbiDecoder(self.args.N+2, abstract_transitions, tau=args.struct_tau)

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def __get_emmissions__(self, logits, tags_list):
        # split [num_of_query_tokens, num_class] into [[num_of_token_in_sent, num_class], ...]
        emmissions = []
        current_idx = 0
        for tags in tags_list:
            emmissions.append(logits[current_idx:current_idx+len(tags)])
            current_idx += len(tags)
        assert current_idx == logits.size()[0]
        return emmissions

    def viterbi_decode(self, logits, query_tags):
        emissions_list = self.__get_emmissions__(logits, query_tags)
        pred = []
        for i in range(len(query_tags)):
            sent_scores = emissions_list[i].cpu()
            sent_len, n_label = sent_scores.shape
            sent_probs = F.softmax(sent_scores, dim=1)
            start_probs = torch.zeros(sent_len) + 1e-6
            sent_probs = torch.cat((start_probs.view(sent_len, 1), sent_probs), 1)
            feats = self.viterbi_decoder.forward(torch.log(sent_probs).view(1, sent_len, n_label+1))
            vit_labels = self.viterbi_decoder.viterbi(feats)
            vit_labels = vit_labels.view(sent_len)
            vit_labels = vit_labels.detach().cpu().numpy().tolist()
            for label in vit_labels:
                pred.append(label-1)
        return torch.tensor(pred).cuda()

    def train(self,
              model,
              model_name,
              learning_rate=1e-4,
              train_iter=30000,
              val_iter=1000,
              val_step=2000,
              load_ckpt=None,
              save_ckpt=None,
              warmup_step=300,
              grad_iter=1,
              use_sgd_for_lm=False):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        learning_rate: Initial learning rate
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        load_ckpt: Path of the checkpoint to load
        save_ckpt: Path of the checkpoint to save
        warmup_step: Num of warmup steps
        grad_iter: Accumulate gradients for grad_iter steps
        use_sgd_for_lm: Whether to use SGD for the language model
        '''
        # Init optimizer
        print('Use bert optim!')
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize 
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if use_sgd_for_lm:
            optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
        else:
            optimizer = AdamW(parameters_to_optimize, lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
        
        # load model
        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)

        model.train()

        # Training
        iter_loss = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        iter_sample = 0
        pred_cnt = 1e-9
        label_cnt = 1e-9
        correct_cnt = 0
        last_step = 0

        print("Start training...")
        with tqdm(self.train_data_loader, total=train_iter, disable=False, desc="Training") as tbar:

            for it, batch in enumerate(tbar):

                if torch.cuda.is_available():
                    for k in batch:
                        if k != 'target_classes' and \
                            k != 'sentence_num' and \
                            k != 'labels' and \
                            k != 'label2tag':
                                batch[k] = batch[k].cuda()

                    label = torch.cat(batch['labels'], 0)
                    label = label.cuda()

                logits, pred, loss = model(batch['inputs'], 
                                            batch['batch_labels'],
                                            batch['valid_masks'],
                                            batch['target_classes'],
                                            batch['sentence_num'])

                loss.backward()
                
                if it % grad_iter == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Calculate metrics
                tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(pred, label)
                
                iter_loss += self.item(loss.data)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct
                iter_sample += 1
                precision = correct_cnt / pred_cnt
                recall = correct_cnt / label_cnt
                f1 = 2 * precision * recall / (precision + recall + 1e-9) # 1e-9 for error'float division by zero'
                
                tbar.set_postfix_str("loss: {:2.6f} | F1: {:3.4f}, P: {:3.4f}, R: {:3.4f}, Correct:{}"\
                                            .format(self.item(loss.data), f1, precision, recall, correct_cnt))
                
                if (it + 1) % val_step == 0:
                    precision, recall, f1, _, _, _, _ = self.eval(model, val_iter, word_map=self.args.dev_word_map)

                    model.train()
                    if f1 > best_f1:
                        # print(f'Best checkpoint! Saving to: {save_ckpt}\n')
                        # torch.save({'state_dict': model.state_dict()}, save_ckpt)
                        best_f1 = f1
                        best_precision = precision
                        best_recall = recall
                        last_step = it
                    else:
                        if it - last_step >= self.args.early_stopping:
                            print('\nEarly Stop by {} steps, best f1: {:.4f}%'.format(self.args.early_stopping, best_f1))
                            raise KeyboardInterrupt
                
                if (it + 1) % 100 == 0:
                    iter_loss = 0.
                    iter_sample = 0.
                    pred_cnt = 1e-9
                    label_cnt = 1e-9
                    correct_cnt = 0

                if (it + 1)  >= train_iter:
                    break

        print("\n####################\n")
        print("Finish training {}, best f1: {:.4f}%".format(model_name, best_f1))
    
    def adapt_fixed(self, model, support, optimizer):
        '''
        Adapt the model to the support set with fixed steps

        model: the model to be adapted
        support: the support set
        optimizer: the optimizer
        '''
        model.train()

        for i in range(self.args.adapt_step):
            logits, pred, loss = model(support['inputs'], 
                                        support['batch_labels'],
                                        support['valid_masks'],
                                        support['target_classes'],
                                        support['sentence_num'])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()

        return logits, pred, loss

    def adapt_auto(self, model, support, optimizer):
        '''
        Adapt the model to the support set with early stop

        model: the model to be adapted
        support: the support set
        optimizer: the optimizer
        '''

        model.train()
        pre_loss = torch.tensor(10000)
        loss = pre_loss - 1
        i = 0
        while True:
            logits, pred, loss = model(support['inputs'], 
                                        support['batch_labels'],
                                        support['valid_masks'],
                                        support['target_classes'],
                                        support['sentence_num'])
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            i += 1
            # Early Stop
            if (pre_loss - loss < 0 and loss < self.args.threshold_beta and i > 1) or i > 50:
                break
            pre_loss = loss

        model.eval()

        return logits, pred, loss

    def eval_other(self,
            model,
            eval_iter,
            ckpt=None,
            word_map=None): 
        '''
        model: a FewShotREModel instance
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        word_map: Word map.

        return: metric results
        '''
        print("")
        p0 = list(model.named_parameters())
        model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        print('ignore {}'.format(name))
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        if word_map is None:
            word_map = self.args.dev_word_map

        pred_cnt = 1e-9 # pred entity cnt
        label_cnt = 1e-9 # true label entity cnt
        correct_cnt = 0 # correct predicted entity cnt

        fp_cnt = 0 # misclassify O as I-
        fn_cnt = 0 # misclassify I- as O
        total_token_cnt = 0 # total token cnt
        within_cnt = 0 # span correct but of wrong fine-grained type 
        outer_cnt = 0 # span correct but of wrong coarse-grained type
        total_span_cnt = 0 # span correct

        config = BertConfig.from_pretrained(self.args.pretrain_ckpt)
        word_encoder = BERTWordEncoder(config, args=self.args)
        eval_model = COPNER(word_encoder, self.args, word_map=word_map)

        parameters_to_optimize = list(eval_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize 
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(parameters_to_optimize, lr=self.args.adapt_lr)

        eval_iter = min(eval_iter, len(eval_dataset))
        with tqdm(eval_dataset, total=eval_iter, disable=False, desc="Evaling") as tbar:
            for it, (support, query) in enumerate(tbar):

                # copy params, Prevent updating the parameters of the train model during adapting
                old_state = model.state_dict()
                eval_state = eval_model.state_dict()

                for name, param in eval_state.items():
                    if name not in old_state:
                        print('not find {}'.format(name))

                for name, param in old_state.items():
                    if name not in eval_state:
                        print('ignore {}'.format(name))
                        continue
                    eval_state[name].copy_(param)
                # eval_model = model

                if torch.cuda.is_available():
                    eval_model = eval_model.cuda()

                    for k in support:
                        if k != 'target_classes' and \
                            k != 'sentence_num' and \
                            k != 'labels' and \
                            k != 'label2tag':
                            support[k] = support[k].cuda()
                            query[k] = query[k].cuda()
                    label = torch.cat(query['labels'], 0)
                    label = label.cuda()

                if not self.args.zero_shot:
                    if self.args.adapt_auto:
                        self.adapt_auto(eval_model, support, optimizer)
                    else:
                        self.adapt_fixed(eval_model, support, optimizer)

                with torch.no_grad():
                    logits, pred, _ = eval_model(query['inputs'], 
                                                query['batch_labels'],
                                                query['valid_masks'],
                                                query['target_classes'],
                                                query['sentence_num'])

                    if self.viterbi:
                        pred = self.viterbi_decode(logits, query['labels'])

                    tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(pred, label)
                    fp, fn, token_cnt, within, outer, total_span = model.error_analysis(pred, label, query)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct

                fn_cnt += self.item(fn.data)
                fp_cnt += self.item(fp.data)
                total_token_cnt += token_cnt
                outer_cnt += outer
                within_cnt += within
                total_span_cnt += total_span

                precision = correct_cnt / pred_cnt
                recall = correct_cnt / label_cnt
                f1 = 2 * precision * recall / (precision + recall + 1e-9)
                fp_error = fp_cnt / total_token_cnt
                fn_error = fn_cnt / total_token_cnt
                within_error = within_cnt / total_span_cnt
                outer_error = outer_cnt / total_span_cnt
                tbar.set_postfix_str("F1: {:3.4f}, P: {:3.4f}, R: {:3.4f}".format(f1, precision, recall))

                if it + 1 == eval_iter:
                    break
        p3 = list(model.named_parameters())

        precision = correct_cnt / pred_cnt
        recall = correct_cnt /label_cnt
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        print('[EVAL] f1: {:3.4f}, precision: {:3.4f}, recall: {:3.4f}\n'.format(f1, precision, recall))
        
        return precision, recall, f1, fp_error, fn_error, within_error, outer_error

    def eval_supervised(self,
            model,
            eval_iter,
            ckpt=None,
            word_map=None):

        '''
        Evaluate the model in the paradim of supervised learning.

        model: a FewShotREModel instance
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        word_map: Word map. 
        return: Metric results
        '''
        print("")
        
        model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        print('ignore {}'.format(name))
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        if word_map is None:
            word_map = self.args.dev_word_map

        pred_cnt = 1e-9 # pred entity cnt
        label_cnt = 1e-9 # true label entity cnt
        correct_cnt = 0 # correct predicted entity cnt

        fp_cnt = 0 # misclassify O as I-
        fn_cnt = 0 # misclassify I- as O
        total_token_cnt = 0 # total token cnt
        within_cnt = 0 # span correct but of wrong fine-grained type 
        outer_cnt = 0 # span correct but of wrong coarse-grained type
        total_span_cnt = 0 # span correct

        eval_iter = min(eval_iter, len(eval_dataset))
        
        with tqdm(eval_dataset, total=eval_iter, disable=False, desc="Evaling") as tbar:
            for it, query in enumerate(tbar):
                if torch.cuda.is_available():
                    for k in query:
                        if k != 'target_classes' and \
                            k != 'sentence_num' and \
                            k != 'labels' and \
                            k != 'label2tag':
                            query[k] = query[k].cuda()
                    label = torch.cat(query['labels'], 0)
                    label = label.cuda()

                with torch.no_grad():
                    logits, pred, _ = model(query['inputs'], 
                                            query['tagging_labels'], 
                                            query['index_labels'],
                                            query['target_classes'],
                                            query['sentence_num'])
                    if self.viterbi:
                        pred = self.viterbi_decode(logits, query['labels'])

                tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(pred, label)
                fp, fn, token_cnt, within, outer, total_span = model.error_analysis(pred, label, query)
                
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct

                fn_cnt += self.item(fn.data)
                fp_cnt += self.item(fp.data)
                total_token_cnt += token_cnt
                outer_cnt += outer
                within_cnt += within
                total_span_cnt += total_span

                precision = correct_cnt / pred_cnt
                recall = correct_cnt /label_cnt
                f1 = 2 * precision * recall / (precision + recall + 1e-9)
                fp_error = fp_cnt / total_token_cnt
                fn_error = fn_cnt / total_token_cnt
                within_error = within_cnt / total_span_cnt
                outer_error = outer_cnt / total_span_cnt
                tbar.set_postfix_str("F1: {:3.4f}, P: {:3.4f}, R: {:3.4f}".format(f1, precision, recall))

                if it + 1 == eval_iter:
                    break

        precision = correct_cnt / pred_cnt
        recall = correct_cnt /label_cnt
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        print('[EVAL] f1: {:3.4f}, precision: {:3.4f}, recall: {:3.4f}\n'.format(f1, precision, recall))
        
        return precision, recall, f1, fp_error, fn_error, within_error, outer_error


    def eval(self,
            model,
            eval_iter,
            ckpt=None,
            word_map=None):
        if self.args.task == 'in-label-space':
            return self.eval_supervised(model, eval_iter, ckpt, word_map)
        elif self.args.task == 'domain-adaptation':
            return self.eval_other(model, eval_iter, ckpt, word_map)
        else:
            return self.eval_other(model, eval_iter, ckpt, word_map)

