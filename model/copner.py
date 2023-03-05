from collections import OrderedDict
import torch
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

from .base import FewShotNERModel

class COPNER(FewShotNERModel):
    
    def __init__(self, word_encoder, args, word_map):
        FewShotNERModel.__init__(self, word_encoder, ignore_index=args.ignore_index)
        self.tokenizer = args.tokenizer
        self.tau = args.tau

        self.loss_fct = CrossEntropyLoss(ignore_index=args.ignore_index)
        self.method = 'euclidean'

        self.class2word = word_map
        self.word2class = OrderedDict()
        for key, value in self.class2word.items():
            self.word2class[value] = key

    def __dist__(self, x, y, dim, normalize=False):
        if normalize:         
            x = F.normalize(x, dim=-1)         
            y = F.normalize(y, dim=-1)
        if self.method == 'dot':
            sim = (x * y).sum(dim)
        elif self.method == 'euclidean':
            sim = -(torch.pow(x - y, 2)).sum(dim)
        elif self.method == 'cosine':
            sim = F.cosine_similarity(x, y, dim=dim)
        return sim / self.tau
    
    def get_contrastive_logits(self, hidden_states, inputs, valid_mask, target_classes):
        class_indexs = [self.tokenizer.get_vocab()[tclass] for tclass in target_classes]

        class_rep = []
        for iclass in class_indexs:
            class_rep.append(torch.mean(hidden_states[inputs.eq(iclass), :].view(-1, hidden_states.size(-1)), 0))
        
        class_rep = torch.stack(class_rep).unsqueeze(0)
        token_rep = hidden_states[valid_mask != self.tokenizer.pad_token_id, :].view(-1, hidden_states.size(-1)).unsqueeze(1)

        logits = self.__dist__(class_rep, token_rep, -1)

        return logits.view(-1, len(target_classes))

    def forward(self,
                input_ids,
                labels,
                valid_masks,
                target_classes,
                sentence_num,
                ):
        assert input_ids.size(0) == labels.size(0) == valid_masks.size(0), \
                print('[ERROR] inputs and labels must have same batch size.')
        assert len(sentence_num) == len(target_classes)

        hidden_states = self.word_encoder(input_ids) # logits, (encoder_hs, decoder_hs)
        
        loss = None
        logits = []
        current_num = 0
        for i, num in enumerate(sentence_num):
            current_hs = hidden_states[current_num: current_num+num]
            current_input_ids = input_ids[current_num: current_num+num]
            current_labels = labels[current_num: current_num+num]
            current_valid_masks = valid_masks[current_num: current_num+num]
            current_target_classes = target_classes[i]

            current_num += num

            contrastive_logits = self.get_contrastive_logits(current_hs, 
                                                        current_input_ids, 
                                                        current_valid_masks, 
                                                        current_target_classes)
            
            current_logits = F.softmax(contrastive_logits, -1)

            if self.training:
                contrastive_loss = self.loss_fct(contrastive_logits, current_labels[current_valid_masks != self.tokenizer.pad_token_id].view(-1))
                loss = contrastive_loss if loss is None else loss + contrastive_loss

            current_logits = current_logits.view(-1, current_logits.size(-1))

            logits.append(current_logits)

        logits = torch.cat(logits, 0)
        _, preds = torch.max(logits, 1)
        
        if loss:
            loss /= len(sentence_num)

        return logits, preds, loss
