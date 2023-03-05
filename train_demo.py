import argparse
parser = argparse.ArgumentParser()

# # Environment Settings
parser.add_argument('--gpu', default='0',
        help='the gpu number for traning')

parser.add_argument('--seed', type=int, default=42,
        help='random seed')

# # Task Settings
parser.add_argument('--mode', default='conll',
        help='training mode, must be in [inter, intra, supervised, i2b2, conll, wnut, mit-movie]')
parser.add_argument('--task', default='domain-transfer',
        help='training task, must be in [cross-label-space, domain-transfer, in-label-space]')

parser.add_argument('--trainN', default=None, type=int,
        help='N in train')
parser.add_argument('--N', default=18, type=int,
        help='N way')
parser.add_argument('--K', default=1, type=int,
        help='K shot')
parser.add_argument('--Q', default=1, type=int,
        help='Num of query per class')

parser.add_argument('--support_num', default=0, type=int,
        help='the id number of support set')

parser.add_argument('--zero_shot', action='store_true',
        help='')

parser.add_argument('--only_test', action='store_true',
        help='only test')

parser.add_argument('--load_ckpt', default=None,
        help='load ckpt')
parser.add_argument('--ckpt_name', type=str, default='',
        help='checkpoint name.')


# # Model Settings
parser.add_argument('--pretrain_ckpt', default='bert-base-uncased',
       help='bert pre-trained checkpoint: bert-base-uncased / bert-base-cased')

parser.add_argument('--prompt', default=1, type=int, choices=[0,1,2],
        help='choice in [0,1,2]:\
                0: Continue Prompt\
                1: Partition Prompt\
                2: Queue Prompt')
parser.add_argument('--pseudo_token', default='[S]', type=str,
        help='pseudo_token')

parser.add_argument('--max_length', default=64, type=int,
        help='max length')

parser.add_argument('--ignore_index', type=int, default=-1,
        help='label index to ignore when calculating loss and metrics')

parser.add_argument('--struct', action='store_true',
        help='StructShot parameter to re-normalizes the transition probabilities')

parser.add_argument('--tau', default=1, type=float,
        help='the temperature rate for contrastive learning')

parser.add_argument('--struct_tau', default=0.32, type=float,
        help='the tau in the viterbi decode')

# # Training Settings
parser.add_argument('--batch_size', default=16, type=int,
        help='batch size')
parser.add_argument('--test_bz', default=1, type=int,
        help='test or val batch size')

parser.add_argument('--train_iter', default=10000, type=int,
        help='num of iters in training')
parser.add_argument('--val_iter', default=200, type=int,
        help='num of iters in validation')
parser.add_argument('--test_iter', default=5000, type=int,
        help='num of iters in testing')
parser.add_argument('--val_step', default=200, type=int,
        help='val after training how many iters')

parser.add_argument('--adapt_step', default=5, type=int,
        help='adapting how many iters in validing or testing')
parser.add_argument('--adapt_auto', action='store_true',
        help='adapting how many iters in validing or testing')

parser.add_argument('--threshold_alpha', default=0.1, type=float,
        help='Gradient descent change threshold for early stopping')
parser.add_argument('--threshold_beta', default=0.5, type=float,
        help='loss threshold for early stopping')

parser.add_argument('--lr', default=1e-4, type=float,
        help='learning rate of Training')

parser.add_argument('--adapt_lr', default=None, type=float,
        help='learning rate of Adapting')

parser.add_argument('--grad_iter', default=1, type=int,
        help='accumulate gradient every x iterations')
parser.add_argument('--early_stopping', type=int, default=3000,
                    help='iteration numbers to stop without performance increasing')

parser.add_argument('--use_sgd_for_lm', action='store_true',
        help='use SGD instead of AdamW for BERT.')


opt = parser.parse_args()

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

import numpy as np
import torch
import random
from copy import Error
from transformers import BertTokenizer, BertConfig

from model.word_encoder import BERTWordEncoder
from model.copner import COPNER
from util.data_loader import get_loader
from util.framework import FewShotNERFramework
from util.word_mapping import FEWNERD_WORD_MAP, ONTONOTES_WORD_MAP, CONLL_WORD_MAP, WNUT_WORD_MAP, I2B2_WORD_MAP, MOVIES_WORD_MAP

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    trainN = opt.trainN if opt.trainN is not None else opt.N
    N = opt.N
    K = opt.K
    Q = opt.Q
    max_length = opt.max_length
    
    if opt.adapt_lr is None and opt.lr:
        opt.adapt_lr = opt.lr

    print("{}-way-{}-shot Few-Shot NER".format(N, K))
    print('task: {}'.format(opt.task))
    print('mode: {}'.format(opt.mode))
    print('prompt: {}'.format(opt.prompt))
    print("support: {}".format(opt.support_num))
    print("max_length: {}".format(max_length))
    print("batch_size: {}".format(opt.test_bz if opt.only_test else opt.batch_size))

    set_seed(opt.seed)
    print('loading model and tokenizer...')
    pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'

    config = BertConfig.from_pretrained(pretrain_ckpt)
    tokenizer = BertTokenizer.from_pretrained(pretrain_ckpt)
    opt.tokenizer = tokenizer
    word_encoder = BERTWordEncoder.from_pretrained(pretrain_ckpt, config=config, args=opt)
    

    print('loading data...')
    if opt.task == 'cross-label-space':
        opt.train = f'data/few-nerd/{opt.mode}/train.txt'
        opt.dev = f'data/few-nerd/{opt.mode}/test.txt'
        opt.test = f'data/few-nerd/{opt.mode}/test.txt'

        opt.train_word_map = opt.dev_word_map = opt.test_word_map = FEWNERD_WORD_MAP

        print(f'loading train data: {opt.train}')
        train_data_loader = get_loader(opt.train, tokenizer, word_map = opt.train_word_map,
                N=trainN, K=1, Q=Q, batch_size=opt.batch_size, max_length=max_length, # K=1 for training
                ignore_index=opt.ignore_index, args=opt, train=True)
        print(f'loading eval data: {opt.dev}')
        val_data_loader = get_loader(opt.dev, tokenizer, word_map = opt.dev_word_map,
                N=N, K=K, Q=Q, batch_size=opt.test_bz, max_length=max_length, 
                ignore_index=opt.ignore_index, args=opt)
        print(f'loading test data: {opt.test}')
        test_data_loader = get_loader(opt.test, tokenizer, word_map = opt.test_word_map,
                N=N, K=K, Q=Q, batch_size=opt.test_bz, max_length=max_length, 
                ignore_index=opt.ignore_index, args=opt)

    elif opt.task == 'domain-transfer':
        opt.train = 'data/ontonotes/train.txt'
        opt.support = f'data/domain/{opt.mode}/support-{K}shot/{opt.support_num}.txt'
        opt.test = f'data/domain/{opt.mode}/test.txt'
        opt.train_word_map = ONTONOTES_WORD_MAP
        if opt.mode == 'conll':
            opt.dev_word_map = opt.test_word_map = CONLL_WORD_MAP
        elif opt.mode == 'wnut':
            opt.dev_word_map = opt.test_word_map = WNUT_WORD_MAP
        elif opt.mode == 'i2b2':
            opt.dev_word_map = opt.test_word_map = I2B2_WORD_MAP
        
        print(f'loading train data: {opt.train}')
        train_data_loader = get_loader(opt.train, tokenizer, word_map = opt.train_word_map,
                N=trainN, K=K, Q=Q, batch_size=opt.batch_size, max_length=max_length, 
                ignore_index=opt.ignore_index, args=opt, train=True)
        print(f'loading val data: {opt.test}')
        val_data_loader = get_loader(opt.test, tokenizer, word_map = opt.dev_word_map,
                N=N, K=K, Q=Q, batch_size=opt.test_bz, max_length=max_length, 
                ignore_index=opt.ignore_index, args=opt, support_file_path=opt.support)
        print(f'loading test data: {opt.test}')
        test_data_loader = get_loader(opt.test, tokenizer, word_map = opt.test_word_map,
                N=N, K=K, Q=Q, batch_size=opt.test_bz, max_length=max_length, 
                ignore_index=opt.ignore_index, args=opt, support_file_path=opt.support)

    elif opt.task == 'in-label-space':
        opt.train = f'data/supervised/{opt.mode}/{K}shot/{opt.support_num}.txt'
        opt.test = f'data/supervised/{opt.mode}/test.txt'

        if opt.mode == 'conll':
            opt.train_word_map = opt.dev_word_map = opt.test_word_map = CONLL_WORD_MAP
        if opt.mode == 'ontonotes':
            opt.train_word_map = opt.dev_word_map = opt.test_word_map = ONTONOTES_WORD_MAP
        elif opt.mode == 'mit-movie':
            opt.train_word_map = opt.dev_word_map = opt.test_word_map = MOVIES_WORD_MAP

        print(f'loading support data: {opt.train}')
        train_data_loader = get_loader(
                        opt.train, tokenizer, word_map = opt.test_word_map,
                        N=trainN, K=K, Q=Q, batch_size=opt.batch_size, max_length=max_length, 
                        ignore_index=opt.ignore_index, args=opt, train=True)
        print(f'loading test data: {opt.test}')
        val_data_loader = test_data_loader = get_loader(
                        opt.test, tokenizer, word_map = opt.test_word_map,
                        N=N, K=K, Q=Q, batch_size=opt.test_bz, max_length=max_length, 
                        ignore_index=opt.ignore_index, args=opt)
    else:
        raise Error("not Implement !!!")

    prefix = '-'.join([opt.task, opt.mode, str(N), str(K), 'seed'+str(opt.seed)])

    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name
    
    model = COPNER(word_encoder, opt, opt.train_word_map if not opt.only_test else opt.test_word_map)

    framework = FewShotNERFramework(opt, train_data_loader, val_data_loader, test_data_loader,
                                        train_fname=opt.train if opt.struct else None, 
                                        viterbi=True if opt.struct else False)

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pt'.format(prefix)

    i = 1
    while os.path.exists(ckpt):
        ckpt = 'checkpoint/{}-{}.pt'.format(prefix, i)
        i += 1

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        print('model-save-path:', ckpt)

        framework.train(model, prefix,
                load_ckpt=opt.load_ckpt, 
                save_ckpt=ckpt,
                val_step=opt.val_step, 
                train_iter=opt.train_iter, 
                warmup_step=int(opt.train_iter * 0.05), 
                val_iter=opt.val_iter, 
                learning_rate=opt.lr, 
                use_sgd_for_lm=opt.use_sgd_for_lm)
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'

#     if not opt.only_test:
#         framework.args.threshold_beta = opt.batch_size
#         framework.args.adapt_auto = True
    precision, recall, f1, fp, fn, within, outer = framework.eval(model, opt.test_iter, ckpt=ckpt, word_map = opt.test_word_map)
    print("RESULT: f1:%.4f, precision: %.4f, recall: %.4f" % (f1, precision, recall))
    print('ERROR ANALYSIS: fp: %.4f, fn: %.4f, within:%.4f, outer: %.4f'%(fp, fn, within, outer))

if __name__ == "__main__":
    main()
