import os
import random
from collections import OrderedDict
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

from .fewshotsampler import SingleFewshotSampler, PairFewshotSampler

def batch_convert_ids_to_tensors(batch_token_ids) -> torch.Tensor:

    bz = len(batch_token_ids)
    batch_tensors = [torch.LongTensor(batch_token_ids[i]).squeeze(0) for i in range(bz)]
    batch_tensors = pad_sequence(batch_tensors, True, padding_value=0).long()
    return batch_tensors

def get_class_words(rawtag, word_map):
    if rawtag.startswith('B-') or rawtag.startswith('I-'):
        return word_map[rawtag[2:]]
    else:
        return word_map[rawtag]

class Sample:
    def __init__(self, filelines, word_map):
        self.word_map = word_map

        filelines = [line.split('\t') for line in filelines]
        self.words, self.tags = zip(*filelines)
        self.words = [word.lower() for word in self.words]
        
        # strip B-, I-
        self.normalized_tags = [get_class_words(tag, word_map) for tag in self.tags]
        self.class_count = {}

    def __count_entities__(self):
        current_tag = self.normalized_tags[0]
        for tag in self.normalized_tags[1:]:
            if tag == current_tag:
                continue
            else:
                if current_tag != self.word_map['O']:
                    if current_tag in self.class_count:
                        self.class_count[current_tag] += 1
                    else:
                        self.class_count[current_tag] = 1
                current_tag = tag
        if current_tag != self.word_map['O']:
            if current_tag in self.class_count:
                self.class_count[current_tag] += 1
            else:
                self.class_count[current_tag] = 1

    def get_class_count(self):
        if self.class_count:
            return self.class_count
        else:
            self.__count_entities__()
            return self.class_count

    def get_tag_class(self):
        # strip 'B' 'I' 
        tag_class = list(set(self.normalized_tags))
        if self.word_map['O'] in tag_class:
            tag_class.remove(self.word_map['O'])
        return tag_class

    def valid(self, target_classes):
        return (set(self.get_class_count().keys()).intersection(set(target_classes))) and not (set(self.get_class_count().keys()).difference(set(target_classes)))

    def __str__(self):
        newlines = zip(self.words, self.tags)
        return '\n'.join(['\t'.join(line) for line in newlines])

class PairDatasetwithEpisodeSample(data.Dataset):
    """
    Fewshot NER Dataset with episode sampling, return support set and query set
    """
    def __init__(self, N, K, Q, filepath, tokenizer, max_length, word_map, ignore_label_id=-1, args=None):
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.class2sampleid = {}
        self.word_map = word_map
        self.word2class = OrderedDict()
        for key, value in self.word_map.items():
            self.word2class[value] = key

        self.BOS = '[CLS]'
        self.EOS = '[SEP]'

        self.max_length = max_length
        self.ignore_label_id = ignore_label_id

        self.samples, self.classes = self.__load_data_from_file__(filepath)
        self.sampler = PairFewshotSampler(N, K, Q, self.samples, classes=self.classes)

        self.prompt = args.prompt
        self.tokenizer = tokenizer
        self.pseudo_token = args.pseudo_token
        self.tokenizer.add_special_tokens({'additional_special_tokens': [args.pseudo_token]})


    def __insert_sample__(self, index, sample_classes):
        for item in sample_classes:
            if item in self.class2sampleid:
                self.class2sampleid[item].append(index)
            else:
                self.class2sampleid[item] = [index]
    
    def __load_data_from_file__(self, filepath):
        samples = []
        classes = []
        with open(filepath, 'r', encoding='utf-8')as f:
            lines = f.readlines()
        samplelines = []
        index = 0
        for line in lines:
            line = line.strip()
            if len(line.split('\t'))>1:
                samplelines.append(line)
            else:
                sample = Sample(samplelines, self.word_map)
                samples.append(sample)
                sample_classes = sample.get_tag_class()
                self.__insert_sample__(index, sample_classes)
                classes += sample_classes
                samplelines = []
                index += 1
        classes = list(set(classes))
        return samples, classes

    def __get_token_label_list__(self, words, tags):
        tokens = []
        valid_masks = []
        labels = []
        for word, tag in zip(words, tags):
            word_token = self.tokenizer.tokenize(word)
            if word_token:
                tokens.extend(word_token)
                # tokenize the label to token id and make it the same number of tokens as the original words
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label = [self.tag2label[tag]] + [self.ignore_label_id] * (len(word_token) - 1)
                labels.extend(label)
                mask = [1] + [self.ignore_label_id] * (len(word_token) - 1)
                # mask = [1] * len(word_token)
                valid_masks.extend(mask)

        return tokens, labels, valid_masks

    def _get_prompt(self, classes):
        # prompt 0
        if self.prompt in [0, 1]:
            prompt = [self.pseudo_token]
            for i in range(len(classes)):
                prompt += [classes[i]] + [self.pseudo_token]
        # prompt 2
        elif self.prompt == 2:
            prompt = [iclass for iclass in classes]

        return prompt

    def __getraw__(self, 
                    tokens, 
                    labels, 
                    valid_masks,
                    prompt_tags):
        # get tokenized word list, attention mask, text mask (mask [CLS], [SEP] as well), tags
        
        # split into chunks of length (max_length-2)
        # 2 is for special tokens [CLS] and [SEP] or <s> and </s>

        o_tokens_list = []
        o_labels_list = []
        o_valid_masks = []

        while len(tokens) > self.max_length - 2:
            o_tokens_list.append(tokens[:self.max_length-2])
            o_labels_list.append(labels[:self.max_length-2])
            o_valid_masks.append(valid_masks[:self.max_length-2])
            tokens = tokens[self.max_length-2:]
            labels = labels[self.max_length-2:]
            valid_masks = valid_masks[self.max_length-2:]
        if len(tokens) > 1:
            o_tokens_list.append(tokens)
            o_labels_list.append(labels)
            o_valid_masks.append(valid_masks)

        # add special tokens and get masks
        input_ids_list = []
        labels_list = []
        valid_masks_list = []
        for i, tokens in enumerate(o_tokens_list):
            assert len(o_labels_list[i]) == len(tokens) == len(o_valid_masks[i]), \
                    print(labels_list[i], tokens, o_valid_masks[i])
            
            tokens = [self.BOS] + tokens + [self.EOS]
            tokens += self._get_prompt(prompt_tags) + [self.EOS]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            labels = [self.ignore_label_id] + o_labels_list[i] + [self.ignore_label_id]
            labels += [self.ignore_label_id] * (len(tokens) - len(labels))

            
            valid_masks = [self.ignore_label_id] + o_valid_masks[i] + [self.ignore_label_id]
            valid_masks += [self.ignore_label_id] * (len(tokens) - len(valid_masks))

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            valid_masks_list.append(valid_masks)
            
        return input_ids_list, labels_list, valid_masks_list

    def __additem__(self, d, inputs, labels, valid_masks):
        d['inputs'] += inputs
        d['labels'] += labels
        d['valid_masks'] += valid_masks

    def __populate__(self, idx_list, target_classes, prompt_tags, savelabeldic=False):
        '''
        populate samples into data dict
        set savelabeldic=True if you want to save label2tag dict
        'index': sample_index
        'word': tokenized word ids
        'mask': attention mask in BERT
        'label': NER labels
        'sentence_num': number of sentences in this set (a batch contains multiple sets)
        'text_mask': 0 for special tokens and paddings, 1 for real text
        '''
        dataset = {'inputs': [], 'labels': [], 'valid_masks': []}
        for idx in idx_list:
            tokens, labels, valid_masks = self.__get_token_label_list__(self.samples[idx].words, self.samples[idx].normalized_tags)
            input_ids, labels, valid_masks = self.__getraw__(tokens, labels, valid_masks, prompt_tags)
            self.__additem__(dataset, input_ids, labels, valid_masks)
        dataset['sentence_num'] = [len(dataset['inputs'])]
        dataset['target_classes'] = target_classes
        if savelabeldic:
            dataset['label2tag'] = [self.label2tag]
        return dataset

    def __getitem__(self, _):
        target_classes, support_idx, query_idx = self.sampler.__next__()
        # add 'none' and make sure 'none' is labeled 0
        distinct_tags = [self.word_map['O']] + target_classes
        prompt_tags = distinct_tags.copy()
        random.shuffle(prompt_tags)
        self.tag2label = {tag:idx for idx, tag in enumerate(distinct_tags)}
        self.label2tag = {idx:self.word2class[tag] for idx, tag in enumerate(distinct_tags)}
        support_set = self.__populate__(support_idx, distinct_tags, prompt_tags)
        query_set = self.__populate__(query_idx, distinct_tags, prompt_tags, savelabeldic=True)
        return support_set, query_set
    
    def __len__(self):
        return 1000000

class SingleDatasetwithEpisodeSample(PairDatasetwithEpisodeSample):

    def __init__(self, N, K, filepath, tokenizer, max_length, word_map, ignore_label_id=-1, args=None):
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.class2sampleid = {}
        self.word_map = word_map
        self.word2class = OrderedDict()
        for key, value in self.word_map.items():
            self.word2class[value] = key

        self.BOS = '[CLS]'
        self.EOS = '[SEP]'

        self.max_length = max_length
        self.ignore_label_id = ignore_label_id

        self.samples, self.classes = self.__load_data_from_file__(filepath)
        
        self.sampler = SingleFewshotSampler(N, K, self.samples, classes=self.classes)

        self.prompt = args.prompt
        self.tokenizer = tokenizer
        self.pseudo_token = args.pseudo_token
        self.tokenizer.add_special_tokens({'additional_special_tokens': [args.pseudo_token]})


    def __getitem__(self, index):
        target_classes, support_idx = self.sampler.__next__()
        # add 'none' and make sure 'none' is labeled 0
        distinct_tags = [self.word_map['O']] + target_classes
        prompt_tags = distinct_tags.copy()
        random.shuffle(prompt_tags)
        self.tag2label = {tag:idx for idx, tag in enumerate(distinct_tags)}
        self.label2tag = {idx:self.word2class[tag] for idx, tag in enumerate(distinct_tags)}
        support_set = self.__populate__(support_idx, distinct_tags, prompt_tags, savelabeldic=True)

        return support_set
    
    def __len__(self):
        return 1000000

class PairDatasetwithFixedSupport(PairDatasetwithEpisodeSample):
    '''
    Few-shot NER dataset with fixed support set, return support set and query set
    '''

    def __init__(self, N, filepath, 
                        support_file_path,
                        tokenizer, 
                        max_length, 
                        word_map, 
                        ignore_label_id=-1, 
                        args=None) -> None:
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert(0)
        
        if not os.path.exists(support_file_path):
            print("[ERROR] Support Data file does not exist!")
            assert(0)

        self.ignore_label_id = ignore_label_id
        self.class2sampleid = {}
        self.word_map = word_map
        self.word2class = OrderedDict()
        for key, value in self.word_map.items():
            self.word2class[value] = key

        self.BOS = '[CLS]'
        self.EOS = '[SEP]'

        self.data_size = N
        self.max_length = max_length

        self.prompt = args.prompt
        self.tokenizer = tokenizer
        self.pseudo_token = args.pseudo_token
        self.tokenizer.add_special_tokens({'additional_special_tokens': [args.pseudo_token]})

        self.support_samples, target_classes = self.__load_data_from_file__(support_file_path)
        self.query_samples, _ = self.__load_data_from_file__(filepath)

        target_classes = [value for key, value in self.word_map.items() if key != 'O']
        self.distinct_tags = [self.word_map['O']] + target_classes 
        
        self.tag2label = {tag:idx for idx, tag in enumerate(self.distinct_tags)}
        self.label2tag = {idx:self.word2class[tag] for idx, tag in enumerate(self.distinct_tags)}
        

    def __generate_support_set__(self, prompt_tags):
        '''
        populate samples into data dict
        'index': sample_index
        'word': tokenized word ids
        'mask': attention mask in BERT
        'label': NER labels
        'sentence_num': number of sentences in this set (a batch contains multiple sets)
        'text_mask': 0 for special tokens and paddings, 1 for real text
        '''
        dataset = {'inputs': [], 'labels': [], 'valid_masks': []}
        for sample in self.support_samples:
            tokens, labels, valid_masks = self.__get_token_label_list__(sample.words, sample.normalized_tags)
            input_ids, labels, valid_masks = self.__getraw__(tokens, labels, valid_masks, prompt_tags)
            self.__additem__(dataset, input_ids, labels, valid_masks)
        dataset['sentence_num'] = [len(dataset['inputs'])]
        dataset['target_classes'] = self.distinct_tags
        return dataset

    def __getitem__(self, index):
        prompt_tags = self.distinct_tags.copy()
        random.shuffle(prompt_tags)

        support_set = self.__generate_support_set__(prompt_tags)
        query_idx = range(index * self.data_size, (index + 1) * self.data_size)
        query_set = self.__populate__(query_idx, self.distinct_tags, prompt_tags, savelabeldic=True)
        return support_set, query_set
    
    def __len__(self):
        return len(self.samples) // self.data_size

class SingleDatasetwithRamdonSample(PairDatasetwithEpisodeSample):
    '''
    Simple NER dataset with random sample
    '''
    def __init__(self, filepath, 
                        tokenizer, 
                        max_length, 
                        word_map, 
                        ignore_label_id=-1, 
                        args=None) -> None:
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert(0)

        self.ignore_label_id = ignore_label_id
        self.class2sampleid = {}
        self.word_map = word_map
        self.word2class = OrderedDict()
        for key, value in self.word_map.items():
            self.word2class[value] = key

        self.BOS = '[CLS]'
        self.EOS = '[SEP]'

        self.max_length = max_length

        self.samples, self.classes = self.__load_data_from_file__(filepath)
        self.distinct_tags = [self.word_map['O']] + self.classes

        self.prompt = args.prompt
        self.tokenizer = tokenizer
        self.pseudo_token = args.pseudo_token
        if self.tokenizer:
            self.tokenizer.add_special_tokens({'additional_special_tokens': [args.pseudo_token]})

    def __getitem__(self, index):

        prompt_tags = self.distinct_tags.copy()
        random.shuffle(prompt_tags)
        self.tag2label = {tag:idx for idx, tag in enumerate(self.distinct_tags)}
        self.label2tag = {idx:self.word2class[tag] for idx, tag in enumerate(self.distinct_tags)}
        item = self.__populate__([index], self.distinct_tags, prompt_tags, savelabeldic=True)
        return item
    
    def __len__(self):
        return len(self.samples)

def batch_data_convertor(batch_dict, data):
    for i in range(len(data)):
        for k in batch_dict:
            if k == 'target_classes':
                batch_dict[k].append(data[i][k])
            else:
                batch_dict[k] += data[i][k]

    batch_keys = list(batch_dict.keys())
    for k in batch_keys:
        if k in ['inputs', 'valid_masks']:
            batch_dict[k] = batch_convert_ids_to_tensors(batch_dict[k])
        elif k == 'labels':
            batch_dict['batch_labels'] = batch_convert_ids_to_tensors(batch_dict[k])
            batch_dict[k] = [torch.LongTensor(item) for item in batch_dict[k]]

    return batch_dict

def pair_collate_fn(data):
    batch_support = {'inputs': [],
                    'labels':[], 
                    'valid_masks': [], 
                    'sentence_num': [], 
                    'target_classes':[]}
    batch_query = {'inputs': [],
                    'labels':[], 
                    'valid_masks': [], 
                    'sentence_num': [], 
                    'target_classes':[], 
                    'label2tag': []}
    support_sets, query_sets = zip(*data)
    
    batch_support = batch_data_convertor(batch_support, support_sets)
    batch_query = batch_data_convertor(batch_query, query_sets)

    return batch_support, batch_query

def single_collate_fn(data):
    
    batch = {'inputs': [],
                    'labels':[], 
                    'valid_masks': [], 
                    'sentence_num': [], 
                    'target_classes':[], 
                    'label2tag': []}
    
    batch = batch_data_convertor(batch, data)

    return batch

def get_loader(filepath, tokenizer, N, K, Q, batch_size, max_length, word_map,
        ignore_index=-1, args=None, num_workers=4, support_file_path=None, train=False):
    if train:
        dataset = SingleDatasetwithEpisodeSample(N, 1, filepath, tokenizer, max_length, 
                                                        ignore_label_id=ignore_index, 
                                                        args=args, word_map=word_map)
        return data.DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=num_workers,
                                collate_fn=single_collate_fn)
    else:
        if args.task in ['cross-label-space']:
            dataset = PairDatasetwithEpisodeSample(N, K, Q, filepath, tokenizer, max_length, 
                                                        ignore_label_id=ignore_index, 
                                                        args=args, word_map=word_map)
            return data.DataLoader(dataset=dataset,
                                    batch_size=1,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=pair_collate_fn)
        elif args.task in ['domain-transfer']:
            dataset = PairDatasetwithFixedSupport(N, filepath, support_file_path, tokenizer, max_length,
                                                        ignore_label_id=ignore_index,
                                                        args=args, word_map=word_map)
            return data.DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=pair_collate_fn)
        elif args.task in ['in-label-space']:
            dataset = SingleDatasetwithRamdonSample(filepath, tokenizer, max_length, 
                                                        ignore_label_id=ignore_index, 
                                                        args=args, word_map=word_map)
        
            return data.DataLoader(dataset=dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        pin_memory=True,
                                        num_workers=num_workers,
                                        collate_fn=single_collate_fn)