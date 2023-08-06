import pdb
import copy
import json
import torch
import random
import numpy as np
from tabulate import tabulate

from math import log
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset


def log2(x, base):
    return int(log(x) / log(base))
 
# Compute power of two greater than or equal to `n`
def findNextPowerOf2(n):
    # decrement `n` (to handle the case when `n` itself
    # is a power of 2)
    n = n - 1
    # calculate the position of the last set bit of `n`
    lg = log2(n, 2)
    # next power of two will have a bit set at position `lg+1`.
    return 1 << lg + 1
 

class Span2conll():
    def __init__(self, visualize=True, max_depth=8, SEP='[SEP]'):
        self.max_depth = max_depth
        self.visualize = visualize
        self.SEP = '@'
    
    def __call__(self, tokens, labels):
        """
        inputs : 
            tokens is a list of word
            labels is a list of tuple(start, end, tag)
            
        output :
            a sentence in conll format
        """
        output = self.span2conll(tokens, labels)
        return output
    
    def index_in_span(self, idx, entity_list, mode='start'):
    
        # Check mode
        if   mode=='start': mode=0
        elif mode=='end':   mode=1
        else:
            raise "Check mode"

        ## get all idx entities
        idx_entity_list = [p[mode] for p in entity_list]

        ## Get index of entity list that start with idx
        idx_entities =  np.where(np.array(idx_entity_list) == idx)[0]

        # There is not entity in the list
        if len(idx_entities) == 0:
            return False

        ## Return list of entities that start with the idx
        return [ entity for idx, entity in enumerate(entity_list) 
                if idx in idx_entities]
    

    def span2conll(self, words, labels):
        max_token=self.max_depth
        result_conll  = []
        entity_queue  = []
        entity_counts = 0

        # Sorted labels
        labels = sorted(labels, key=lambda x:(x[0], -x[1]))

        # mask entities
        labels = [(e[0], e[1], f"{str(idx+1)}{self.SEP}{e[2]}") 
                          for idx, e in enumerate(labels)]
        # Re-format to conll
        for idx, word in enumerate(words):
            # Push : when idx in start span of label --> sorted entity queue
            # Show : every words
            # Pop  : when idx in end span of label
            ## Push entity in entities queue
            # Check new entities from current idx.
            start_entities = self.index_in_span(idx, labels)
            # There are new entities.
            if start_entities :
                # Push new entities
                entity_queue.extend(start_entities)
                # Sort entities in the queue by (min_start_idx, max_stop)
                # print(entity_queue)
                entity_queue = sorted(entity_queue, key=lambda x:(x[0], -x[1]))

            ## Pop entity out of entity equeue
            end_entities = self.index_in_span(idx, labels, 'end')
            # print('\n',end_entities)
            # print(entity_queue)

            # Pop the entities from entities queue
            if end_entities:
                entity_queue = [end_en for end_en in entity_queue 
                                if end_en not in end_entities]
            # Keep result
            temp_result = [ x[-1] for x in entity_queue]
            # temp_result = [ x[-1].split(self.SEP)[-1] for x in entity_queue]
            temp_result+=['O']*(max_token-len(temp_result))
            result_conll.append([word]+temp_result)
            # result_conll.append(temp_result)
            ## Show ###
            if self.visualize:
                # print word
                token = word[0:max_token] \
                        if len(word) <= max_token \
                        else word[0:max_token]+'...' 
                print(f"\n{idx:<3} {token:<15} \t", end=' ')
                # print label
                for label in entity_queue:
                    label = label[-1]
                    label = label[0:max_token+3] \
                            if len(label) <= max_token \
                            else label[0:max_token]+'...' 
                    print(f"{label:<15}", end=' ')
        # Remove entities that stop at the last index
        entity_queue = [e for e in entity_queue if e[1]!=idx+1]
        if len(entity_queue) != 0:
            pdb.set_trace()
        return result_conll

    @staticmethod
    def processed_entities(label):
        temp_label = []
        for index in range(len(label)):
            _start, _end = label[index]['span']
            _tag = label[index]['entity_type']
            temp = (_start, _end, _tag)
            temp_label.append(temp)
        return temp_label
    
    @staticmethod
    def tag_bio(entities):
        num_tokens = len(entities)
        results = []
        for index in range(num_tokens):
            tag = entities[index]
            if tag=="O": 
                results.append(tag)
            elif tag!=entities[index-1]:
                tag = tag.split("@")[-1]
                results.append(f"B-{tag}")
            elif tag!="O":
                tag = tag.split("@")[-1]
                results.append(f"I-{tag}")
            else:
                raise "Error"
        return results


def is_same_NE(NE1, NE2):
    state=0
    if NE1[0]=='B' and NE2[0]=='I':
        state=1
    if NE1[0]=='B' and NE2[0]=='E':
        state=1
    if NE1[0]=='I' and NE2[0]=='I':
        state=1
    if NE1[0]=='I' and NE2[0]=='E':
        state=1
    if state==1 and NE1[1:] == NE2[1:]:
        return True
    else:
        return False


def is_S(prev, now, next):
    prev2now = is_same_NE(prev, now)
    now2next = is_same_NE(now, next)
    if prev2now==False:
        if prev2now == now2next:
            if now[0]!='O':
                return True
    else:
        return False

def is_B(prev, now, next):
    prev2now = is_same_NE(prev, now)
    now2next = is_same_NE(now, next)
    if prev2now==False:
        if now2next==True:
            return True
    return False

def is_I(prev, now, next):
    prev2now = is_same_NE(prev, now)
    now2next = is_same_NE(now, next)
    if prev2now==True:
        if now2next==True:
            return True
    return False

def is_E(prev, now, next):
    prev2now = is_same_NE(prev, now)
    now2next = is_same_NE(now, next)
    if prev2now==True:
        if now2next==False:
            return True
    return False

def is_O(prev, now, next):
    if now[0]=='O':
        return True
    return False


def fix_labels(labels, tag_type, visualize=False):
    prev = "O"
    fixed_labels = []
    length = len(labels)
    for index in range(length):
        change_state=0
        now = labels[index]
        next = labels[index+1] if index < length-1 else "O"

        if is_O(prev, now, next):
            now="O"
            # if visualize:
            #     print("O", end='')

        elif is_S(prev, now, next):
            if set(tag_type)==set("BIO"):
                now="B"+now[1:]
                if visualize:
                    print("B", end='')

            elif set(tag_type)==set("BIOES"):
                now="S"+now[1:]
                if visualize:
                    print("S", end='')
            else:
                raise "Error tag_type"

            change_state+=1
        
        if is_B(prev, now, next):
            now="B"+now[1:]
            change_state+=1
            if visualize:
                print("B", end='')
        
        if is_I(prev, now, next):
            now="I"+now[1:]
            change_state+=1
            if visualize:
                print("I", end='')

        if is_E(prev, now, next):
            if set(tag_type)==set("BIOES"):
                now="E"+now[1:]
                if visualize:
                    print("E", end='')

            elif set(tag_type)==set("BIO"):
                now="I"+now[1:]
                if visualize:
                    print("I", end='')
            else:
                raise "Error tag_type"
            change_state+=1
        prev=now
        fixed_labels.append(now)

        #Check duble change
        if change_state>1:
            print("Duble change")
            breakpoint()

        # Visualize
        if visualize and change_state>=1 and labels[index]!=now:
            print(f"\t{labels[index]} -> {now}\t\t {labels[index-1], labels[index], next}")
        else:
            pass
    return fixed_labels


def remove_incorrect_tag(labels, tag_type):
    assert set(tag_type)==set("BIOES")
    prev = "O"
    count = 0
    state = False
    fixed_labels = []
    length = len(labels)
    START = 0; END = 0
    correct_entity = False
    for index in range(length):
        now = labels[index]
        next = labels[index+1] if index < length-1 else "O"
    
        # Reset
        prev2now = is_same_NE(prev, now)
        if prev2now: 
            pass
        else: 
            state = False
    
        if state:
            if now[0]=="E":
                END=index
                state = False
                correct_entity=True
        else:
            # Set
            if now[0] in ['B']:
                count+= 1
                state = True
                START = index

        prev=now
        if correct_entity:
            fixed_labels.extend(list(range(START, END+1)))
            correct_entity=False 
        
        if now[0]=='S':
            fixed_labels.extend([index])
        
        # if now[0]=='B' and is_same_NE(now, next)==False:
        #     fixed_labels.extend([index])
            
    results = []
    for ids, tag in enumerate(labels):
        if ids in fixed_labels:
            results.append(tag)
        else:
            results.append("O")
    return results


class CONLLLabels():
    def __init__(self, max_layers, boundary_type, debug):
        self.max_layers = max_layers
        self.boundary_type = boundary_type
        self.span_conll = Span2conll(
            visualize=debug, 
            max_depth=self.max_layers)
    
    def __call__(self, tokens, entities):
        labels = self.span_conll.processed_entities(entities)
        labels = self.span_conll(tokens, labels)
        labels = np.array(labels)
        targets = [list(labels[:,i+1]) for i in range(self.max_layers)]
        
        temp_target = {}
        for layer, label in enumerate(targets):
            
            # BIO tagging
            boundary_tag = self.span_conll.tag_bio(label)
            
            # BIOES tagging
            if set(self.boundary_type)==set("BIOES"):
                boundary_tag = fix_labels(boundary_tag,"BIOES")
                
            # Keep each layer
            temp_target[layer]=boundary_tag
            
        return temp_target
    
    def map_srt_ids(self, data, dict_data, unk):
        resutls = []
        for d in data:
            if d in dict_data:
                resutls.append(dict_data.get(d))
            else:
                resutls.append(dict_data.get(unk))
        return resutls
    

class InputLM():
    def __init__(self, path_lm, max_length) -> None:
        self.path_lm = path_lm
        self.max_length = max_length
        self.lm_name = path_lm.split('/')[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(path_lm)
        
        ## Add special token
        self.end_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.word2ids = self.tokenizer.get_vocab()
        self.start_id = self.tokenizer.bos_token_id


    def __call__(self, tokens, entities):
        tokens = tokens.copy()
        entities = entities.copy()  
        input_ids, attention_mask, encode_dict = self._tokenizer_input(tokens)
        shifted_entities = self._shifted_entities_index(input_ids, entities, encode_dict)
        lm_tokens=[self.tokenizer.decode(w) for w in input_ids]

        item = {}
        item['lm_tokens'] = lm_tokens
        item['input_ids'] = input_ids
        item['lm_encode_dict'] = encode_dict
        item['lm_entities'] = shifted_entities
        item['attention_mask'] = attention_mask
        return item

    def _tokenizer_input(self, tokens):
        max_length = self.max_length
        encode_dict = {}
        input_ids = [self.start_id]
        for index in range(len(tokens)):
            word = tokens[index]
            shifted = len(input_ids)
            # tokenize each word
            ids = self.tokenizer.encode(word)
            ids = ids[1:-1]
            input_ids.extend(ids)
            encode_dict[index] = (shifted, shifted+len(ids))

        # Add end of word
        input_ids.append(self.end_id)

        # Create mask
        num_ids = len(input_ids)
        mask = [1]*num_ids
        mask+= [0]*(max_length-num_ids)
        assert len(mask)==max_length, 'Error create mask'

        # Add padding
        input_ids+=[self.pad_id] * (max_length-num_ids)

        return input_ids, mask, encode_dict


    def _shifted_entities_index(self, input_ids, entities, encode_dict):
        
        shifted_entities = []
        # Shift labels index
        for index in range(len(entities)):
            entity = entities[index]
            entity_type = entity['entity_type']
            start, end = entity['span']
            text = entity['text']

            # shifting start, end
            (shifted_start, _) = encode_dict.get(start)
            (_, shifted_end) = encode_dict.get(end-1)

            decode_text = input_ids[shifted_start:shifted_end]
            decode_text = [self.tokenizer.decode(w) for w in decode_text]
            decode_text = decode_text
            
            shifted_entities.append({
                'entity_type':entity_type,
                'span':[shifted_start, shifted_end],
                'text': decode_text
            })
            
        return shifted_entities

    @staticmethod
    def check_entities(sample):
        temp = [['original_en', 'orginal_span', 'decode_en', 'decode_span']]
        for index in range(len(sample['org_entities'])):
            org_entity = sample['org_entities'][index]
            original_ne = org_entity['text']
            original_span = org_entity['span']
            decode_entity = sample['entities'][index]
            decode_ne =decode_entity['text']
            decode_span = decode_entity['span']
            temp.append([original_ne, 
                        original_span, 
                        decode_ne, 
                        decode_span])
                        
        print(tabulate(temp))

    @staticmethod
    def check_input_ids_and_mask(sample):
        temp = [['index', 'input_text', 'input_ids', 'mask']]
        for index in range(len(sample['input_ids'])):
            
            original_ids = sample['input_ids'][index]
            mask = sample['mask'][index]
            input_text = sample['input_text'][index]
            
            temp.append([index, input_text, original_ids, mask])
        print(tabulate(temp))


class NERDataset(Dataset):
    def __init__(self,
     data,
     input_lm,
     conll_labels,
     sent_length,
     word2ids,
     span2ids,
     ne2ids,
     spantag2ids,
     tag2ids,
     use_lm,
     unk="<unk>",
     pad="<pad>",
     get_org_text=False
    ):
        '''
        - root_dir = root directory of the dataset 
        '''
        random.seed(123)    
        np.random.seed(123)
        self.unk = unk
        self.pad = pad
        self.data = data
        self.ne2ids = ne2ids
        self.use_lm = use_lm
        self.tag2ids = tag2ids
        self.input_lm = input_lm
        self.word2ids = word2ids
        self.span2ids = span2ids
        self.spantag2ids = spantag2ids
        self.sent_length = sent_length
        self.conll_labels = conll_labels
        self.get_org_text = get_org_text

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        instance = self.data[idx]
        temp_features = {}
        item = copy.deepcopy(instance)
        tokens=item['tokens']
        entities=item['entities']
        temp_features['org_tokens']=tokens
        # temp_features['org_entities']=entities
        temp_features['sentence_id']=item['sentence_id']

        # Preprocessing tokens and entities
        lm_item = self.input_lm(tokens, entities)
        input_ids = lm_item['input_ids']
        mask = lm_item['attention_mask']
        
        # Change tokens and entities for generate conll labels
        tokens = lm_item['lm_tokens']
        entities = lm_item['lm_entities']
        # temp_features['encode_dict'] = lm_item['lm_encode_dict']

        # Input embedding
        temp_features['mask']= mask
        temp_features['tokens']="|".join(tokens)
        temp_features['input_ids']= input_ids
        
        temp_features['entities'] = [
            (ne['span'][0], ne['span'][1], self.tag2ids[ne["entity_type"]]) 
            for ne in entities]

        nested_conll = self.conll_labels(tokens, entities)
        flat_conll = nested_conll[0]
        temp_features['flat_conll'] = flat_conll

        flat_conll_ids = self.conll_labels.map_srt_ids(
            nested_conll[0], self.spantag2ids, unk="O")
        temp_features['flat_conll_ids'] = flat_conll_ids

        # Span and type labels
        span = [item['span'] for item in entities]
        type = [item['entity_type'] for item in entities]
        span_conll = [t.split('-')[0]+"-NE" if t!="O" else t for t in flat_conll]
        span_conll_ids = [self.span2ids[x] for x in span_conll]
        temp_features['span'] = span
        temp_features['span_conll'] = span_conll
        temp_features['span_conll_ids'] = span_conll_ids
        return temp_features


    def padding_tokens(self, tokens, max_length, pad):
        num_tokens = len(tokens)
        num_padding = max_length-num_tokens
        if num_padding>=0:
            padding = [pad]*num_padding
        if num_padding<0:
            raise TypeError(f"Exceed max length tokens: {num_tokens}")
            
        mask = [1]*num_tokens + [0]*num_padding
        return tokens+padding, mask

    def map_srt_ids(self, data, dict_data, unk):
            resutls = []
            for d in data:
                if d in dict_data:
                    resutls.append(dict_data.get(d))
                else:
                    resutls.append(dict_data.get(unk))
            return resutls


class Loader():    
    def __init__(self, 
        path_data, 
        batch_size,
        boundary_type="BIOES", 
        max_layers=1,
        sent_length=256,
        path_lm=None,
        debug=False,
        sample_data=False
        ):

        self.debug = debug
        self.path_lm = path_lm
        self.path_data = path_data
        self.batch_size = batch_size
        self.max_layers = max_layers
        self.boundary_type = boundary_type
        self.sample_data = "sample_" if sample_data else ""

        self.use_lm = False if path_lm == None else True
        self.sent_length = sent_length if path_lm is not None else 0

        # Data
        self.dev = None
        self.test = None
        self.train = None
        self.config = None

        self.pad = None
        self.unk = None

        # EntityDetection
        self.ne2ids = None
        self.ids2ne = None
        self.num_ne = None

        # TransitionDetection
        self.span2ids = None
        self.ids2span = None
        self.num_span = None

        # Decoder
        self.tag2ids = None
        self.ids2tag = None
        self.num_tag = None
        
        self._loader()
        self.path_lm = path_lm
        self.input_lm = InputLM(self.path_lm, self.sent_length)

        self._gen_dict()
        self.conll_labels = CONLLLabels(
            max_layers=self.max_layers, 
            boundary_type=self.boundary_type,
            debug=self.debug
        )

    def get_dataset(self, data, get_org_text=False):
        dataset = NERDataset(
                data, self.input_lm, self.conll_labels, self.sent_length, 
                self.word2ids, self.span2ids, self.ne2ids, self.spantag2ids, self.tag2ids,
                use_lm=self.use_lm, unk="<unk>", pad="<pad>", get_org_text=get_org_text)
        return dataset

    def custom_collate(self, data):
        keys_features = data[0].keys()
        batch = {key:[] for key in keys_features}

        for ids in range(len(data)):
            featues = data[ids]

            for key in keys_features:
                batch[key].append(featues[key])

        span = batch['span']
        tokens = batch['tokens']
        entities = batch['entities']
        # print("batch", batch['sentence_id'])
        sentence_id = batch['sentence_id']
        mask = torch.tensor(batch['mask'])
        input_ids = torch.tensor(batch['input_ids'])
        flat_conll_ids = torch.tensor(batch['flat_conll_ids'])
        span_conll_ids = torch.tensor(batch['span_conll_ids'])
        return sentence_id, tokens, input_ids, mask, flat_conll_ids, span_conll_ids, span, entities

    def get_train(self, shuffle=True):
        temp = self.get_dataset(self.train)
        return DataLoader(temp, batch_size=self.batch_size, shuffle=shuffle, collate_fn=self.custom_collate)

    def get_dev(self, shuffle=False):
        temp = self.get_dataset(self.dev)
        return DataLoader(temp, batch_size=self.batch_size, shuffle=shuffle, collate_fn=self.custom_collate)

    def get_test(self, shuffle=False):
        temp = self.get_dataset(self.test)
        return DataLoader(temp, batch_size=self.batch_size, shuffle=shuffle, collate_fn=self.custom_collate)
    
    def _gen_dict(self):
        self.pad = "<pad>"
        self.unk = "<unk>"
        
        # Encoder
        add_words = [self.unk, self.pad]
        self.word2ids = self.input_lm.word2ids
        self.ids2word = {ids:word for word, ids in self.word2ids.items()}
        self.num_word = len(self.word2ids)

        ## Check unique words
        assert self.num_word == len(set(self.word2ids))

        # EntityDetection
        ne = ["O", "NE"]
        self.ne2ids = {tag:ids for ids, tag in enumerate(ne)}
        self.ids2ne = {ids:tag for tag, ids in self.ne2ids.items()}
        self.num_ne = len(self.ne2ids)

        # SpanDetection
        span = self.boundary_type.replace("O","")
        span = ["O"] + [f"{s}-NE" for s in span]
        self.span2ids = {tag:ids for ids, tag in enumerate(span)}
        self.ids2span = {ids:tag for tag, ids in self.span2ids.items()}
        self.num_span = len(self.span2ids)

        # TagDetection {'O': 0, 'ORG': 1, 'MISC': 2, 'PER': 3, 'LOC': 4}
        tags = self.config['mentions']['unique_labels']
        tags = ["O"]+[x for x in list(tags)]
        self.tag2ids = {tag:ids for ids, tag in enumerate(tags)}
        self.ids2tag = {ids:tag for tag, ids in self.tag2ids.items()}
        self.num_tag = len(self.tag2ids)
        
        # FlattenDecoder: {'O': 0, 'B-ORG': 1, 'I-ORG': 2, 'E-ORG}
        unique_labels = self.config['mentions']['unique_labels']
        boundary_type=self.boundary_type.replace("O","")
        spantags = [f"{s}-{t}" for t in unique_labels for s in boundary_type]
        self.spantag2ids = {tag:ids for ids, tag in enumerate(['O']+spantags)}
        self.ids2spantag = {ids:tag for tag, ids in self.spantag2ids.items()}
        self.num_spantag = len(self.spantag2ids)

        print(f"num_vocab: {self.num_word}")
        print(f"num_tag: {self.num_tag}")
        print(f"num_span: {self.num_span}")
        print(f"num_spantag: {self.num_spantag}")


    def _load(self, data_type):
        # Load dataset
        if data_type in ["train", "test", 'dev']:
            path = f"{self.path_data}/{self.sample_data}{data_type}.json"
            data = json.load(open(path))
            data = [item for item in data if len(item['tokens'])>0] 
            # Update sentence length
            for item in data:
                sent_length = len(item['tokens'])
                if sent_length>self.sent_length:
                    self.sent_length=findNextPowerOf2(sent_length)
        # Load config
        elif data_type=="config":
            path = f"{self.path_data}/config.json"
            data = json.load(open(path))
        else:
            log = f"FlatLMDataloader, input data_type error {data_type}"
            raise log
        return data

    def _loader(self):
        self.train = self._load("train")
        self.test = self._load("test")
        self.dev = self._load("dev")
        self.config = self._load("config")

        print(f"Train : {len(self.train)} sentences")
        print(f"Dev : {len(self.dev)} sentences")
        print(f"Test : {len(self.test)} sentences")
        print(f"Max sents length: {self.sent_length} tokens")