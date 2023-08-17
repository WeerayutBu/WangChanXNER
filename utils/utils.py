import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


## Decoder
def _decode_tags(logit, ids2tag):
    return [ids2tag[t] for t in logit]

def decode_tags(preds, ids2tag):
    out = [_decode_tags(t, ids2tag) for t in preds]
    return out

# Directory
def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
    


## Processing output 
def span2json(tokens, span_labels):
    results = []
    for item in span_labels:
        start, end, tag = item
        results.append({
            'text': tokens[start:end],
            'span':[start, end],
            'entity_type':tag})
    return results


def conll2span(label):
        stack = []
        label = np.array(label)
        state = {}
        labels = []
        for idx, tag in enumerate(label):
            state['Nb'] = tag.split('-')[0]
            state['Nt'] = tag.split('-')[-1]
            if state['Nb'] == "S":
                labels.append((idx,idx+1,state['Nt']))
            elif state['Nb'] == "B": 
                stack.append((idx, state['Nt']))
            elif state['Nb'] == "E":
                if state['Nt'] == stack[-1][1]:
                    temp_tag = stack.pop()
                    labels.append((temp_tag[0], idx+1, state['Nt']))
                else:
                    raise("Error :: Unbalanced") 
        if len(stack) != 0: 
            print("tag", tag)
            print("\nstack", stack)
            print('\nLabel', labels)
            raise "Error :: Unbalanced"
        return labels



import torch
import numpy as np 
from pythainlp.tokenize import word_tokenize
# from utils.utils import span2json
# from utils.utils import conll2span
from utils.dataloader import InputLM
from utils.dataloader import fix_labels
from utils.dataloader import remove_incorrect_tag

def show(x):
    text = f"{str(x['span']):<15}"
    text+= f"{x['entity_type']:<15}"
    text+= f"{''.join(x['text'])}"
    print(text)


def get_dict_prediction(tokens, preds, attention_mask, ids2tag):
    temp_preds=[]
    for index in range(len(preds)):    
        if attention_mask[index] == 1:
            Ptag = ids2tag.get(preds[index].item())
            temp_preds.append(Ptag)
    temp_preds = remove_incorrect_tag(temp_preds, "BIOES")
    temp_preds = fix_labels(temp_preds, "BIOES")    
    temp_preds = conll2span(temp_preds)
    temp_preds = span2json(tokens, temp_preds)   
    return temp_preds
    

def predict(model, text, lm_path, ids2tag, max_sent_length=512):
    tokens = word_tokenize(text, engine='newmm')
    out = InputLM(lm_path, max_sent_length)(tokens,[])
    
    mask = out['attention_mask']
    lm_tokens = out['lm_tokens']
    input_ids = out['input_ids']
    
    mask = torch.tensor([mask]).to(model.lm.device)
    input_ids = torch.tensor([input_ids]).to(model.lm.device)
    loss, logits = model(input_ids, mask)
    preds = logits.argmax(axis=-1)
    entity = get_dict_prediction(lm_tokens, preds[0], mask[0], ids2tag)
    return lm_tokens, sorted(entity, key=lambda t: t['span'])