import numpy as np
from seqeval.scheme import IOBES
from seqeval.metrics import classification_report
# from utils.dataloader import Loader as module_data

## Classification report
def evaluate(target, prediction, logger=None):
    report = classification_report(
        [list(np.concatenate(target))], 
        [list(np.concatenate(prediction))], 
        mode='strict', scheme=IOBES, digits=4)
    if logger is not None:
        logger.info(report)
    report = report.split()
    assert report[-18]=='micro' 
    f1 = float(report[-14])
    # pdb.set_trace()
    return f1

## Decoder
def _decode_tags(logit, ids2tag):
    return [ids2tag[t] for t in logit]

def decode_tags(preds, ids2tag):
    out = [_decode_tags(t, ids2tag) for t in preds]
    return out

def f1(logits, target, mask, ids2tag):
    outputs = {"input_ids":[], "predictions":[], "entities":[]}
    entities = decode_tags(target.cpu().numpy(), ids2tag)
    preds = decode_tags(logits.cpu().argmax(dim=-1).numpy(), ids2tag)
    
    for ids, m in enumerate(mask):
        num_token = sum(m)
        outputs['predictions'].append(preds[ids][:num_token])
        outputs['entities'].append(entities[ids][:num_token])
    try:
        report = classification_report(
            outputs['entities'],
            outputs['predictions'], 
            mode='strict', scheme=IOBES,
            digits=4)
        report = report.split()
        assert report[-18]=='micro' 
        f1 = float(report[-14])
    except:
        f1, report = 0, None

    return f1, report

def sequence_f1(logits, target, mask, ids2tag):
    return f1(logits, target, mask, ids2tag)