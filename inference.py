import os
import json
import torch
import argparse
import numpy as np 

import model.model as module_arch
import utils.dataloader as module_data

from utils.metric import evaluate 
from utils.utils import decode_tags
from utils.metric import sequence_f1
from utils.parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')
    # setup dataloader instances
    data_loader = config.init_obj('dataloader', module_data)
    # build model architecturea
    model = config.init_obj(
        'arch', module_arch, 
        num_tag=data_loader.num_tag,
        path_lm=data_loader.path_lm)
    logger.info(model)

    # get function handles of loss and metrics
    metric_fns = {"sequence_f1": sequence_f1}
    
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']

    if config['n_gpu']>1 and torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    org_text = []
    sequenc_preds = []
    sequenc_target = []
    with torch.no_grad():
        for batch_idx, (sentence_id, tokens, input_ids, mask, target, span_conll_ids, span, entities) in enumerate(data_loader.get_test()):
            mask = mask.to(device)
            target = target.to(device)
            input_ids = input_ids.to(device)
            loss_sequenc, logit_sequenc = model(input_ids, mask, target)
            # sequence 
            _sequenc_preds = decode_tags(
                logit_sequenc.cpu().argmax(dim=-1).numpy(), 
                data_loader.ids2spantag
            )
            _sequenc_target = decode_tags(
                target.cpu().numpy(), data_loader.ids2spantag
            )
            for ids, m in enumerate(mask):
                num_token = sum(m)
                org_text.append(tokens[ids].split("|")[:num_token])
                # sequence
                sequenc_preds.append(_sequenc_preds[ids][:num_token])
                sequenc_target.append(_sequenc_target[ids][:num_token])

    logger.info("### sequence labeling  prediction ###")
    evaluate(sequenc_target, sequenc_preds, logger)
    
    n_samples = len(data_loader.test)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples 
            for i, (_,met) in enumerate(metric_fns.items())})
    logger.info(log)

    # Save predictions
    temp_resume = str(config.resume).split('/')
    path = "/".join(temp_resume[:-1]) + "/outputs"

    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        print("The new directory is created!")

    ## conll
    with open(path+"/text.json", 'w') as F: 
        json.dump(org_text, F)
    with open(path+"/sequence_pred.json", 'w') as F: 
        json.dump(sequenc_preds, F)
    with open(path+"/sequence_labels.json", 'w') as F: 
        json.dump(sequenc_preds, F)
    print(f"Saved at: {path}_pred.json")
    print(f"Saved at: {path}_labels.json")
    print(f"Saved at: {path}_text.json")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
