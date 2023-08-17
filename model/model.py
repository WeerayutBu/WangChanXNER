import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig


class MLP(nn.Module):
    def __init__(self, input_feature, output_feature):
        super(MLP, self).__init__()
        self.input_feature = input_feature
        self.hidden_feature = input_feature
        self.output_feature = output_feature
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(
            self.input_feature, 
            self.output_feature
        )

    def forward(self, X):
        X = self.dropout(X)
        X = self.fc(X)
        return X


class NERModel(nn.Module):
    def __init__(self, num_tag, path_lm=None, **kwargs):
        super(NERModel, self).__init__()
        self.num_tag = num_tag
        self.path_lm = path_lm
        self.fine_tune_lm = kwargs['fine_tune_lm']

        # Initial lm
        self.lm  = AutoModel.from_pretrained(path_lm, output_hidden_states=True)
        for param in self.lm.parameters():
            param.requires_grad = self.fine_tune_lm

        self.n_span = len("BIOES")
        self.n_spantype = int((self.num_tag-1) * (self.n_span-1))+1
        self.hidden_size = self.lm.config.hidden_size
        self.sequence_decoder = MLP(self.hidden_size, self.n_spantype)


    def encoder(self, input_ids, mask=None):
        outputs = self.lm(input_ids, mask)
        # print(len(outputs))  # 3 for roberta and bert, 4 for luke
        hidden_states = outputs[2]
        # print(len(hidden_states))  # 13
        # embedding_output = hidden_states[0]
        hidden_states = hidden_states[1:]
        hidden_states = torch.stack(hidden_states,dim=-1).mean(dim=-1)
        return hidden_states

    def forward(self, input_ids, mask, labels=None, lambda_max_loss=1e-3):
        loss = torch.tensor(0)
        output = self.encoder(input_ids, mask=mask)
        logits = self.sequence_decoder(output)

        if labels==None:
            return loss, logits

        # calculate loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        base_loss = loss_fct(logits.transpose(1,2), labels)
        base_loss = (base_loss*mask) if mask is not None else loss
        loss = torch.mean(base_loss)+lambda_max_loss*base_loss.max(-1).values.mean()
        return loss, logits

    def features(self, input_ids, mask=None):
        pass