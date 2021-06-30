import torch
import Config
import transformers
import torch.nn as nn


class TOXICMODEL(nn.Module):
    def __init__(self,conf):
        super(TOXICMODEL,self).__init__()
        self.conf = conf
        self.bert = transformers.BertModel.from_pretrained(self.conf)
        self.dropout = torch.nn.Dropout(Config.CLASSIFIER_DROPOUT_PROB)
        self.classifier = torch.nn.Linear(Config.HIDDEN_SIZE,Config.NUMBER_OF_LABEL)

    def forward(self,input_ids,attention_mask,token_type_ids):
        _, output = self.bert(input_ids,attention_mask,token_type_ids,return_dict = False)
        output = self.dropout(output)
        output = self.classifier(output)   
        return output 
