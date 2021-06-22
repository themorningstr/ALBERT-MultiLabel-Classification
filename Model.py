import torch
import transformers
import Config


class TOXICMODEL(transformers.AlbertPreTrainedModel):
    def __init__(self):
        super(TOXICMODEL,self).__init__()
        self.Num_Label = Config.NUMBER_OF_LABEL
        self.config = transformers.AlbertConfig.from_pretrained(Config.MODEL_BASE_PATH)
        self.albert = transformers.AlbertForSequenceClassification.from_pretrained(self.config,return_dict = False)
        self.dropout = torch.nn.Dropout(Config.CLASSIFIER_DROPOUT_PROB)
        self.classfier = torch.nn.Linear(Config.HIDDEN_SIZE,self.Num_Label)
        self.init_weights()

    def forward(self,input_ids,attention_mask,token_type_ids,labels):
        output = self.albert(input_ids,attention_mask,token_type_ids,labels)
        pooled_output = output[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)   
        return logits 
         