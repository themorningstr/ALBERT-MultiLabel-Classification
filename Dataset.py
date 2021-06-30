import torch
import Config


class TOXICDATASET:
    def __init__(self,comment_text,target):
        self.comment_text = comment_text
        self.target = target
        self.tokenizer = Config.TOKENIZER
        self.max_len = Config.MAX_LEN
    
    def __len__(self):
        return len(self.comment_text)
    
    def __getitem__(self,item):
      comment_text = str(self.comment_text[item])
      comment_text = " ".join(comment_text.split())
        
      inputs = self.tokenizer.encode_plus(comment_text,
                                          None,
                                          add_special_tokens = True,
                                          max_length = self.max_len,
                                          truncation=True)      
      ids = inputs["input_ids"]
      masks = inputs["attention_mask"]
      token_type_ids = inputs["token_type_ids"]


      padding_length = self.max_len - len(ids)
      ids = ids + ([0] * padding_length)
      masks = masks + ([0] * padding_length)
      token_type_ids = token_type_ids + ([0] * padding_length)


      return {
          "input_ids" : torch.tensor(ids,dtype = torch.long),
          "attention_masks" : torch.tensor(masks,dtype = torch.long),
          "token_type_ids" : torch.tensor(token_type_ids,dtype = torch.long),
          "targets" : torch.tensor(self.target[item],dtype = torch.float)
      }




