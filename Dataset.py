import torch
import Config


def createDataset(input_id,attention_mask,label,token_type_id):
    dataset = torch.utils.data.TensorDataset(input_id,attention_mask,label,token_type_id)
    return dataset 



class TOXICDATASET:
    def __init__(self,data,target):
        self.data = data
        self.target = target
        self.tokenizer = Config.TOKENIZER 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self):
        input_ids = []
        attention_masks = []
        token_type_ids = []

        for data in self.data:
            encoded_dict = self.tokenizer.batch_encode_plus(data,
                                                            add_special_tokens = True,
                                                            max_length = Config.MAX_LEN,
                                                            pad_to_max_length = True,
                                                            return_attention_mask = True,
                                                            return_tensors = 'pt',
                                                            truncation=True)
            
            input_ids.append(encoded_dict["input_ids"])
            attention_masks.append(encoded_dict["attention_mask"])
            token_type_ids.append(encoded_dict["token_type_ids"])

        return {
            "input_ids" : torch.cat(input_ids,dim = 0),
            "attention_masks" : torch.cat(attention_masks,dim = 0),
            "token_type_ids" : torch.cat(token_type_ids,dim = 0),
            "targets" : torch.tensor(self.target,dtype = torch.float64)
        }




