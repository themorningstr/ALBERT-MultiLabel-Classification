import torch 
import numpy as np


def Loss_Func(output,targets):
  return torch.nn.BCEWithLogitsLoss()(output,targets)


def Train_Func(dataLoader,model,optimizer,device,scheduler):
    model.train()

    for index,batch in enumerate(dataLoader):
      ids = batch["input_ids"]
      masks = batch["attention_masks"]
      token = batch["token_type_ids"]
      target = batch["targets"]

      ids = ids.to(device,dtype = torch.long)
      masks = masks.to(device,dtype = torch.long)
      token = token.to(device,dtype = torch.long)
      target = target.to(device,dtype = torch.float)
              
      optimizer.zero_grad()
      output = model(input_ids = ids,
                    attention_mask = masks,
                    token_type_ids = token)
      loss = Loss_Func(output,target)
      loss.backward()
      optimizer.step()

      if scheduler is not None:
        scheduler.step() 

      if index / 10 == 0:
        print(f"Index : {index} >>>=============================>>> Loss : {loss}")



def Eval_Func(dataLoader,model,device):
    model.eval()
    final_targets = []
    final_outputs = []
    
    for index,batch in enumerate(dataLoader):
      ids = batch["input_ids"]
      masks = batch["attention_masks"]
      token = batch["token_type_ids"]
      target = batch["targets"]


      ids = ids.to(device,dtype = torch.long)
      masks = masks.to(device,dtype = torch.long)
      token = token.to(device,dtype = torch.long)
      target = target.to(device,dtype = torch.float)

      output = model(input_ids = ids,
                    attention_mask = masks,
                    token_type_ids = token)    
      loss = Loss_Func(output,target)

      final_targets.extend(target.cpu().detach().numpy().tolist())
      final_outputs.extend(torch.sigmoid(output).cpu().detach().numpy().tolist())

      return loss, np.vstack(final_outputs),np.vstack(final_targets)




            
  
