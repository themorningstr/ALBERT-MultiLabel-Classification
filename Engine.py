import torch
import Config 
import numpy as np   


def Loss():
    loss_func = torch.nn.BCEWithLogitsLoss()
    return loss_func

def Train_Func(dataLoader,model,optimizer,scheduler):
    model.train()
    final_loss = 0

    for index,batch in enumerate(dataLoader):
        batch = tuple(t.to(Config.DEVICE) for t in batch)
        Batch_input_ids,Batch_attention_masks,Batch_target_labels,Batch_token_type_ids = batch
        optimizer.zero_grad()
        logits = model(Batch_input_ids,attention_mask = Batch_attention_masks,token_type_ids = Batch_token_type_ids)
        loss = Loss(logits.view(-1,Config.NUMBER_OF_LABEL),Batch_target_labels.type_as(logits).view(-1,Config.NUMBER_OF_LABEL))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss

def Eval_Func(dataLoader,model):
    model.eval()
    final_loss = 0
    predicted_label, true_label = [],[]
    with torch.no_grad():
        for _,batch in enumerate(dataLoader):
            batch = tuple(t.to(Config.DEVICE) for t in batch)
            Batch_input_ids,Batch_attention_masks,Batch_target_labels,Batch_token_type_ids = batch    
            logits = model(Batch_input_ids,attention_mask = Batch_attention_masks,token_type_ids = Batch_token_type_ids)
            loss = Loss(logits.view(-1,Config.NUMBER_OF_LABEL),Batch_target_labels.type_as(logits).view(-1,Config.NUMBER_OF_LABEL))
            final_loss += loss.item()
            prediction_label = torch.sigmoid(logits)

            prediction_label = prediction_label.to("cpu").numpy()
            Batch_target_labels = Batch_target_labels.to("cpu").numpy()

            predicted_label.append(prediction_label)
            true_label.append(Batch_target_labels)

    return final_loss,predicted_label,true_label


            
  