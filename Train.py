import torch
import Config     
import Model 
import Engine 
import Dataset
import Utils
import transformers
import numpy as np
from tqdm import trange
import pandas as pd
from sklearn import model_selection

def train():

  df = pd.read_csv(Config.TRAINING_FILE)
  
  target_cols = df.columns[2:]


  Train_Data,Valid_Data,Train_Target,Valid_Target = model_selection.train_test_split(df.comment_text.values,
                                                                               df[target_cols].values,
                                                                               test_size = .2,
                                                                               random_state = 2021,
                                                                               shuffle = True)
  
  Train_dataset = Dataset.TOXICDATASET(comment_text = Train_Data,target = Train_Target)
    
  

  Train_DataLoader = torch.utils.data.DataLoader(Train_dataset,
                                                 batch_size = Config.TRAIN_BATCH_SIZE,
                                                 sampler = torch.utils.data.RandomSampler(Train_dataset)
                                                 )
  
  Valid_dataset = Dataset.TOXICDATASET(comment_text = Valid_Data,target = Valid_Target)

  Valid_DataLoader = torch.utils.data.DataLoader(Valid_dataset,
                                                 batch_size = Config.VALID_BATCH_SIZE,
                                                 sampler = torch.utils.data.SequentialSampler(Valid_dataset)
                                              )

  model = Model.TOXICMODEL(conf = Config.MODEL_BASE_PATH)
  model.to(Config.DEVICE)


  optimizer_grouped_parameters = Utils.optimizer_params(model)

  optimizer = transformers.AdamW(optimizer_grouped_parameters,lr=2e-5,correct_bias=True)

  total_steps = int(len(df) / 16 * 3)
  scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

  best_loss = np.inf
  for epoch in trange(3, desc = "EPOCHS"):

    Engine.Train_Func(dataLoader = Train_DataLoader,optimizer = optimizer, device = Config.DEVICE , model = model,scheduler = scheduler)
    Valid_loss, output, target = Engine.Eval_Func(dataLoader = Valid_DataLoader, model = model,device = Config.DEVICE)

    if Valid_loss < best_loss:
      torch.save(model.state_dict(),Config.MODEL_PATH)
      Valid_loss = best_loss



              
 

    
 



 

if __name__ == "__main__":
    train()
  


