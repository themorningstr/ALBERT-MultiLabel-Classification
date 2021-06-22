import torch
import Model
import pandas as pd
import Config
import sklearn

def optimizer_params(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    return optimizer_parameters   


def save_checkpoint(state,filename = "My_Checkpoint.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state,filename)


def load_checkpoint(checkpoint):
    print("=> Loading Checkpoint")
    model = Model.TOXICMODEL()
    myModel = model.load_state_dict(checkpoint["state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer"])
    return myModel


def processing(path):
    df = pd.read_csv(path)
    df["One_Hot_Label"] = list(df[list(df.columns[2:])].values)
    targets = list(df.One_Hot_Label.values)
    data = list(df.comment_text.values)
    return data,targets

def accuracy_score(True_label,Predicted_label):
    Accuracy_Score = sklearn.metrics.accuracy_score(True_label, Predicted_label)
    return f"The Accuracy Score is : {Accuracy_Score * 100}"


def f1_score(True_label,Predicted_label):
    F1_Score = sklearn.metrics.f1_score(True_label,Predicted_label,average = "micro")
    return f"The F1_Score is : {F1_Score * 100}"


def multilabel_confusiom_matrix(True_label,Predicted_label,labels):
    MCM = sklearn.metrics.multilabel_confusion_matrix(y_true=True_label,
                                                      y_pred = Predicted_label,
                                                      labels =labels)
    return f"Multilabel_Confusiom_Matrix : {MCM}"


       
                                                                      
         


    



