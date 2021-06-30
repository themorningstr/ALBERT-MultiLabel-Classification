import pandas as pd
import sklearn
import numpy as np

def optimizer_params(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    return optimizer_parameters   

       
    # True_label :- true label
    # Predicted_label :- predicted label
def accuracy_score(True_label,Predicted_label):
    Accuracy_Score = sklearn.metrics.accuracy_score(True_label, Predicted_label)
    return Accuracy_Score
        # True_label :- true label
    # Predicted_label :- predicted label
def f1_score(True_label,Predicted_label):
    F1_Score = sklearn.metrics.f1_score(True_label,Predicted_label,average = "micro")
    return F1_Score

    # True_label :- true label
    # Predicted_label :- predicted label
    # labels = :- a list of column label name
def multilabel_confusiom_matrix(True_label,Predicted_label,labels):
    MCM = sklearn.metrics.multilabel_confusion_matrix(y_true=True_label,
                                                      y_pred = Predicted_label,
                                                      labels =labels)
    return MCM

#Creating the Accuracy Measurement Function
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(predicted, true):
    pred_flat = np.argmax(predicted, axis=1).flatten()
    labels_flat = true.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


       
                                                                      
         


    



