import numpy as np
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix,log_loss
LABELS=['neg','neu','pos']
def metrics(y_true,probabilities):
    probs=np.asarray(probabilities);pred=np.asarray(LABELS)[probs.argmax(axis=1)]
    return {'accuracy':float(accuracy_score(y_true,pred)),'macro_f1':float(f1_score(y_true,pred,labels=LABELS,average='macro',zero_division=0)),
        'log_loss':float(log_loss(y_true,probs,labels=LABELS)),
        'per_class':classification_report(y_true,pred,labels=LABELS,output_dict=True,zero_division=0),
        'confusion_matrix':confusion_matrix(y_true,pred,labels=LABELS).tolist(),'label_order':LABELS}
