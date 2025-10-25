import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

def compute_basic_metrics(y_true, y_pred, y_prob=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = None
    cm = confusion_matrix(y_true, y_pred)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc, 'confusion_matrix': cm.tolist()}

def early_warning_auc(labels, times, probs, k_minutes=60):
    """
    labels: list of ground truth per post (0/1)
    times: list of lists - each inner list are timestamps (in seconds) of shares for that post
    probs: list of lists - predicted probability of fake computed using only shares up to that time
    For simplicity, this function expects for each post a list of times and corresponding probs.
    Computes AUC across posts on whether using only early windows predicts final label.
    """
    from sklearn.metrics import roc_auc_score
    y_true = []
    y_score = []
    for lab, tlist, plist in zip(labels, times, probs):
        # find index where time <= k_minutes*60
        cutoff = k_minutes*60
        # find last index <= cutoff
        idx = 0
        for i,tt in enumerate(tlist):
            if tt <= cutoff:
                idx = i
        y_true.append(lab)
        y_score.append(plist[idx])
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None
