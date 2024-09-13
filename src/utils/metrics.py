import numpy as np

def AP(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average precision (AP) is formulated as AP = 1/N * sum_{k=1}^K( P(k) x rel(k) )
    where N is the number of positive samples in the test set, P(k) is the precision 
    of the top K test samples, and rel(k) is an indicator function equaling 1 if 
    the item at rank k is a positive sample, 0 otherwise"""
    
    # Sort indices in descending order of prediction scores
    sortind = np.argsort(-y_pred)
    
    # Sorted true labels according to prediction scores
    sorted_true = y_true[sortind]
    
    # True positives and false positives
    tp = sorted_true == 1
    fp = sorted_true != 1
    
    # Cumulative sums of true positives and false positives
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Number of positive samples
    npos = np.sum(y_true == 1)
    
    # Recall and precision
    rec = tp_cumsum / npos
    prec = tp_cumsum / (fp_cumsum + tp_cumsum)
    
    # Average precision calculation
    ap = np.sum(prec[tp]) / npos
    
    return ap


def compute_multilabel_mAP(y_true: np.ndarray, y_pred: np.ndarray, num_labels: int) -> float:
    mAP = []
    for class_id in range(num_labels):
        ap = AP(y_true[:, class_id], y_pred[:, class_id])
        if not np.isnan(ap):
            mAP.append(ap)
    return np.mean(mAP)