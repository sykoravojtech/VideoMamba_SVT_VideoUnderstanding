import numpy as np

def AP(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average precision (AP) is formulated as AP = 1/N * sum_{k=1}^K( P(k) x rel(k) )
    where N is the number of positive samples in the test set, P(k) is the precision 
    of the top K test samples, and rel(k) is an indicator function equaling 1 if 
    the item at rank k is a positive sample, 0 otherwise"""
    sortind = np.argsort(-y_pred)
    tp = y_true[sortind] == 1
    fp = y_true[sortind] != 1
    npos = np.sum(y_true == 1)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / npos
    prec = tp / (fp + tp)

    ap = 0
    for i in range(len(y_pred)):
        if y_true[sortind][i] == 1:
            ap += prec[i]
    ap /= npos

    return ap


def compute_multilabel_mAP(y_true: np.ndarray, y_pred: np.ndarray, num_labels: int) -> float:
    mAP = []
    for class_id in range(num_labels):
        ap = AP(y_true[:, class_id], y_pred[:, class_id])
        if not np.isnan(ap):
            mAP.append(ap)
    return np.mean(mAP)