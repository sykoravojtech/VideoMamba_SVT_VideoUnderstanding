from typing import Dict, List

import torch
from torch import nn
from torchmetrics.classification import Accuracy, MultilabelAveragePrecision
from fvcore.common.config import CfgNode

from .model_abstract import ModelAbstract
from .encoders import create_encoder, EncoderAbstract
from .heads import create_head, HeadAbstract
from ..utils.metrics import compute_multilabel_mAP

def calc_class_weights(class_counts: List[int]) -> torch.tensor:
    """
        Calculate class weights for imbalanced dataset
    """
    class_counts = torch.tensor(class_counts, dtype=torch.float32)
    total_samples = torch.sum(class_counts)
    num_classes = len(class_counts)

    # Compute class weights as the inverse of class frequency
    class_weights = total_samples / (num_classes * class_counts)

    # Normalize weights to have a maximum of 1 (optional)
    class_weights = class_weights / torch.max(class_weights)
    
    return torch.tensor(class_weights, dtype=torch.float32)

charades_class_counts_train = [
    635, 640, 479, 303, 266, 81, 618, 27, 784, 879, 46, 728, 202, 42, 241, 1055, 
    673, 219, 298, 217, 678, 356, 280, 278, 89, 187, 615, 280, 248, 92, 257, 58, 
    483, 575, 276, 286, 154, 125, 232, 96, 331, 181, 201, 180, 154, 34, 100, 259, 
    99, 91, 82, 347, 341, 233, 177, 162, 160, 196, 73, 1260, 46, 1075, 495, 683, 
    52, 459, 29, 400, 154, 161, 443, 205, 323, 200, 105, 139, 385, 171, 159, 161, 
    134, 503, 228, 74, 133, 28, 62, 231, 198, 53, 87, 39, 305, 94, 111, 37, 449, 
    1444, 364, 100, 127, 25, 237, 36, 207, 189, 1134, 1014, 276, 481, 536, 49, 
    410, 595, 218, 324, 168, 175, 763, 452, 375, 112, 160, 484, 168, 395, 308, 
    478, 282, 162, 207, 46, 353, 98, 257, 315, 35, 213, 57, 81, 31, 506, 192, 
    250, 165, 237, 245, 338, 332, 588, 357, 964, 1029, 643, 1341, 395, 936
]
charades_class_counts_test = [
    227, 209, 180, 113, 95, 23, 204, 28, 273, 285, 16, 197, 72, 14, 96, 265, 
    187, 71, 115, 42, 186, 91, 74, 76, 21, 68, 211, 96, 84, 54, 97, 21, 150, 
    208, 118, 109, 65, 64, 77, 46, 115, 68, 72, 77, 58, 6, 32, 105, 30, 49, 
    37, 131, 95, 87, 73, 42, 60, 63, 32, 368, 7, 367, 211, 245, 22, 111, 13, 
    120, 46, 57, 174, 97, 109, 79, 61, 59, 116, 58, 67, 51, 28, 154, 92, 45, 
    57, 15, 30, 71, 78, 10, 17, 7, 88, 24, 42, 9, 134, 439, 106, 44, 53, 8, 
    56, 15, 73, 50, 300, 380, 56, 211, 240, 20, 134, 199, 83, 145, 59, 60, 
    267, 145, 146, 30, 62, 189, 48, 142, 135, 152, 71, 37, 80, 13, 108, 45, 
    78, 143, 9, 69, 25, 33, 20, 238, 45, 58, 60, 74, 85, 76, 120, 211, 112, 
    279, 393, 175, 381, 134, 273
]
charades_cls_weights_train = calc_class_weights(charades_class_counts_train)
# charades_cls_weights_test = calc_class_weights(charades_class_counts_test)
CLASS_WEIGHTS = charades_cls_weights_train
# print(f"{CLASS_WEIGHTS=}")

class VideoClassificationModel(ModelAbstract):
    def __init__(self, config: CfgNode) -> None:
        super().__init__(config)

    def create_encoder(self) -> EncoderAbstract:
        return create_encoder(self.config)

    def create_head(self) -> HeadAbstract:
        return create_head(self.config)

    def create_loss_function(self) -> nn.Module:
        loss = self.config.MODEL.LOSS
        if loss == 'BCEWithLogitsLoss':
            class_weights = None
            if self.config.MODEL.get('USE_CLASS_WEIGHTS', False):
                class_weights = CLASS_WEIGHTS
            return nn.BCEWithLogitsLoss(pos_weight=class_weights)
        return nn.CrossEntropyLoss()
            
    def compute_metrics(self, step_outputs: List) -> Dict:
        """
            Compute metrics after each epoch
            Args:
                outputs: list of dictionary contain predictions, labels
        """
        all_probas = torch.cat([out['preds'].detach().cpu() for out in step_outputs])
        all_labels = torch.cat([out['labels'].detach().cpu() for out in step_outputs])

        if self.config.MODEL.HEAD.MULTI_LABEL:
            all_preds = all_probas.sigmoid().numpy()
            all_labels = all_labels.int().numpy()
            mAP = compute_multilabel_mAP(y_true=all_labels, y_pred=all_preds, 
                                        num_labels=self.config.MODEL.HEAD.NUM_CLASSES)
            return {'mAP': mAP}
        else:
            acc_calculator = Accuracy(task="multiclass", 
                                        num_classes=self.config.MODEL.HEAD.NUM_CLASSES)
            all_preds = torch.argmax(all_probas, axis=1)
            acc = float(acc_calculator(all_preds, all_labels))
            return {'accuracy': acc}

