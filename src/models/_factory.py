from fvcore.common.config import CfgNode

from .classification_model import VideoClassificationModel
from .captioning_model import VideoCaptioningModel

def create_model(config: CfgNode, weight_path: str = None):
    model_type = config.MODEL.TYPE
    if(model_type == 'classification'):
        if weight_path:
            return VideoClassificationModel.load_from_checkpoint(weight_path)
        return VideoClassificationModel(config)
    if(model_type == 'captioning'):
        if weight_path:
            return VideoCaptioningModel.load_from_checkpoint(weight_path)
        return VideoCaptioningModel(config)
    raise NotImplementedError(f'{model_type} is not implemented.')