from fvcore.common.config import CfgNode

from .classification_model import VideoClassificationModel
from .captioning_model import VideoCaptioningModel

def create_model(config: CfgNode):
    model_type = config.MODEL.TYPE
    if(model_type == 'classification'):
        return VideoClassificationModel(config)
    if(model_type == 'captioning'):
        return VideoCaptioningModel(config)
    raise NotImplementedError(f'{model_type} is not implemented.')