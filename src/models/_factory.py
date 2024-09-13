from fvcore.common.config import CfgNode

from .classification_model import VideoClassificationModel
from .captioning_model import VideoCaptioningModel, VideoCaptioningModel_VM
from .captioning_model_linear_proj import VideoCaptioningModelLinear

def create_model(config: CfgNode, weight_path: str = None):
    model_type = config.MODEL.TYPE
    if(model_type == 'classification'):
        if weight_path:
            return VideoClassificationModel.load_from_checkpoint(weight_path)
        return VideoClassificationModel(config)
    if(model_type == 'captioning'):
        if config.MODEL.ENCODER.TYPE == "VideoMamba":
            if weight_path:
                return VideoCaptioningModel_VM.load_from_checkpoint(weight_path)
            return VideoCaptioningModel_VM(config)
        else:
            if weight_path:
                return VideoCaptioningModel.load_from_checkpoint(weight_path)
            return VideoCaptioningModel(config)
    if(model_type == 'captioning-linear-proj'):
        if weight_path:
            return VideoCaptioningModelLinear.load_from_checkpoint(weight_path)
        return VideoCaptioningModelLinear(config)

    raise NotImplementedError(f'{model_type} is not implemented.')