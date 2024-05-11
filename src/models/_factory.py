from .classification_model import VideoClassificationModel

def create_model(config):
    model_type = config.MODEL.TYPE
    if(model_type == 'classification'):
        return VideoClassificationModel(config)