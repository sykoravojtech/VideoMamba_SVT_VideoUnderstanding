from .classification_model import VideoClassificationModel

def create_model(config):
    model_type = config['model_type']
    if(model_type == 'classification'):
        return VideoClassificationModel(config)