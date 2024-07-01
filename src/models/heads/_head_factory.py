from .mlp_head import MLPHead
from .generative_head import GenerativeHead

def create_head(config):
    head_type = config.MODEL.HEAD.TYPE
    if head_type == 'MLP':
        return MLPHead(config)
    elif head_type == 'Generative':
        return GenerativeHead(config)
    raise ModuleNotFoundError(f'No head called: {head_type}')