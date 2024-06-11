from .mlp_head import MLPHead
from .gpt2 import load_language_model

def create_head(config):
    head_type = config.MODEL.HEAD.TYPE
    if head_type == 'MLP':
        return MLPHead(config)
    elif "gpt2" in head_type:
        return load_language_model(head_type)
    raise ModuleNotFoundError(f'No head called: {head_type}')