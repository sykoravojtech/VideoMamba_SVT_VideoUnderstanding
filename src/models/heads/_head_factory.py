from .mlp_head import MLPHead

def create_head(config):
    head_type = config.MODEL.HEAD.TYPE
    if head_type == 'MLP':
        return MLPHead(config)
    raise ModuleNotFoundError(f'No downstream head called: {head_type}')