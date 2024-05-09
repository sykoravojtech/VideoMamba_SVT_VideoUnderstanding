from .fully_connected_head import FullyConnectedHead

def create_downstream_head(config):
    downstream_head_type = config['downstream_head']['type']
    if downstream_head_type == 'FullyConnected':
        return FullyConnectedHead(config)
    raise ModuleNotFoundError(f'No downstream head called: {downstream_head_type}')