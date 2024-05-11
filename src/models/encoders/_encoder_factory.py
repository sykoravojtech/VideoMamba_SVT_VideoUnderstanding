from .video_transformer import VideoTransformerEncoder
from .video_mamba import VideoMambaEncoder

def create_encoder(config):
    encoder_type = config.MODEL.ENCODER.TYPE
    if encoder_type == 'VideoTransformer':
        return VideoTransformerEncoder(config)
    if encoder_type == 'VideoMamba':
        return VideoMambaEncoder(config)
    raise ModuleNotFoundError(f'No encoder called:{encoder_type}')