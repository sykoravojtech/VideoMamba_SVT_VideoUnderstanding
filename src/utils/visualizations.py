
import imageio
import numpy as np
from IPython.display import Image

def investigate_video(sample_video, id2label):
    """Utility to investigate the keys present in a single video sample."""
    for k in sample_video:
        print(k)
        if k == "video":
            print(k, sample_video["video"].shape)
        else:
            print(k, sample_video[k])
    print(f"Video label: {id2label[sample_video[k]]}")


def unnormalize_img(img, mean, std):
    """Un-normalizes the image pixels."""
    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)


def create_gif(video_tensor, filename,  mean, std):
    """Prepares a GIF from a video tensor.

    The video tensor is expected to have the following shape:
    (num_frames, num_channels, height, width).
    """
    frames = []
    for video_frame in video_tensor:
        frame_unnormalized = unnormalize_img(video_frame.permute(1, 2, 0).numpy(), mean, std)
        frames.append(frame_unnormalized)
    kargs = {"duration": 0.25}
    imageio.mimsave(filename, frames, "GIF", **kargs)
    return filename


def display_gif(video_tensor, gif_name,  mean, std):
    """Prepares and displays a GIF from a video tensor."""
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    gif_filename = create_gif(video_tensor, gif_name, mean, std)
    return Image(filename=gif_filename)