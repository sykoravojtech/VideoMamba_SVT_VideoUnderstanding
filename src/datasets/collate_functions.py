
import torch

def classification_collate_fn(config):
    def inner_collate_fn(examples):
        """The collation function to be used by `Trainer` to prepare data batches."""
        # permute to (num_frames, num_channels, height, width)
        pixel_values = torch.stack(
            [example["video"].permute(1, 0, 2, 3) for example in examples]
        )
        if "video_label" in examples: # multilabel
            config = 1
            num_output_logits = config.MODEL.NUM_CLASSES
            labels = torch.zeros((len(examples), num_output_logits))
            labels[num_output_logits] = 1
        elif "label" in examples: # single class label
            labels = torch.tensor([example["label"] for example in examples])
        return pixel_values, labels
    return inner_collate_fn