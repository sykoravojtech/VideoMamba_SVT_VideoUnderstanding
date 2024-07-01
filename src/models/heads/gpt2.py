# https://github.com/fkodom/clip-text-decoder

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer


def check_language_model(name: str) -> None:
    allowed = ["distilgpt2", "gpt2", "gpt2-medium"]
    if name not in allowed:
        raise ValueError(
            f"Unsupported language model '{name}'. Allowed: {allowed}.")


def load_language_model(name: str, device: Optional[Union[str, torch.device]] = None) -> nn.Module:
    check_language_model(name)
    config = GPT2Config.from_pretrained(name, add_cross_attention=True)
    model = GPT2LMHeadModel.from_pretrained(name, config=config)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    return model


def load_tokenizer(name: str) -> GPT2Tokenizer:
    check_language_model(name)
    tokenizer = GPT2Tokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
