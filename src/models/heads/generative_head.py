from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
from fvcore.common.config import CfgNode
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from .head_abstract import HeadAbstract

class GenerativeHead(HeadAbstract):
    def __init__(self, config: CfgNode) -> None:
        super().__init__()
        self.config = config
        model_name = self.config.MODEL.HEAD.LANGUAGE_MODEL
        lm_config = AutoConfig.from_pretrained(model_name, add_cross_attention=True)
        self.language_model = AutoModelForCausalLM.from_pretrained(model_name, config=lm_config)
        # self.language_model = AutoModelForCausalLM.from_config(lm_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, enc_hidden: torch.Tensor, y: Dict[str, torch.Tensor]) -> torch.Tensor:
        enc_hidden = enc_hidden.unsqueeze(1) # make shape (bs x 1 x hidden_size)
        output = self.language_model(input_ids=y['input_ids'],
                                   encoder_hidden_states=enc_hidden,
                                   attention_mask=y['attention_mask'])
        return output

    @torch.cuda.amp.autocast()
    @torch.no_grad()
    def beam_search(self, encoder_hidden_states: torch.Tensor, max_len: int = 64, beam_size: int = 1) -> str:
        """Inference using beam search. For beam search, we predict one token per step.
        After each step, we keep only the 'beam_size' output sequences with the highest
        end-to-end confidence score. Repeat this process until at most 'max_len' tokens
        have been generated.
        """
        encoder_hidden_states = encoder_hidden_states.reshape(1, 1, -1).to(self.device)
        # Since we haven't performed any beam search steps yet, we just have one
        # set of input IDs (with a single "start" token). We use 'None' for the log
        # probability of this sequence, since it's not being predicted by the model.
        input_ids = [torch.tensor([self.tokenizer.bos_token_id], device=self.device)]
        beam_logprobs: Optional[List[float]] = None

        def _get_beam_outputs(_input_ids: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
            """Performs inference on the 'input_ids' Tensor, and collects the top
            'beam_size' results by score. Returns a list of output Tensors, and
            their respective log-probabilities.
            """
            outputs = self.language_model(input_ids=_input_ids.unsqueeze(0), 
                                          encoder_hidden_states=encoder_hidden_states)
            # print(outputs.logits)
            logits: torch.Tensor = outputs.logits[0, -1]
            logprobs = F.log_softmax(logits, dim=-1)

            topk_logprobs = logprobs.topk(k=beam_size)
            indices = topk_logprobs.indices
            logprobs = topk_logprobs.values
            output_ids = [
                torch.cat([_input_ids, idx.reshape(-1)], dim=0) for idx in indices
            ]

            return output_ids, logprobs

        for _ in range(max_len - 1):
            output_ids: List[torch.Tensor] = []
            logprobs: List[float] = []
            beams_done: List[bool] = []

            # Collect the top 'beam_size' results from each beam individually.
            for beam_idx, ids in enumerate(input_ids):
                # If 'beam_logprobs' is already defined, then we've predicted at least
                # one token already. And if the last token is equal to the "stop" token,
                # we don't need to perform inference with this beam anymore.
                if beam_logprobs and ids[-1].item() == self.tokenizer.eos_token_id:
                    output_ids.append(ids)
                    logprobs.append(beam_logprobs[beam_idx])
                    beams_done.append(True)
                    continue

                _output_ids, _logprobs = _get_beam_outputs(ids)
                if beam_logprobs is not None:
                    # Sum the log-probabilities of the existing beam and our predicted
                    # token to get the total log-probability.
                    _logprobs += beam_logprobs[beam_idx]

                # Append the results from this beam to the aggregate lists.
                output_ids += _output_ids
                logprobs += _logprobs.tolist()
                beams_done.append(False)

            if all(beams_done):
                # All search beams are done generating text.
                break

            # Keep only the top 'beam_size' beams by total log-probability.
            indices = torch.tensor(logprobs).topk(k=beam_size).indices
            input_ids = [output_ids[idx] for idx in indices]
            beam_logprobs = [logprobs[idx] for idx in indices]

        # Find the predicted beam with highest overall log-probability.
        best_beam_idx: int = torch.tensor(beam_logprobs).argmax().item()  # type: ignore
        # Decode the predicted token IDs into a text string.
        return self.tokenizer.decode(input_ids[best_beam_idx], skip_special_tokens=True)
