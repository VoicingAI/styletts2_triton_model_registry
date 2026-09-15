"""Reference source for triton_model_repoistory_with_diffusion/int_steps_2.

NOT a live Triton backend. `int_steps_2/config.pbtxt` declares the onnxruntime
backend and the directory ships a traced `model.onnx`, so this file never ran —
it is kept because it is the only readable record of what that graph computes,
and the graph has no other exporter in this repo.

The graph is pure tensor arithmetic with no weights, so it is speaker- and
checkpoint-independent: it is copied from the template rather than re-exported
per model.

What it does, given per-token duration logits:
  1. sigmoid + sum over the 50 duration bins -> a frame count per token
  2. divide by the requested speed, round, clamp to >= 1
  3. force the first and last token to one frame, and cap the second-to-last
  4. build the [tokens, frames] alignment matrix from the cumulative durations
  5. expand the prosody features (D_OUT) and the text encoding (T_EN) to frames

Note the `hifigan_mode` bug preserved below: it is assigned `True` but compared
against the string `"hifigan"`, so the one-frame shift never runs. The traced
`model.onnx` has the same behaviour. `StyleTTS2Synth.synthesize` in
common_code/inference_core.py *does* apply the shift, so torch and Triton
output differ for hifigan decoders. Fixing it means re-tracing this graph.
"""

import json
import os
import re
import sys
import yaml
from typing import Dict, List
import torch
import librosa
import numpy as np
import triton_python_backend_utils as pb_utils
from nltk.tokenize import word_tokenize
import random


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = False



class TritonPythonModel:
    def initialize(self,args):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hifigan_mode = True
    
    def create_alignment_efficient(self,pred_dur, total_frames, device, use_fp16=True):
        """Memory-efficient alignment tensor creation"""
        
        # Calculate cumulative positions
        end_indices = torch.cumsum(pred_dur, dim=0)
        start_indices = torch.cat((torch.tensor([0], device=device), end_indices[:-1]))
        
        # Create alignment tensor with appropriate dtype
        dtype = torch.float16 if use_fp16 else torch.float32
        pred_aln_trg = torch.zeros((len(pred_dur), total_frames), device=device, dtype=dtype)
        
        # Efficient vectorized assignment
        frame_range = torch.arange(total_frames, device=device).unsqueeze(0).expand(len(pred_dur), -1)
        mask = (frame_range >= start_indices.unsqueeze(1)) & (frame_range < end_indices.unsqueeze(1))
        pred_aln_trg[mask] = 1.0
        
        return pred_aln_trg
    
    
    def execute(self, requests):
        responses: List[pb_utils.InferenceResponse] = []

        for request in requests:
            try:
                duration_tsr = pb_utils.get_input_tensor_by_name(request, "DURATION")
                d_tsr = pb_utils.get_input_tensor_by_name(request, "D_OUT")
                t_en_tsr = pb_utils.get_input_tensor_by_name(request, "T_EN")
                speed_out = pb_utils.get_input_tensor_by_name(request, "SPEED_OUT")
                
                duration_np = duration_tsr.as_numpy()
                d_np = d_tsr.as_numpy()
                t_en_np = t_en_tsr.as_numpy()
                speed = float(speed_out.as_numpy()[0][0]) if speed_out is not None else 1
                
                device = self.device
                
                duration = torch.from_numpy(duration_np).to(device).to(torch.float16)
                d = torch.from_numpy(d_np).to(device).to(torch.float16)
                t_en = torch.from_numpy(t_en_np).to(device).to(torch.float16)
                
                duration = torch.sigmoid(duration).sum(axis=-1)
                speed = max(speed,1e-3)
                pred_dur = torch.round(duration.squeeze()/speed).clamp(min=1)
                
                print(f"pred_dur before changes:  {pred_dur}")
                
                if pred_dur[0]>1:
                    pred_dur[0] = 1

                if pred_dur[-1]>1:
                    pred_dur[-1] = 1

                if pred_dur[-2] > 15:
                    new_val = max(pred_dur[-2] / 2, 10)
                    new_val = min(new_val, 16)
                    pred_dur[-2] = new_val
                    
                print(f"pred_dur after changes:  {pred_dur}")

                
                total_frames = int(pred_dur.sum().item())
               
                pred_aln_trg = self.create_alignment_efficient(pred_dur, total_frames, device)
                
                d = d.to(torch.float16)
                
                en = d.transpose(-1, -2) @ pred_aln_trg
                
                if self.hifigan_mode == "hifigan":
                    asr_new = torch.zeros_like(en)
                    asr_new[:, :, 0] = en[:, :, 0]
                    asr_new[:, :, 1:] = en[:, :, 0:-1]
                    en = asr_new
                asr = t_en @ pred_aln_trg.unsqueeze(0)
                
                if self.hifigan_mode == "hifigan":
                    asr_new = torch.zeros_like(asr)
                    asr_new[:, :, 0] = asr[:, :, 0]
                    asr_new[:, :, 1:] = asr[:, :, 0:-1]
                    asr = asr_new
                
                en_np = en.float().cpu().numpy()
                asr_np = asr.float().cpu().numpy()

                out_en = pb_utils.Tensor("EN", en_np)
                out_asr = pb_utils.Tensor("ASR", asr_np)
                
                
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_en,out_asr]))
            except Exception as exc:  # pragma: no cover - propagated as Triton error
                err = pb_utils.TritonError(f"Frontend preprocessing failed: {exc}")
                responses.append(pb_utils.InferenceResponse(output_tensors=[], error=err))

        return responses

    def finalize(self):
        pass
