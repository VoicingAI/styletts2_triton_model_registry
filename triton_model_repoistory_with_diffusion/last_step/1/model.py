import json
import os
import re
import sys
import yaml
from typing import Dict, List

import torch
import numpy as np
import triton_python_backend_utils as pb_utils
from nltk.tokenize import word_tokenize

import os
import yaml
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import torchaudio

import random
from munch import Munch

import onnxruntime as ort

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = False

class TritonPythonModel:
    def initialize(self,args):
        
        d = "fbd"

    def execute(self, requests):
        responses: List[pb_utils.InferenceResponse] = []

        for request in requests:
            try:
                
                audio_out_tsr = pb_utils.get_input_tensor_by_name(request, "AUDIO_OUT")
                
                audio_out_np = audio_out_tsr.as_numpy()
                
                out = torch.from_numpy(audio_out_np)
                
                print(f"Shape of audio tensor {audio_out_np.shape}")
                
                # Convert output to CPU immediately
                out_data = out.squeeze().float().cpu().numpy()


                out_audio = pb_utils.Tensor("AUDIO", out_data)
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_audio]))
                
                
            except Exception as exc:  # pragma: no cover - propagated as Triton error
                err = pb_utils.TritonError(f"Frontend preprocessing failed: {exc}")
                responses.append(pb_utils.InferenceResponse(output_tensors=[], error=err))

        return responses
    
    def finalize(self):
        pass
