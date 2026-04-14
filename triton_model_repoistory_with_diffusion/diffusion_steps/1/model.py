import json
import os
import re
import sys
import yaml
from typing import Dict, List
import torch.nn.functional as F
import torch
import numpy as np
import triton_python_backend_utils as pb_utils
from nltk.tokenize import word_tokenize
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import torchaudio

import random
import time


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
    
    def _to_256(self,v):
        # v: [B, D]
        D = v.size(-1)
        if D >= 256:
            return v[..., :256]
        else:
            
            return F.pad(v, (0, 256 - D))
    
    def execute(self, requests):
        responses: List[pb_utils.InferenceResponse] = []

        for request in requests:
            try:
                bert_dur = pb_utils.get_input_tensor_by_name(request, "BERT_DUR")
                ref_s = pb_utils.get_input_tensor_by_name(request, "REF_S")
                embedding_scale = pb_utils.get_input_tensor_by_name(request, "EMBEDDING_SCALE_OUT")
                
                bert_dur_np = bert_dur.as_numpy()
                ref_s_np = ref_s.as_numpy()
                device = self.device
                bert_dur_tensor = torch.from_numpy(bert_dur_np).to(device)
                ref_s_tensor = torch.from_numpy(ref_s_np).to(device)

                embedding_scale = float(embedding_scale.as_numpy()[0][0]) if embedding_scale is not None else 0.75

                # pool BERT over time to get a content summary
                t_ctx = bert_dur_tensor.mean(dim=1)          # [B, H]
                t_ctx_256 = self._to_256(t_ctx)            # [B, 256]

                # blend ref style with text summary (controlled by embedding_scale in [0, +))
                gamma = float(embedding_scale)
                gamma = max(0.0, min(gamma, 1.0))     # clamp to [0,1] for safety

                # simple L2 norm to keep scales sane
                def _lnorm(x, eps=1e-6):
                    return x / (x.norm(dim=-1, keepdim=True) + eps)

                s_pred = _lnorm((1 - gamma) * ref_s_tensor + gamma * t_ctx_256) * ref_s_tensor.norm(dim=-1, keepdim=True)
                
                s_pred = s_pred.cpu().numpy()
                
                out_s_pred = pb_utils.Tensor("S_PRED", s_pred)

                responses.append(pb_utils.InferenceResponse(output_tensors=[out_s_pred]))
            except Exception as exc:  # pragma: no cover - propagated as Triton error
                err = pb_utils.TritonError(f"Frontend preprocessing failed: {exc}")
                responses.append(pb_utils.InferenceResponse(output_tensors=[], error=err))

        return responses



    def finalize(self):
        pass
