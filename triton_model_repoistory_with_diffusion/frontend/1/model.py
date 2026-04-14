import json
import os
import re
import sys
from typing import Dict, List
import torch

import numpy as np
import triton_python_backend_utils as pb_utils
from nltk.tokenize import word_tokenize

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.abspath(os.path.join(_THIS_DIR, "../../../../common_code"))
sys.path.insert(0, COMMON)

from common_code.config import SPEAKER_MODEL_MAP, get_supported_languages
from common_code.phonemizer_utils import PhonemizerManager
from common_code.text_utils import TextCleaner


class TritonPythonModel:
    def initialize(self, args):
        
        
        cfg = json.loads(args["model_config"])
        params = cfg.get("parameters", {})
        

        self.default_language = "en"
        self.cleaner = TextCleaner()
        self.special_eos_speakers = {"chloe", "dora", "regina", "isha"}
        self.phoneme_cleanup = re.compile(r"\([^)]*\)")
        

        supported_langs = {lang for _, lang in SPEAKER_MODEL_MAP.keys()}
        if not supported_langs:
            supported_langs = set(get_supported_languages())
        self.phonemizer_manager = PhonemizerManager(languages=supported_langs)
        
        self.reference_styles = {}

    def _text_to_tokens(self, text: str, language: str, speaker: str) -> List[int]:
        backend = self.phonemizer_manager.get_backend(language) or self.phonemizer_manager.get_backend(self.default_language)
        if backend is None:
            raise ValueError(f"No phonemizer available for language '{language}'.")

        phoneme = backend.phonemize([text.strip()])[0]
        try:
            phoneme = " ".join(word_tokenize(phoneme))
        except LookupError:
            import nltk

            nltk.download('punkt', quiet=True)
            phoneme = " ".join(word_tokenize(phoneme))
        phoneme = self.phoneme_cleanup.sub("", phoneme)

        token_ids = self.cleaner(phoneme)
        token_ids = [0] + token_ids
        
        token_ids.insert(1, 16)
        if speaker in self.special_eos_speakers and token_ids[-1] != 16:
            token_ids.append(16)

        return token_ids
    
    def length_to_mask(self,lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def execute(self, requests):
        responses: List[pb_utils.InferenceResponse] = []

        for request in requests:
            try:
                text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
                speaker_tensor = pb_utils.get_input_tensor_by_name(request, "SPEAKER")
                language_tensor = pb_utils.get_input_tensor_by_name(request, "LANGUAGE")
                speed_tensor = pb_utils.get_input_tensor_by_name(request, "SPEED")
                embedding_scale_tensor = pb_utils.get_input_tensor_by_name(request, "EMBEDDING_SCALE")

                if text_tensor is None or speaker_tensor is None:
                    raise ValueError("TEXT and SPEAKER inputs are required.")


                text = text_tensor.as_numpy()[0][0].decode("utf-8")
                speaker = speaker_tensor.as_numpy()[0][0].decode("utf-8")
                language = language_tensor.as_numpy()[0][0].decode("utf-8") if language_tensor is not None else self.default_language
                language = language.lower()
                
                if speaker not in self.reference_styles:
                    # 1. Load the tensor (may be 1D or 2D)
                    ref_s_loaded = torch.load(f"/workspace/reference_styles/{speaker}_style.pt").detach().cpu().float()
                    
                    # 2. Check and reshape if it's 1D (D,) to make it 2D (1, D)
                    if ref_s_loaded.dim() == 1:
                        ref_s_loaded = ref_s_loaded.unsqueeze(0)
                        
                    self.reference_styles[speaker] = ref_s_loaded
                    
                ref_s = self.reference_styles[speaker]
                
                token_ids = self._text_to_tokens(text, language, speaker)
                
                token_array = np.asarray(token_ids, dtype=np.int64)
                
                token_tensor_torch = torch.from_numpy(token_array).unsqueeze(0)
                
                print(f"token_array changes:  {token_array}")
                
                # === Create input lengths and mask ===
                input_lengths = torch.LongTensor([token_array.shape[-1]])
                text_mask = self.length_to_mask(input_lengths)  # Boolean mask
                attention_mask = (~text_mask).int()       
                
                
                alpha = torch.tensor([0.3], dtype=torch.float32)
                beta = torch.tensor([0.7], dtype=torch.float32)
                
                
                # === Move to CPU and convert to numpy ===
                out_tokens_np = token_tensor_torch.detach().cpu().numpy() 
                input_lengths_np = input_lengths.cpu().numpy().astype(np.int64)
                attention_mask_np = attention_mask.cpu().numpy().astype(np.int64)
                text_mask_np = text_mask.cpu().numpy()
                ref_s_np = ref_s.detach().cpu().float().numpy()
                
                alpha_np = alpha.detach().cpu().float().numpy().astype(np.float32)
                beta_np = beta.detach().cpu().float().numpy().astype(np.float32)
                
                # === Wrap as Triton tensors ===
                out_tokens = pb_utils.Tensor("TOKENS", out_tokens_np)
                out_inp_lengths = pb_utils.Tensor("INPUT_LENGTHS", input_lengths_np)
                out_text_mask = pb_utils.Tensor("TEXT_MASK", text_mask_np)
                out_attention_mask = pb_utils.Tensor("ATTENTION_MASK", attention_mask_np)
                out_ref_s = pb_utils.Tensor("REF_S", ref_s_np)
                
                out_alpha = pb_utils.Tensor("ALPHA", alpha_np)
                out_beta = pb_utils.Tensor("BETA", beta_np)
                
                speed_out_tensor = pb_utils.Tensor(
                    "SPEED_OUT", 
                    speed_tensor.as_numpy() # Shape is (1, 1)
                )
                
                # embedding_scale_out_tensor = pb_utils.Tensor(
                #     "EMBEDDING_SCALE_OUT", 
                #     embedding_scale_tensor.as_numpy() # Shape is (1, 1)
                # )
                

                responses.append(pb_utils.InferenceResponse(output_tensors=[out_alpha, out_beta, out_tokens, out_inp_lengths, out_text_mask, out_attention_mask,out_ref_s,speed_out_tensor]))
            except Exception as exc:  # pragma: no cover - propagated as Triton error
                err = pb_utils.TritonError(f"Frontend preprocessing failed: {exc}")
                responses.append(pb_utils.InferenceResponse(output_tensors=[], error=err))

        return responses

    def finalize(self):
        pass
