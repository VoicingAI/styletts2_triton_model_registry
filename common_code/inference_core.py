# common_code/inference_core.py

import os
import yaml
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import torchaudio

import random
from munch import Munch

from common_code.styletts2.models import build_model, build_model_onnx, load_ASR_models, load_F0_models
from common_code.styletts2.Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from common_code.styletts2.Utils.PLBERT.util import load_plbert

import onnxruntime as ort

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = False

def load_plbert_only(plbert_dir):
    return load_plbert(plbert_dir)

def load_plbert_onnx(plbert_dir):
    onnx_path = os.path.join(plbert_dir, "plbert.onnx")
    config_path = os.path.join(plbert_dir, "config.yml")
    plbert_config = yaml.safe_load(open(config_path))
    
    bert_session = ort.InferenceSession(onnx_path,providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        
    bert_hidden_size = plbert_config['model_params']['hidden_size']
    bert_max_position_embeddings = plbert_config['model_params']['max_position_embeddings']
        
    return bert_session, (bert_hidden_size, bert_max_position_embeddings)


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch({k: recursive_munch(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [recursive_munch(x) for x in d]
    else:
        return d


def _build_sampler(model, model_params):
    diffusion_cfg = getattr(model_params, "diffusion", {}) or {}
    sampler_cfg = diffusion_cfg.get("sampler_config", {})

    sigma_min = sampler_cfg.get("sigma_min", 1e-4)
    sigma_max = sampler_cfg.get("sigma_max", 3.0)
    rho = sampler_cfg.get("rho", 9.0)
    clamp = sampler_cfg.get("clamp", False)

    schedule = KarrasSchedule(sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
    return DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=schedule,
        clamp=clamp,
    )


class StyleTTS2Synth:
    def __init__(self, model_dir, asr_path, asr_config, f0_path, plbert_dir, ref_audio_dir, device):
        self.device = device
        self.ref_audio_dir = ref_audio_dir
        self.reference_styles = {}
        self.alpha, self.beta = 0.3, 0.7

        cfg_path = os.path.join(model_dir, "config_ft.yml")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"config_ft.yml not found in {model_dir}")

        config = yaml.safe_load(open(cfg_path))
        self.model_params = recursive_munch(config['model_params'])

        # plbert = load_plbert_only(plbert_dir)
        self.bert_session, (bert_hidden_size, bert_max_position_embeddings) = load_plbert_onnx(plbert_dir)

        self.model = build_model_onnx(self.model_params, bert_hidden_size, bert_max_position_embeddings)
        _ = [self.model[k].to(self.device).eval() for k in self.model]

        ckpts = [f for f in os.listdir(model_dir) if f.startswith("model") and f.endswith(".pth")]
        if not ckpts:
            raise FileNotFoundError(f"No model*.pth in {model_dir}")
        ckpts.sort()
        ckpt = os.path.join(model_dir, ckpts[-1])
        state = torch.load(ckpt, map_location="cpu")["net"]
        print(state.keys())

        for key in self.model:
            if key in state:
                try:
                    self.model[key].load_state_dict(state[key])
                except Exception:
                    from collections import OrderedDict
                    new_sd = OrderedDict()
                    for k, v in state[key].items():
                        name = k[7:] if k.startswith("module.") else k
                        new_sd[name] = v
                    self.model[key].load_state_dict(new_sd, strict=False)
        _ = [self.model[k].eval() for k in self.model]

        self.sampler = _build_sampler(self.model, self.model_params)
        
        
        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        ).to(self.device)
        self._mel_mean = -4.0
        self._mel_std = 4.0

    @staticmethod
    def _length_to_mask(lengths):
        max_len = torch.max(lengths).item()
        mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
        return ~mask

    def _load_reference_style(self, speaker: str) -> torch.Tensor:
        if speaker in self.reference_styles:
            return self.reference_styles[speaker]

        ref_path = os.path.join(self.ref_audio_dir, speaker, "reference.wav")
        if not os.path.isfile(ref_path):
            raise FileNotFoundError(f"Reference audio not found for speaker '{speaker}' at {ref_path}")

        wave, sr = librosa.load(ref_path, sr=24000)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)

        wave_tensor = torch.from_numpy(audio).float().to(self.device)
        mel_tensor = self._mel_transform(wave_tensor.unsqueeze(0))
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self._mel_mean) / self._mel_std

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        ref = torch.cat([ref_s, ref_p], dim=1)
        self.reference_styles[speaker] = ref
        return ref

    def synthesize(self, tokens: torch.LongTensor, speaker: str, speed: float, diffusion_steps: int | None = 6) -> np.ndarray:
        device = self.device
        tokens = tokens.to(device).unsqueeze(0)
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = self._length_to_mask(input_lengths)

        ref_s = self._load_reference_style(speaker)

        model = self.model
        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        attention_mask = (~text_mask).int()
        
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        steps = diffusion_steps if diffusion_steps is not None else self.default_sampler_steps
        steps = int(max(1, steps))
        noise = torch.randn((1, 256), device=device).unsqueeze(1)
        s_pred = self.sampler(
            noise=noise,
            embedding=bert_dur,
            embedding_scale=1.0,
            features=ref_s,
            num_steps=steps,
        ).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = self.alpha * ref + (1 - self.alpha) * ref_s[:, :128]
        s = self.beta * s + (1 - self.beta) * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)

        spd = max(float(speed), 1e-3)
        pred_dur = torch.round(duration.squeeze() / spd).clamp(min=1)


        total_frames = int(pred_dur.sum().item())
        if total_frames <= 0:
            total_frames = int(pred_dur.shape[0])

        pred_aln_trg = torch.zeros(pred_dur.shape[0], total_frames, device=device)
        c_frame = 0
        for idx in range(pred_dur.shape[0]):
            step = int(pred_dur[idx].item())
            step = max(step, 1)
            upper = min(c_frame + step, total_frames)
            pred_aln_trg[idx, c_frame:upper] = 1.0
            c_frame = upper
        if c_frame < total_frames:
            pred_aln_trg[-1, c_frame:] = 1.0

        align = pred_aln_trg.unsqueeze(0)

        en = d.transpose(-1, -2) @ align
        if self.model_params.decoder.type == "hifigan":
            en_shift = torch.zeros_like(en)
            en_shift[:, :, 0] = en[:, :, 0]
            en_shift[:, :, 1:] = en[:, :, :-1]
            en = en_shift

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = t_en @ align
        if self.model_params.decoder.type == "hifigan":
            asr_shift = torch.zeros_like(asr)
            asr_shift[:, :, 0] = asr[:, :, 0]
            asr_shift[:, :, 1:] = asr[:, :, :-1]
            asr = asr_shift

        out = model.decoder(asr, F0_pred, N_pred, ref)
        wav = out.squeeze().detach().cpu().numpy()
        if wav.shape[0] > 50:
            wav = wav[:-50]
        
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
                    
        
        return wav.astype(np.float32, copy=False)
