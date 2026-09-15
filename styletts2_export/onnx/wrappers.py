"""ONNX-traceable wrappers around StyleTTS2 sub-models.

These exist because the training-time modules use patterns the ONNX exporter
cannot represent — multiple return values, ``torch.chunk`` on dynamic shapes,
Python control flow over tensor values. Each wrapper keeps the numerics
identical and only reshapes the interface.

Anything defined here is traced, so keep it free of data-dependent branching:
a Python ``if`` on a tensor bakes one branch into the graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "BertEncoderWrapper",
    "PredLSTMWrapper",
    "ProsodyPredictorONNXWrapper",
    "IntermediateSteps1",
    "KDiffusionDenoiserONNX",
    "KarrasScheduleONNX",
    "ADPM2ONNX",
    "DiffusionONNX",
]


class BertEncoderWrapper(nn.Module):
    """PLBERT projection, transposed to the ``[B, C, T]`` the predictor wants.

    Runs in fp16 (the export casts the weights) but accepts fp32, matching the
    Triton contract: ``BERT_DUR`` fp32 in, ``D_EN`` fp16 out.
    """

    def __init__(self, bert_encoder: nn.Module):
        super().__init__()
        self.bert_encoder = bert_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float16)
        return self.bert_encoder(x).transpose(1, 2)


class PredLSTMWrapper(nn.Module):
    """Duration LSTM, dropping the ``(h, c)`` state that ONNX would expose.

    Kept in fp32 for stability; only the output is narrowed to fp16 for the
    downstream TensorRT projection.
    """

    def __init__(self, lstm: nn.Module):
        super().__init__()
        self.lstm = lstm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out.to(torch.float16)


class ProsodyPredictorONNXWrapper(nn.Module):
    """Exposes ``ProsodyPredictor.F0Ntrain`` as a plain ``forward``."""

    def __init__(self, prosody_predictor: nn.Module):
        super().__init__()
        self.predictor = prosody_predictor

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        # x: [B, C, T], s: [B, style_dim]
        return self.predictor.F0Ntrain(x, s)


class IntermediateSteps1(nn.Module):
    """Blends the diffusion-predicted style with the reference speaker's.

    ``s_pred`` packs both halves: ``[:128]`` is the acoustic style and
    ``[128:]`` the prosodic one. ``alpha`` weights the first, ``beta`` the
    second, each against the corresponding half of ``ref_s``.

    Also emits the padding mask for the phoneme sequence, which is all-False
    here: the frontend sends one un-batched sequence, so nothing is padded.
    """

    @staticmethod
    def length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
        """``True`` at padded positions, shape ``[B, max(lengths)]``."""
        max_len = lengths.max()
        positions = torch.arange(max_len, device=lengths.device)
        positions = positions.unsqueeze(0).expand(lengths.shape[0], -1)
        return torch.gt(positions + 1, lengths.unsqueeze(1))

    def forward(self, alpha, beta, s_pred, ref_s, t_en):
        alpha_b = alpha.view(-1, 1)
        beta_b = beta.view(-1, 1)

        ref = s_pred[:, :128]
        s = s_pred[:, 128:]

        ref = alpha_b * ref + (1 - alpha_b) * ref_s[:, :128]
        s = beta_b * s + (1 - beta_b) * ref_s[:, 128:]

        batch_size = s_pred.size(0)
        seq_len = t_en.size(-1)
        input_lengths = torch.full(
            (batch_size,), seq_len, dtype=torch.long, device=t_en.device
        )
        text_mask_pred = self.length_to_mask(input_lengths)

        return ref, s, text_mask_pred


# ─────────────────────────────────────────────────────────────────────────────
#  Style diffusion, unrolled for ONNX
#
#  Mirrors KDiffusion + ADPM2Sampler + KarrasSchedule from
#  common_code/styletts2/Modules/diffusion/.
#
#  The sampler loop runs in Python, so tracing unrolls it and the step count
#  becomes part of the graph's structure. `num_steps` is therefore a
#  constructor argument, not a forward input -- passing it as a tensor would be
#  a lie, since the tracer would fold it away and emit a graph that silently
#  ignores whatever the caller sent. To change the step count, re-export with a
#  different `export.diffusion_num_steps`. Making it a true runtime input needs
#  the loop rewritten as an ONNX `Loop` op.
# ─────────────────────────────────────────────────────────────────────────────

class KDiffusionDenoiserONNX(nn.Module):
    """One denoising call with the Karras preconditioning applied inline."""

    def __init__(self, net: nn.Module, sigma_data: float):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data

    def forward(self, x_noisy, sigma, bert_dur, features):
        if sigma.ndim == 0:
            sigma = sigma.expand(x_noisy.shape[0])
        sigma = sigma.view(-1, 1, 1)

        sigma_data = self.sigma_data
        variance = sigma_data ** 2 + sigma ** 2

        c_skip = (sigma_data ** 2) / variance
        c_out = sigma * sigma_data / torch.sqrt(variance)
        c_in = 1.0 / torch.sqrt(variance)
        c_noise = torch.log(sigma) * 0.25

        x_pred = self.net(
            x=c_in * x_noisy,
            time=c_noise.squeeze(-1).squeeze(-1),
            embedding=bert_dur,
            features=features,
        )
        return c_skip * x_noisy + c_out * x_pred


class KarrasScheduleONNX(nn.Module):
    """Karras et al. 2022, eq. 5 — the noise levels to step through."""

    def __init__(self, sigma_min: float, sigma_max: float, rho: float):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def forward(self, num_steps: int, device=None) -> torch.Tensor:
        n = int(num_steps)
        if n < 2:
            raise ValueError(
                f"the Karras schedule divides by (num_steps - 1); "
                f"num_steps must be >= 2, got {n}"
            )

        steps = torch.arange(n, device=device, dtype=torch.float32)
        rho_inv = 1.0 / self.rho
        sigmas = (
            self.sigma_max ** rho_inv
            + (steps / (n - 1)) * (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)
        ) ** self.rho
        # Trailing zero terminates the schedule, matching KarrasSchedule.
        return torch.cat([sigmas, torch.zeros(1, device=sigmas.device)], dim=0)


class ADPM2ONNX(nn.Module):
    """Ancestral DPM-Solver++(2M), unrolled.

    Faithful to ``ADPM2Sampler``, including its ``range(num_steps - 1)`` loop:
    the trailing zero sigma is produced by the schedule but never stepped to.
    """

    def __init__(self, denoiser: nn.Module, rho: float = 1.0, clamp: bool = False):
        super().__init__()
        self.denoiser = denoiser
        self.rho = rho
        self.clamp = clamp

    def forward(self, noise, num_steps: int, sigmas, bert_dur, features):
        x = sigmas[0] * noise

        for i in range(int(num_steps) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            sigma_up = torch.sqrt(
                torch.clamp(
                    sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / (sigma ** 2),
                    min=0.0,
                )
            )
            sigma_down = torch.sqrt(torch.clamp(sigma_next ** 2 - sigma_up ** 2, min=0.0))
            sigma_mid = (
                (sigma ** (1 / self.rho) + sigma_down ** (1 / self.rho)) / 2
            ) ** self.rho

            denoised = self.denoiser(x, sigma, bert_dur, features)
            d = (x - denoised) / sigma

            x_mid = x + d * (sigma_mid - sigma)
            denoised_mid = self.denoiser(x_mid, sigma_mid, bert_dur, features)
            d_mid = (x_mid - denoised_mid) / sigma_mid

            x = x + d_mid * (sigma_down - sigma)
            x = x + torch.randn_like(x) * sigma_up      # ancestral noise

        return torch.clamp(x, -1.0, 1.0) if self.clamp else x


class DiffusionONNX(nn.Module):
    """Full style-sampling graph: schedule, sampler and denoiser in one module.

    ``num_steps`` is fixed at construction because the sampler loop is unrolled
    during tracing — see the section comment above. The exported graph takes
    ``(BERT_DUR, REF_S)`` and returns ``S_PRED``.
    """

    def __init__(self, unet, sigma_data, sigma_min, sigma_max, rho, clamp, num_steps: int):
        super().__init__()
        self.num_steps = int(num_steps)
        self.schedule = KarrasScheduleONNX(sigma_min, sigma_max, rho)
        self.denoiser = KDiffusionDenoiserONNX(net=unet, sigma_data=sigma_data)
        self.sampler = ADPM2ONNX(denoiser=self.denoiser, clamp=clamp)

    def forward(self, bert_dur, features):
        batch_size = bert_dur.shape[0]
        noise = torch.randn(
            (batch_size, 256), device=features.device, dtype=torch.float32
        ).unsqueeze(1)

        sigmas = self.schedule(self.num_steps, device=features.device)
        output = self.sampler(noise, self.num_steps, sigmas, bert_dur, features)

        if output.ndim == 3 and output.shape[1] == 1:
            output = output.squeeze(1)
        return output
