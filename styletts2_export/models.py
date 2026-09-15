"""ONNX-friendly rebuild of the StyleTTS2 prosody stack, and the model factory.

Two modules here shadow their counterparts in
``common_code/styletts2/models.py`` on purpose:

``DurationEncoder``
    The training version calls ``pack_padded_sequence`` and permutes between
    ``[T, B, C]`` and ``[B, T, C]`` several times per layer. Packed sequences do
    not survive ONNX tracing, and the permutes make the traced graph depend on
    concrete lengths. This version keeps everything in ``[B, T, C]`` and masks
    instead of packing — identical numerics for a single un-padded sequence,
    which is all the Triton frontend ever sends.

``AdaLayerNormONNX``
    Replaces ``torch.chunk`` with explicit slicing and uses the statically
    known channel count in ``F.layer_norm`` rather than reading it off the
    input shape, both of which the exporter needs to fold into constants.

``build_model`` returns a ``Munch`` whose decoder is split three ways
(``prep_decoder`` / ``decoder`` / ``generator``) to match the Triton ensemble;
see ``model_loader`` for how a checkpoint is routed into that split.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from munch import Munch

from common_code.styletts2.models import LinearNorm, StyleEncoder, TextEncoder
from common_code.styletts2.Modules.diffusion.diffusion import AudioDiffusionConditional
from common_code.styletts2.Modules.diffusion.modules import StyleTransformer1d, Transformer1d
from common_code.styletts2.Modules.diffusion.sampler import KDiffusion, LogNormalDistribution
from common_code.styletts2.Modules.hifigan_latest import (
    AdainResBlk1d,
    Decoder_block,
    Decoder_preprocessing,
    Generator_block,
)

__all__ = [
    "AdaLayerNormONNX",
    "DurationEncoder",
    "ProsodyPredictor",
    "build_model",
]


class AdaLayerNormONNX(nn.Module):
    """Style-conditioned layer norm, written so the exporter can fold shapes."""

    def __init__(self, style_dim: int, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] with C == self.channels;  s: [B, style_dim]
        h = self.fc(s)
        gamma = h[:, : self.channels].unsqueeze(1)
        beta = h[:, self.channels :].unsqueeze(1)

        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        return (1 + gamma) * x + beta


class DurationEncoder(nn.Module):
    """Alternating BiLSTM / AdaLayerNorm stack over the phoneme sequence.

    Input ``x`` is ``[B, C, T]``; output is ``[B, T, C + style_dim]`` — the
    style vector stays concatenated because the duration LSTM downstream
    expects it (``d_hid + style_dim`` inputs).
    """

    def __init__(self, sty_dim: int, d_model: int, nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(
                nn.LSTM(
                    d_model + sty_dim,
                    d_model // 2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout,
                )
            )
            self.lstms.append(AdaLayerNormONNX(sty_dim, d_model))

        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x: torch.Tensor, style: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] · style: [B, S] · text_mask: [B, T], True where padded."""
        masks = text_mask.bool().unsqueeze(-1)          # [B, T, 1]

        x = x.permute(0, 2, 1)                          # [B, T, C]
        style_seq = style.unsqueeze(1).expand(x.shape[0], x.shape[1], -1)
        # The cat also promotes an fp16 x back to fp32; bert_encoder emits fp16.
        x = torch.cat([x, style_seq], dim=-1)
        x = x.masked_fill(masks, 0.0)

        for block in self.lstms:
            if isinstance(block, AdaLayerNormONNX):
                x = block(x, style)                     # [B, T, d_model]
                x = torch.cat([x, style_seq], dim=-1)   # re-attach for next LSTM
                x = x.masked_fill(masks, 0.0)
            else:
                block.flatten_parameters()
                x, _ = block(x)                         # [B, T, d_model]
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class ProsodyPredictor(nn.Module):
    """Predicts per-phoneme duration, and F0/energy curves per mel frame.

    Only the pieces the export needs are implemented: ``text_encoder`` + ``lstm``
    + ``duration_proj`` for durations, and ``F0Ntrain`` for the curves. The
    training-time ``forward`` (which needs an alignment matrix) is deliberately
    absent — the alignment is built in the Triton ``int_steps_2`` step instead.
    """

    def __init__(self, style_dim: int, d_hid: int, nlayers: int, max_dur: int = 50, dropout: float = 0.1):
        super().__init__()
        self.text_encoder = DurationEncoder(
            sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout
        )

        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)

        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)

        self.F0 = nn.ModuleList([
            AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout),
        ])
        self.N = nn.ModuleList([
            AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout),
        ])

        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def F0Ntrain(self, x: torch.Tensor, s: torch.Tensor):
        """x: [B, d_hid + style_dim, T] · s: [B, style_dim] -> two [B, 2T] curves.

        The middle block of each branch upsamples by 2, so the outputs are at
        twice the input frame rate.
        """
        x, _ = self.shared(x.transpose(-1, -2))

        f0 = x.transpose(-1, -2)
        for block in self.F0:
            f0 = block(f0, s)
        f0 = self.F0_proj(f0)

        n = x.transpose(-1, -2)
        for block in self.N:
            n = block(n, s)
        n = self.N_proj(n)

        return f0.squeeze(1), n.squeeze(1)

    @staticmethod
    def length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(lengths.max()).unsqueeze(0)
        positions = positions.expand(lengths.shape[0], -1).type_as(lengths)
        return torch.gt(positions + 1, lengths.unsqueeze(1))


def build_model(args, bert) -> Munch:
    """Build the export-shaped StyleTTS2 model from a config's ``model_params``.

    ``args`` is the munched ``model_params`` block; ``bert`` is a loaded PLBERT.
    Weights are left randomly initialised — see ``model_loader.load_checkpoint``.
    """
    if args.decoder.type not in ("istftnet", "hifigan"):
        raise ValueError(
            f"unknown decoder type '{args.decoder.type}'; expected 'istftnet' or 'hifigan'"
        )

    decoder_kwargs = dict(
        dim_in=args.hidden_dim,
        style_dim=args.style_dim,
        dim_out=args.n_mels,
        resblock_kernel_sizes=args.decoder.resblock_kernel_sizes,
        upsample_rates=args.decoder.upsample_rates,
        upsample_initial_channel=args.decoder.upsample_initial_channel,
        resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
        upsample_kernel_sizes=args.decoder.upsample_kernel_sizes,
    )

    # The training-time decoder is split into three Triton models so the
    # strided convs and the vocoder can be scheduled (and quantised) apart.
    prep_decoder = Decoder_preprocessing()
    decoder = Decoder_block(**decoder_kwargs)
    generator = Generator_block(**decoder_kwargs)

    text_encoder = TextEncoder(
        channels=args.hidden_dim, kernel_size=5, depth=args.n_layer, n_symbols=args.n_token
    )
    predictor = ProsodyPredictor(
        style_dim=args.style_dim,
        d_hid=args.hidden_dim,
        nlayers=args.n_layer,
        max_dur=args.max_dur,
        dropout=args.dropout,
    )

    style_encoder = StyleEncoder(          # acoustic
        dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim
    )
    predictor_encoder = StyleEncoder(      # prosodic
        dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim
    )

    if args.multispeaker:
        transformer = StyleTransformer1d(
            channels=args.style_dim * 2,
            context_embedding_features=bert.config.hidden_size,
            context_features=args.style_dim * 2,
            **args.diffusion.transformer,
        )
    else:
        transformer = Transformer1d(
            channels=args.style_dim * 2,
            context_embedding_features=bert.config.hidden_size,
            **args.diffusion.transformer,
        )

    diffusion = AudioDiffusionConditional(
        in_channels=1,
        embedding_max_length=bert.config.max_position_embeddings,
        embedding_features=bert.config.hidden_size,
        embedding_mask_proba=args.diffusion.embedding_mask_proba,
        channels=args.style_dim * 2,
        context_features=args.style_dim * 2,
    )
    diffusion.diffusion = KDiffusion(
        net=diffusion.unet,
        sigma_distribution=LogNormalDistribution(
            mean=args.diffusion.dist.mean, std=args.diffusion.dist.std
        ),
        # Overwritten from the checkpoint's own value during training; kept
        # here so the denoiser preconditioning matches what was trained.
        sigma_data=args.diffusion.dist.sigma_data,
        dynamic_threshold=0.0,
    )
    diffusion.diffusion.net = transformer
    diffusion.unet = transformer

    return Munch(
        prep_decoder=prep_decoder,
        decoder=decoder,
        generator=generator,
        bert=bert,
        bert_encoder=nn.Linear(bert.config.hidden_size, args.hidden_dim),
        predictor=predictor,
        text_encoder=text_encoder,
        predictor_encoder=predictor_encoder,
        style_encoder=style_encoder,
        diffusion=diffusion,
    )
