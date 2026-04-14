
import os
import yaml
import torch
from pathlib import Path
import yaml
from munch import Munch
import onnxruntime as ort
import numpy as np
import random
import librosa
import torch.nn.functional as F
import random
import torch.nn as nn

from collections import OrderedDict
import torch.nn.functional as F
from common_code.styletts2.models import build_model
from common_code.styletts2.Modules.diffusion.sampler import *
from common_code.styletts2.Utils.PLBERT.util import load_plbert
from common_code.styletts2.models import *
from common_code.inference_core import *
import os
import shutil
# from common_code.styletts2.Modules.hifigan import *
from common_code.styletts2.Modules.hifigan_latest import *


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = False




class AdaLayerNormONNX(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        # x: [B, T, C] where C == self.channels
        # s: [B, style_dim]

        # ONNX-compatible slicing instead of chunk
        h = self.fc(s)
        gamma = h[:, :self.channels].unsqueeze(1)
        beta  = h[:, self.channels:].unsqueeze(1)

        # Use fixed self.channels instead of dynamic x.shape[2]
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x


class DurationEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
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

    def forward(self, x, style, text_mask):
        """
        x: [B, C, T] -> float32
        style: [B, S] -> float32
        text_mask: [B, T] -> bool
        """
        # Ensure mask is boolean
        masks = text_mask.bool()  # [B, T]

        # Permute x to [B, T, C]
        x = x.permute(0, 2, 1)  # [B, T, C]

        # Expand style and concatenate
        s = style.unsqueeze(1).expand(x.shape[0], x.shape[1], -1)  # [B, T, S]
        x = torch.cat([x, s], dim=-1)  # [B, T, C+S]

        # Apply mask
        x = x.masked_fill(masks.unsqueeze(-1), 0.0)

        
        # LSTM / AdaLayerNorm blocks
        for block in self.lstms:
            if isinstance(block, AdaLayerNormONNX):
                x = block(x, style)  # [B, T, C]
                # Re-concatenate style to match next LSTM input
                x = torch.cat([x, s], dim=-1)  # [B, T, d_model + sty_dim] = [B, T, 640]
                # Re-apply mask
                x = x.masked_fill(masks.unsqueeze(-1), 0.0)
            else:
                # LSTM expects [B, T, C]
                block.flatten_parameters()
                x, _ = block(x)  # [B, T, d_model]
                x = F.dropout(x, p=self.dropout, training=self.training)

        
        # Return [B, C, T] for consistency
        return x


class Decoder(nn.Module):
    def __init__(self, dim_in=512, F0_channel=512, style_dim=64, dim_out=80, 
                resblock_kernel_sizes = [3,7,11],
                upsample_rates = [10,5,3,2],
                upsample_initial_channel=512,
                resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
                upsample_kernel_sizes=[20,10,6,4]):
        super().__init__()
        
        self.decode = nn.ModuleList()
        
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True))

        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        
        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        
        self.asr_res = nn.Sequential(
            weight_norm(nn.Conv1d(512, 64, kernel_size=1)),
        )
        
        
        self.generator = Generator(style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes)

        
    def forward(self, asr, F0_curve, N, s):
        
        if self.training:
            downlist = [0, 3, 7]
            F0_down = downlist[random.randint(0, 2)]
            downlist = [0, 3, 7, 15]
            N_down = downlist[random.randint(0, 3)]
            if F0_down:
                F0_curve = nn.functional.conv1d(F0_curve.unsqueeze(1), torch.ones(1, 1, F0_down).to('cuda'), padding=F0_down//2).squeeze(1) / F0_down
            if N_down:
                N = nn.functional.conv1d(N.unsqueeze(1), torch.ones(1, 1, N_down).to('cuda'), padding=N_down//2).squeeze(1)  / N_down

        
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N.unsqueeze(1))
        
        x = torch.cat([asr, F0, N], axis=1)
        x = self.encode(x, s)
        
        asr_res = self.asr_res(asr)
        
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
                
        x = self.generator(x, s, F0_curve)
        return x
    


# class Decoder(nn.Module):
#     def __init__(self, dim_in=512, F0_channel=512, style_dim=64, dim_out=80, 
#                 resblock_kernel_sizes = [3,7,11],
#                 upsample_rates = [10,5,3,2],
#                 upsample_initial_channel=512,
#                 resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
#                 upsample_kernel_sizes=[20,10,6,4]):
#         super().__init__()
        
#         self.decode = nn.ModuleList()
        
#         self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        
#         self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
#         self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
#         self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
#         self.decode.append(AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True))

#         # self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        
#         # self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        
#         self.asr_res = nn.Sequential(
#             weight_norm(nn.Conv1d(512, 64, kernel_size=1)),
#         )
        
        
#         self.generator = Generator(style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes)


#     def forward(self, asr,F0_curve, F0, N, s):
        
#         # F0 = self.F0_conv(F0_curve.unsqueeze(1))
#         # N = self.N_conv(N.unsqueeze(1))
        

        
#         x = torch.cat([asr, F0, N], axis=1)
#         x = self.encode(x, s)
        
#         asr_res = self.asr_res(asr)
        
#         res = True
#         for block in self.decode:
#             if res:
#                 x = torch.cat([x, asr_res, F0, N], axis=1)
#             x = block(x, s)
#             if block.upsample_type != "none":
#                 res = False
                
#         x = self.generator(x, s, F0_curve)
#         return x
    


# class Decoder(nn.Module):
#     def __init__(self, dim_in=512, F0_channel=512, style_dim=64, dim_out=80, 
#                 resblock_kernel_sizes = [3,7,11],
#                 upsample_rates = [10,5,3,2],
#                 upsample_initial_channel=512,
#                 resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
#                 upsample_kernel_sizes=[20,10,6,4]):
#         super().__init__()
        
#         self.decode = nn.ModuleList()
        
#         self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        
#         self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
#         self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
#         self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
#         self.decode.append(AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True))

#         self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        
#         self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        
#         self.asr_res = nn.Sequential(
#             weight_norm(nn.Conv1d(512, 64, kernel_size=1)),
#         )
        
        
#         self.generator = Generator(style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes)


#     def forward(self, asr, F0_curve, N, s, asr_mask, f0_mask):
#         """
#         asr:      [B, 512, T]
#         F0_curve: [B, T]
#         N:        [B, T]
#         s:        [B, style_dim]
        
#         asr_mask: [B, T]
#         f0_mask:  [B, T]
#         """

#         # -----------------------------
#         # Apply masking if available
#         # -----------------------------
#         # asr = asr * asr_mask.unsqueeze(1)

#         # F0_curve = F0_curve * f0_mask
#         # N        = N * f0_mask
        
#         # print(asr.shape)
#         # print(F0_curve.shape)
#         # print(N.shape)

#         # ======================================================
#         #  ORIGINAL COMPUTATION STARTS BELOW
#         # ======================================================
        
#         # Apply F0 and Noise convs
#         F0 = self.F0_conv(F0_curve.unsqueeze(1))  # → [B, 1, T']
#         N  = self.N_conv(N.unsqueeze(1))          # → [B, 1, T']

#         # Merge conditioning features
#         x = torch.cat([asr, F0, N], dim=1)

#         # First block
#         x = self.encode(x, s)

#         # ASR residual branch
#         asr_res = self.asr_res(asr)

#         res = True
#         for block in self.decode:
#             if res:
#                 x = torch.cat([x, asr_res, F0, N], dim=1)
#             x = block(x, s)
#             if block.upsample_type != "none":
#                 res = False

#         # Final generation
#         x = self.generator(x, s, F0_curve)
#         return x
    



class ProsodyPredictor(nn.Module):

    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__() 
        self.text_encoder = DurationEncoder(sty_dim=style_dim, 
                                            d_model=d_hid,
                                            nlayers=nlayers, 
                                            dropout=dropout)

        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)


    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        
        batch_size = d.shape[0]
        text_size = d.shape[1]
        
        # predict duration
        input_lengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            d, input_lengths, batch_first=True, enforce_sorted=False)
        
        m = m.to(text_lengths.device).unsqueeze(1)
        
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
        
        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]])

        x_pad[:, :x.shape[1], :] = x
        x = x_pad.to(x.device)
                
        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))
        
        en = (d.transpose(-1, -2) @ alignment)

        return duration.squeeze(-1), en
    
    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(-1, -2))
        
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)

        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        
        return F0.squeeze(1), N.squeeze(1)
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask
    
    
def build_model(args, bert):
    assert args.decoder.type in ['istftnet', 'hifigan'], 'Decoder type unknown'
    
    decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                upsample_rates = args.decoder.upsample_rates,
                upsample_initial_channel=args.decoder.upsample_initial_channel,
                resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=args.decoder.upsample_kernel_sizes) 
        
    text_encoder = TextEncoder(channels=args.hidden_dim, kernel_size=5, depth=args.n_layer, n_symbols=args.n_token)
    
    predictor = ProsodyPredictor(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer, max_dur=args.max_dur, dropout=args.dropout)
    
    style_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # acoustic style encoder
    predictor_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # prosodic style encoder
        
    # define diffusion model
    if args.multispeaker:
        transformer = StyleTransformer1d(channels=args.style_dim*2, 
                                    context_embedding_features=bert.config.hidden_size,
                                    context_features=args.style_dim*2, 
                                    **args.diffusion.transformer)
    else:
        transformer = Transformer1d(channels=args.style_dim*2, 
                                    context_embedding_features=bert.config.hidden_size,
                                    **args.diffusion.transformer)
    
    diffusion = AudioDiffusionConditional(
        in_channels=1,
        embedding_max_length=bert.config.max_position_embeddings,
        embedding_features=bert.config.hidden_size,
        embedding_mask_proba=args.diffusion.embedding_mask_proba, # Conditional dropout of batch elements,
        channels=args.style_dim*2,
        context_features=args.style_dim*2,
    )
    
    diffusion.diffusion = KDiffusion(
        net=diffusion.unet,
        sigma_distribution=LogNormalDistribution(mean = args.diffusion.dist.mean, std = args.diffusion.dist.std),
        sigma_data=args.diffusion.dist.sigma_data, # a placeholder, will be changed dynamically when start training diffusion model
        dynamic_threshold=0.0 
    )
    diffusion.diffusion.net = transformer
    diffusion.unet = transformer

    
    nets = Munch(
            bert=bert,
            bert_encoder=nn.Linear(bert.config.hidden_size, args.hidden_dim),

            predictor=predictor,
            decoder=decoder,
            text_encoder=text_encoder,

            predictor_encoder=predictor_encoder,
            style_encoder=style_encoder,
            diffusion=diffusion,

       )
    
    return nets



def build_model(args, bert):
    assert args.decoder.type in ['istftnet', 'hifigan'], 'Decoder type unknown'
    
    # decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
    #             resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
    #             upsample_rates = args.decoder.upsample_rates,
    #             upsample_initial_channel=args.decoder.upsample_initial_channel,
    #             resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
    #             upsample_kernel_sizes=args.decoder.upsample_kernel_sizes) 
    
    
    prep_decoder = Decoder_preprocessing()
    
    decoder = Decoder_block(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                upsample_rates = args.decoder.upsample_rates,
                upsample_initial_channel=args.decoder.upsample_initial_channel,
                resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=args.decoder.upsample_kernel_sizes)
    
    generator = Generator_block(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                upsample_rates = args.decoder.upsample_rates,
                upsample_initial_channel=args.decoder.upsample_initial_channel,
                resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=args.decoder.upsample_kernel_sizes)
        
    text_encoder = TextEncoder(channels=args.hidden_dim, kernel_size=5, depth=args.n_layer, n_symbols=args.n_token)
    
    predictor = ProsodyPredictor(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer, max_dur=args.max_dur, dropout=args.dropout)
    
    style_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # acoustic style encoder
    predictor_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # prosodic style encoder
        
    # define diffusion model
    if args.multispeaker:
        transformer = StyleTransformer1d(channels=args.style_dim*2, 
                                    context_embedding_features=bert.config.hidden_size,
                                    context_features=args.style_dim*2, 
                                    **args.diffusion.transformer)
    else:
        transformer = Transformer1d(channels=args.style_dim*2, 
                                    context_embedding_features=bert.config.hidden_size,
                                    **args.diffusion.transformer)
    
    diffusion = AudioDiffusionConditional(
        in_channels=1,
        embedding_max_length=bert.config.max_position_embeddings,
        embedding_features=bert.config.hidden_size,
        embedding_mask_proba=args.diffusion.embedding_mask_proba, # Conditional dropout of batch elements,
        channels=args.style_dim*2,
        context_features=args.style_dim*2,
    )
    
    diffusion.diffusion = KDiffusion(
        net=diffusion.unet,
        sigma_distribution=LogNormalDistribution(mean = args.diffusion.dist.mean, std = args.diffusion.dist.std),
        sigma_data=args.diffusion.dist.sigma_data, # a placeholder, will be changed dynamically when start training diffusion model
        dynamic_threshold=0.0 
    )
    diffusion.diffusion.net = transformer
    diffusion.unet = transformer

    
    nets = Munch(
            prep_decoder=prep_decoder,
            bert=bert,
            bert_encoder=nn.Linear(bert.config.hidden_size, args.hidden_dim),

            predictor=predictor,
            decoder=decoder,
            generator=generator,
            text_encoder=text_encoder,

            predictor_encoder=predictor_encoder,
            style_encoder=style_encoder,
            diffusion=diffusion,

       )
    
    return nets
    
