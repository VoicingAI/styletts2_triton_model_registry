import torch
import torch.nn as nn

# Wrap transpose inside a small nn.Module for export
class BertEncoderWrapper(torch.nn.Module):
    def __init__(self, bert_encoder):
        super().__init__()
        self.bert_encoder = bert_encoder
    def forward(self, x):
        x = x.to(torch.float16) 
        return self.bert_encoder(x).transpose(1, 2)  # B x C x T



# Wrap the model to return only the main output
class PredLSTMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out, _ = self.model(x)  # discard residuals
        out = out.to(torch.float16) 
        return out



class ProsodyPredictorONNXWrapper(torch.nn.Module):
    def __init__(self, prosody_model):
        super().__init__()
        self.model = prosody_model

    def forward(self, x, s):
        # x: [B, C, T], s: [B, style_dim]
        F0_out, N_out = self.model.F0Ntrain(x, s)
        return F0_out, N_out



class IntermediateSteps1(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def length_to_mask(lengths):
        """
        lengths: (B,) LongTensor of sequence lengths
        returns: mask of shape (B, max_len)
        """
        
        max_len = lengths[0]
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(lengths.shape[0], -1)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask

    def forward(self, alpha, beta, s_pred, ref_s, t_en):
        
        alpha_b = alpha.view(-1, 1)  # Reshape to [B, 1]
        beta_b = beta.view(-1, 1)    # Reshape to [B, 1]
        
        
        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha_b * ref + (1 - alpha_b) * ref_s[:, :128]
        s = beta_b * s + (1 - beta_b) * ref_s[:, 128:]

        # Compute input lengths dynamically using tensor ops
        batch_size = s_pred.size(0)
        seq_len = t_en.size(-1)  # tensor-aware
        input_lengths_pred = torch.full((batch_size,), seq_len, dtype=torch.long, device=t_en.device)
        text_mask_pred = self.length_to_mask(input_lengths_pred)

        return ref, s, text_mask_pred



class KDiffusionDenoiserONNX(nn.Module):
    def __init__(self, net: nn.Module, sigma_data: float):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data

    def forward(self, x_noisy, sigma, bert_dur, features):
        """
        x_noisy: [B, C, T]
        sigma:   scalar tensor OR [B]
        """

        # Ensure sigma is [B]
        if sigma.ndim == 0:
            sigma = sigma.expand(x_noisy.shape[0])

        sigma = sigma.view(-1, 1, 1)

        sigma_data = self.sigma_data

        # === scale weights (INLINE get_scale_weights) ===
        c_skip = (sigma_data ** 2) / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / torch.sqrt(sigma_data ** 2 + sigma ** 2)
        c_in = 1.0 / torch.sqrt(sigma_data ** 2 + sigma ** 2)
        c_noise = torch.log(sigma) * 0.25

        # === UNet forward ===
        x_pred = self.net(
            x=c_in * x_noisy,
            time=c_noise.squeeze(-1).squeeze(-1),
            embedding=bert_dur,
            features=features,
        )

        # === skip connection ===
        x_denoised = c_skip * x_noisy + c_out * x_pred
        return x_denoised


class KarrasScheduleONNX(nn.Module):
    def __init__(self, sigma_min: float, sigma_max: float, rho: float):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def forward(self, num_steps: torch.Tensor):
        # num_steps: scalar int tensor
        n = num_steps.to(torch.int64)

        steps = torch.arange(
            n, device=num_steps.device, dtype=torch.float32
        )

        rho_inv = 1.0 / self.rho

        sigmas = (
            self.sigma_max ** rho_inv
            + (steps / (n - 1))
            * (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)
        ) ** self.rho

        # Append final zero
        sigmas = torch.cat(
            [sigmas, torch.zeros(1, device=sigmas.device)], dim=0
        )

        return sigmas

class ADPM2ONNX(nn.Module):
    def __init__(self, denoiser: nn.Module, rho: float = 1.0, clamp: bool = False):
        super().__init__()
        self.denoiser = denoiser
        self.rho = rho
        self.clamp = clamp

    def forward(self, noise,num_steps, sigmas, bert_dur, features):
        x = sigmas[0] * noise

        for i in range(num_steps - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            # --- get_sigmas ---
            sigma_up = torch.sqrt(torch.clamp(
                    sigma_next**2 * (sigma**2 - sigma_next**2) / (sigma**2),
                    min=0.0,
                ))
            sigma_down = torch.sqrt(torch.clamp(sigma_next**2 - sigma_up**2, min=0.0))
            
            sigma_mid = ((sigma ** (1 / self.rho) + sigma_down ** (1 / self.rho)) / 2) ** self.rho
            
            # --- step ---
            denoised = self.denoiser(x, sigma, bert_dur, features)
            d = (x - denoised) / sigma

            x_mid = x + d * (sigma_mid - sigma)
            denoised_mid = self.denoiser(x_mid, sigma_mid, bert_dur, features)
            d_mid = (x_mid - denoised_mid) / sigma_mid

            x = x + d_mid * (sigma_down - sigma)

            # randomness
            x = x + torch.randn_like(x) * sigma_up

        return torch.clamp(x, -1.0, 1.0) if self.clamp else x

class DiffusionONNX(nn.Module):
    def __init__(self, unet,sigma_data, sigma_min, sigma_max, rho, clamp):
        super().__init__()
        self.schedule = KarrasScheduleONNX(sigma_min, sigma_max, rho)
        
        self.denoiser = KDiffusionDenoiserONNX(
            net=unet,
            sigma_data=sigma_data,
        )

        self.sampler = ADPM2ONNX(denoiser=self.denoiser, clamp=clamp)

    def forward(self, num_steps, bert_dur, features):
        if num_steps.ndim == 1:
            num_steps = num_steps[0]
            
        batch_size = bert_dur.shape[0]
        noise = torch.randn((batch_size, 256), device=features.device, dtype=torch.float32).unsqueeze(1)

        sigmas = self.schedule(num_steps)
        output = self.sampler(noise,num_steps, sigmas, bert_dur, features)
    
        if output.ndim == 3 and output.shape[1] == 1:
            output = output.squeeze(1)
            
        return output

class ProsodyPredictorMerged(nn.Module):
    def __init__(self, lstm, duration_proj):
        super().__init__()
        self.lstm = lstm
        self.duration_proj = duration_proj

    def forward(self, d_in):
        # 1. LSTM remains FP32 for stability
        lstm_out, _ = self.lstm(d_in)
        
        # 2. Duration Projection
        duration = self.duration_proj(lstm_out)
        
        # 3. Explicitly return as FP16
        return duration.half()
    
class StyleTTS2ProsodyBlock(nn.Module):
    def __init__(self, encoder, lstm, duration_proj):
        super().__init__()
        self.encoder = encoder
        self.lstm = lstm
        self.duration_proj = duration_proj

    def forward(self, d_en, s, text_mask):
        # 1. Text Encoder: [B, C, T] -> [B, T, C]
        # In StyleTTS2, this often involves a transpose internally
        d_out = self.encoder(d_en, s, text_mask)
        
        # 2. LSTM: [B, T, C] -> [B, T, Hidden]
        # We take only the sequence output [0], ignoring (h, c)
        lstm_out, _ = self.lstm(d_out)
        
        # 3. Duration Projection: [B, T, Hidden] -> [B, T, 1]
        duration = self.duration_proj(lstm_out)
        
        # Return both as requested for the Triton ensemble
        # D_OUT remains FP32 for next steps; DURATION cast to FP16
        return d_out, duration.half()

class AcousticMegaWrapper(nn.Module):
    def __init__(self, text_encoder, bert, bert_encoder):
        super().__init__()
        self.text_encoder = text_encoder
        self.bert = bert
        self.bert_encoder = bert_encoder

    def forward(self, tokens, input_lengths, text_mask, attention_mask):
        # 1. Parallel Branch: Text Encoder
        t_en = self.text_encoder(tokens, input_lengths, text_mask)
        
        # 2. Parallel Branch: BERT
        bert_dur = self.bert(tokens, attention_mask)
        
        # 3. Sequential Branch: BERT Encoder
        # BERT_DUR flows into BERT Encoder
        d_en = self.bert_encoder(bert_dur)
        
        # Return all three as requested
        return t_en, bert_dur, d_en.half().transpose(1, 2) 