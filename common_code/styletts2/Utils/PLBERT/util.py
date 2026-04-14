import os
import yaml
import torch
from transformers import AlbertConfig, AlbertModel

class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state

def load_plbert(log_dir):
    config_path = os.path.join(log_dir, "config.yml")
    plbert_config = yaml.safe_load(open(config_path))
    
    albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
    bert = CustomAlbert(albert_base_configuration)

    files = os.listdir(log_dir)
    ckpts = [f for f in files if f.startswith("step_") and f.endswith(".t7")]
    
    if not ckpts:
        raise FileNotFoundError(f"No PLBERT checkpoint found in {log_dir}")
        
    iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts]
    iters = sorted(iters)[-1]

    checkpoint = torch.load(os.path.join(log_dir, f"step_{iters}.t7"), map_location='cpu')
    state_dict = checkpoint['net']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        if name.startswith('encoder.'):
            name = name[8:] # remove `encoder.`
        new_state_dict[name] = v

    # --- FIX: Make robust like original codebase ---
    try:
        del new_state_dict["embeddings.position_ids"]
    except KeyError:
        pass
        
    bert.load_state_dict(new_state_dict, strict=False)
    
    return bert