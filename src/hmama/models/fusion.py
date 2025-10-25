import torch
import torch.nn as nn
import torch.nn.functional as F

class CoAttentionFusion(nn.Module):
    def __init__(self, dim_text=256, dim_img=512, fused_dim=512, hidden=256):
        super().__init__()
        # Project text and image separately to hidden
        self.text_proj = nn.Linear(dim_text, hidden)
        self.img_proj = nn.Linear(dim_img, hidden)
        self.co_att = nn.Linear(hidden, 1)
        # gating MLP to combine global cues
        self.gate = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # weights for [text,image]
            nn.Softmax(dim=-1)
        )
        # projection after concatenation
        self.fused_proj = nn.Linear(hidden * 2, fused_dim)

    def forward(self, z_text, z_img, contradiction, aigc_flag=None):
        # z_text: (B, dim_text), z_img: (B, dim_img)
        t = torch.tanh(self.text_proj(z_text))  # (B, hidden)
        v = torch.tanh(self.img_proj(z_img))    # (B, hidden)
        
        # co-attention score
        joint = t * v
        att = torch.sigmoid(self.co_att(joint)).squeeze(-1)  # (B,)
        avg_att = att.mean(dim=0, keepdim=True)
        
        # gate inputs: [mean_att, contradiction_mean, aigc_flag_mean]
        c = contradiction.mean(dim=0, keepdim=True) if contradiction.dim() > 0 else contradiction.unsqueeze(0)
        aig = aigc_flag.mean(dim=0, keepdim=True) if aigc_flag is not None else torch.zeros_like(c)
        g_in = torch.cat([avg_att.unsqueeze(0), c.unsqueeze(0), aig.unsqueeze(0)], dim=0).squeeze(-1)
        
        # reshape to (B,3): tile stats to batch
        g_tile = g_in.unsqueeze(0).repeat(z_text.size(0),1)
        w = self.gate(g_tile)  # (B,2)
        
        # weighted t and v
        t_weighted = w[:,0:1] * t
        v_weighted = w[:,1:2] * v
        
        # fused vector: concatenate then project
        fused = self.fused_proj(torch.cat([t_weighted, v_weighted], dim=1))  # (B, fused_dim)
        
        return fused, w
