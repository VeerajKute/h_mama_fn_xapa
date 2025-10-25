import torch
import torch.nn as nn
import torch.nn.functional as F

def enable_mc_dropout(module):
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.train()

class ExpertHead(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, 2)
        )

    def forward(self, x):
        return self.net(x)

class UGMOP_Adaptive(nn.Module):
    """
    Uncertainty-Guided Modality Prioritization with MC-Dropout uncertainty estimation.
    Inputs:
      - z_text, z_ocr, z_img: feature tensors (B, D)
      - contradiction: tensor (B,) in [0,1]
      - aigc_flag: tensor (B,) bool/int indicating suspicious AIGC
    Output:
      - fused_logits (B,2), gate_weights (B,3)
    """
    def __init__(self, in_text, in_ocr, in_img, hidden=256, mc_passes=8, device='cpu'):
        super().__init__()
        self.text_head = ExpertHead(in_text, hidden=hidden)
        self.ocr_head = ExpertHead(in_ocr, hidden=hidden)
        self.img_head = ExpertHead(in_img, hidden=hidden)
        self.gate = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )
        self.mc_passes = mc_passes
        self.device = device

    def _mc_uncertainty(self, head, x):
        # run multiple stochastic forward passes with Dropout active
        head.train()
        logits = []
        with torch.no_grad():
            for _ in range(self.mc_passes):
                logits.append(F.softmax(head(x), dim=-1))
        P = torch.stack(logits, dim=0).mean(dim=0)  # (B, C)
        entropy = -(P * (P + 1e-12).log()).sum(dim=-1)  # (B,)
        return entropy

    def forward(self, z_text, z_ocr, z_img, contradiction, aigc_flag=None):
        # single deterministic logits
        lt = self.text_head(z_text)    # (B,2)
        lo = self.ocr_head(z_ocr)
        li = self.img_head(z_img)
        # compute uncertainties
        ut = self._mc_uncertainty(self.text_head, z_text)
        uo = self._mc_uncertainty(self.ocr_head,  z_ocr)
        ui = self._mc_uncertainty(self.img_head,  z_img)
        # normalize uncertainties to [0,1]
        unc = torch.stack([ut, uo, ui], dim=1)
        unc = (unc - unc.min(dim=1, keepdim=True).values) / (unc.max(dim=1, keepdim=True).values + 1e-8)
        max_conf = torch.stack([lt.softmax(-1).max(dim=-1).values,
                                lo.softmax(-1).max(dim=-1).values,
                                li.softmax(-1).max(dim=-1).values], dim=1).max(dim=1).values
        avg_conf = torch.stack([lt.softmax(-1).max(dim=-1).values,
                                lo.softmax(-1).max(dim=-1).values,
                                li.softmax(-1).max(dim=-1).values], dim=1).mean(dim=1)
        # gate input: ut, uo, ui, contradiction, max_conf, avg_conf
        g_in = torch.stack([unc[:,0], unc[:,1], unc[:,2], contradiction, max_conf, avg_conf], dim=1)
        w = self.gate(g_in)  # (B,3)
        # adjust weights if aigc_flag (downweight text if suspicious)
        if aigc_flag is not None:
            # aigc_flag: (B,) float 0/1 -> reduce text weight by 0.5 when flagged
            factor = (1.0 - 0.5 * aigc_flag).unsqueeze(1)
            w = w * torch.cat([factor, torch.ones_like(factor), torch.ones_like(factor)], dim=1)
            w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
        fused = w[:,0:1]*lt + w[:,1:2]*lo + w[:,2:3]*li
        return fused, w
