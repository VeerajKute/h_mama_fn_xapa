import torch
import torch.nn as nn

class ExpertHead(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 2)
        )

    def forward(self, x):
        return self.net(x)

class UGMOP(nn.Module):
    def __init__(self, in_text, in_ocr, in_img):
        super().__init__()
        self.text_head = ExpertHead(in_text)
        self.ocr_head = ExpertHead(in_ocr)
        self.img_head = ExpertHead(in_img)
        self.gate = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, z_text, z_ocr, z_img, contradiction):
        lt = self.text_head(z_text)
        lo = self.ocr_head(z_ocr)
        li = self.img_head(z_img)
        # uncertainty proxy: prediction entropy from softmax
        pt = torch.softmax(lt, dim=-1)
        po = torch.softmax(lo, dim=-1)
        pi = torch.softmax(li, dim=-1)
        ut = -(pt * (pt+1e-12).log()).sum(dim=-1)
        uo = -(po * (po+1e-12).log()).sum(dim=-1)
        ui = -(pi * (pi+1e-12).log()).sum(dim=-1)
        max_conf = torch.stack([pt.max(dim=-1).values, po.max(dim=-1).values, pi.max(dim=-1).values], dim=1).max(dim=1).values
        avg_conf = torch.stack([pt.max(dim=-1).values, po.max(dim=-1).values, pi.max(dim=-1).values], dim=1).mean(dim=1)
        # gate input
        g_in = torch.stack([ut, uo, ui, contradiction, max_conf, avg_conf], dim=1)
        w = self.gate(g_in)
        fused = w[:,0:1]*lt + w[:,1:2]*lo + w[:,2:3]*li
        return fused, w
