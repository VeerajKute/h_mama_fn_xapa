from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn

class ResNetEncoder:
    def __init__(self, device='cpu'):
        self.device = device
        res = models.resnet18(pretrained=True)
        res.fc = nn.Identity()
        self.model = res.to(device)
        self.transform = T.Compose([T.Resize((224,224)), T.ToTensor()])

    @torch.no_grad()
    def encode_image(self, pil_images):
        # pil_images: list[PIL.Image]
        import torch
        imgs = [self.transform(im).unsqueeze(0) for im in pil_images]
        batch = torch.cat(imgs, dim=0).to(self.model.fc.weight.device)
        feats = self.model(batch)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats

    @torch.no_grad()
    def encode_text(self, texts):
        # placeholder: not used
        return None

    @torch.no_grad()
    def contradiction(self, texts, pil_images):
        # simple heuristic: use CLIP-like similarity stub using image features and text length
        img_feats = self.encode_image(pil_images)
        tlen = torch.tensor([len(t.split()) for t in texts], dtype=torch.float32)
        tlen = (tlen - tlen.mean())/ (tlen.std()+1e-8)
        tproj = tlen.unsqueeze(1).repeat(1, img_feats.size(1)).to(img_feats.device) * 0.01
        sim = (img_feats * tproj).sum(dim=1)
        # map to [0,1]
        c = (1 - torch.tanh(sim)).clamp(0,1)
        return c
