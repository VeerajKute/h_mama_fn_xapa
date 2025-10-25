import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class CLIPEncoder:
    def __init__(self, model_name='openai/clip-vit-base-patch32', device='cpu'):
        self.device = device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.proc = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_image(self, pil_images):
        inputs = self.proc(images=pil_images, return_tensors='pt', padding=True).to(self.device)
        feats = self.model.get_image_features(**inputs)
        return torch.nn.functional.normalize(feats, p=2, dim=-1)

    @torch.no_grad()
    def encode_text(self, texts):
        inputs = self.proc(text=texts, return_tensors='pt', padding=True).to(self.device)
        feats = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(feats, p=2, dim=-1)

    @torch.no_grad()
    def similarity(self, texts, pil_images):
        t = self.encode_text(texts)
        i = self.encode_image(pil_images)
        sims = (t * i).sum(dim=1)  # cosine since normalized
        return sims

    @torch.no_grad()
    def contradiction_score(self, texts, pil_images):
        # contradiction = 1 - cosine_similarity mapped to [0,1]
        sim = self.similarity(texts, pil_images)
        c = (1 - sim).clamp(0,2) / 2.0
        return c
