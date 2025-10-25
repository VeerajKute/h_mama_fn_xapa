import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch.nn as nn

class CLIPEncoder:
    def __init__(self, model_name='openai/clip-vit-base-patch32', device='cpu', fine_tune=False):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.proc = CLIPProcessor.from_pretrained(model_name)
        # CLIP text max length is typically 77 tokens
        try:
            self.max_text_len = int(getattr(self.proc.tokenizer, 'model_max_length', 77))
        except Exception:
            self.max_text_len = 77
        self.fine_tune = fine_tune
        if fine_tune:
            # small projection head for text (to allow fine-tuning downstream)
            txt_dim = self.model.text_model.config.hidden_size
            self.text_proj = nn.Linear(txt_dim, txt_dim).to(device)
            # initialize identity-ish
            nn.init.eye_(self.text_proj.weight)
            nn.init.zeros_(self.text_proj.bias)
        else:
            self.text_proj = None

    @torch.no_grad()
    def encode_image(self, pil_images):
        inputs = self.proc(images=pil_images, return_tensors='pt', padding=True).to(self.device)
        feats = self.model.get_image_features(**inputs)
        return torch.nn.functional.normalize(feats, p=2, dim=-1)

    def encode_text(self, texts):
        inputs = self.proc(text=texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_text_len).to(self.device)
        feats = self.model.get_text_features(**inputs)
        if self.fine_tune and self.text_proj is not None:
            feats = self.text_proj(feats)
        return torch.nn.functional.normalize(feats, p=2, dim=-1)

    @torch.no_grad()
    def similarity(self, texts, pil_images):
        t = self.encode_text(texts)
        i = self.encode_image(pil_images)
        sims = (t * i).sum(dim=1)
        return sims

    @torch.no_grad()
    def contradiction_score(self, texts, pil_images):
        sim = self.similarity(texts, pil_images)
        c = (1 - sim).clamp(0,2) / 2.0
        return c
