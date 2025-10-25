import math
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import numpy as np
from PIL import Image
import cv2

class TextAIGCSentinel:
    def __init__(self, model_name='gpt2', device='cpu', low_thresh=12.0, high_thresh=200.0):
        self.device = device
        self.tok = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.low = low_thresh
        self.high = high_thresh

    def perplexity(self, text: str):
        enc = self.tok(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            out = self.model(**enc, labels=enc['input_ids'])
            ppl = math.exp(out.loss.item())
        return ppl

    def suspicious(self, text: str):
        p = self.perplexity(text)
        # heuristic: very low or very high perplexity suspicious
        return float(p < self.low or p > self.high), float(p)

class ImageArtifactSentinel:
    def __init__(self):
        pass

    def high_freq_energy(self, pil_image: Image.Image):
        # compute approximate high-frequency energy via Laplacian
        im = np.array(pil_image.convert('L'), dtype=np.float32)
        lap = cv2.Laplacian(im, cv2.CV_32F)
        e = np.mean(np.abs(lap))
        return float(e)

    def suspicious(self, pil_image: Image.Image, threshold=10.0):
        e = self.high_freq_energy(pil_image)
        return float(e > threshold), e
