import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

class PretrainedTextEncoder(nn.Module):
    """
    Uses a Transformer encoder (e.g., 'bert-base-uncased') and supports whitening fit.
    """
    def __init__(self, model_name='bert-base-uncased', whiten_dim=256, device='cpu'):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.whiten_dim = whiten_dim
        self.kernel = None
        self.bias = None

    @torch.no_grad()
    def encode(self, texts, batch_size=16):
        self.model.eval()
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=160, return_tensors='pt')
            enc = {k:v.to(self.device) for k,v in enc.items()}
            out = self.model(**enc)
            cls = out.last_hidden_state[:,0]  # CLS
            if self.kernel is not None:
                k = self.kernel.to(self.device)
                b = self.bias.to(self.device)
                v = (cls @ k.T) + b
            else:
                v = cls[:, :self.whiten_dim]
            v = torch.nn.functional.normalize(v, p=2, dim=-1)
            embs.append(v.cpu())
        return torch.cat(embs, dim=0)

    def fit_whitening(self, sentences, batch_size=32):
        """
        Compute a whitening projection using SVD on CLS embeddings.
        Save self.kernel (whiten_dim x cls_dim) and self.bias.
        """
        self.model.eval()
        embs = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=160, return_tensors='pt')
            enc = {k:v.to(self.device) for k,v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
                cls = out.last_hidden_state[:,0].cpu().numpy()
                embs.append(cls)
        X = np.vstack(embs)
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
        cov = np.cov(Xc, rowvar=False)
        U, S, Vt = np.linalg.svd(cov)
        W = (U / np.sqrt(S + 1e-12)).T  # whiten matrix (cls_dim x cls_dim)
        W_cut = W[:self.whiten_dim, :]
        self.kernel = torch.tensor(W_cut, dtype=torch.float32)
        self.bias = torch.tensor((-mu @ W_cut.T).squeeze(), dtype=torch.float32)
