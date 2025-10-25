import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class BertWhitening:
    def __init__(self, model_name='bert-base-uncased', whiten_dim=256, device='cpu'):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.device = device
        self.whiten_dim = whiten_dim
        self.kernel = None
        self.bias = None

    def fit_whitening(self, sentences):
        # Very small whiten implementation: collect CLS embeddings and compute mean; no PCA for toy.
        embs = []
        for i in range(0, len(sentences), 8):
            batch = sentences[i:i+8]
            tok = self.tok(batch, padding=True, truncation=True, max_length=160, return_tensors='pt')
            out = self.bert(**tok)
            cls = out.last_hidden_state[:,0].detach().cpu().numpy()
            embs.append(cls)
        X = np.vstack(embs)
        mu = X.mean(0)
        W = np.eye(X.shape[1])[:self.whiten_dim,:]
        self.kernel = torch.tensor(W, dtype=torch.float32)
        self.bias = torch.tensor(-mu[:self.whiten_dim], dtype=torch.float32)

    def encode(self, sentences):
        tok = self.tok(sentences, padding=True, truncation=True, max_length=160, return_tensors='pt')
        out = self.bert(**tok)
        cls = out.last_hidden_state[:,0]
        if self.kernel is None:
            # fallback: return CLS projected to first whiten_dim dims
            return cls[:,:self.whiten_dim]
        k = self.kernel.to(cls.device)
        b = self.bias.to(cls.device)
        e = torch.matmul(cls, k.T) + b
        e = torch.nn.functional.normalize(e, p=2, dim=-1)
        return e
