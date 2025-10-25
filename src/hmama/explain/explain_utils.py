import torch
from captum.attr import IntegratedGradients
from transformers import AutoTokenizer
import numpy as np

def text_integrated_gradients(model, tokenizer, text, label_index=1, device='cpu'):
    """
    model: should accept input_ids and attention_mask and return logits
    tokenizer: corresponding tokenizer
    Returns: list of token, attribution scores
    """
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=160)
    input_ids = enc['input_ids'].to(device)
    att = enc['attention_mask'].to(device)
    model = model.to(device)
    def predict(input_ids):
        outputs = model(input_ids=input_ids, attention_mask=att)
        # map to logits (batch, classes)
        return outputs[0]
    ig = IntegratedGradients(predict)
    attributions, delta = ig.attribute(inputs=input_ids, target=label_index, return_convergence_delta=True)
    at = attributions.sum(dim=-1).squeeze(0)  # sum embedding dims
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    scores = at.detach().cpu().numpy()
    # normalize
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    return list(zip(tokens, scores.tolist()))

# Image saliency via gradients for torchvision models (simple)
def image_gradient_saliency(model, input_tensor, target_class=None):
    # input_tensor: (1,C,H,W), requires_grad True
    model.eval()
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    out = model(input_tensor)
    if target_class is None:
        target_class = out.argmax(dim=1).item()
    score = out[0, target_class]
    score.backward()
    saliency = input_tensor.grad.abs().sum(dim=1).squeeze(0)  # HxW
    sal = saliency.detach().cpu().numpy()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-12)
    return sal
