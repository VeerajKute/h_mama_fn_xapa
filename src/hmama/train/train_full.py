import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from hmama.data.datasets import LocalMultimodalDataset, LocalMultimodalDatasetWithImages, reddit_collate_fn
from hmama.config_small import ModelConfigSmall
from hmama.models.text_encoder_full import PretrainedTextEncoder
from hmama.models.clip_encoder_ft import CLIPEncoder
from hmama.models.ug_mop_adaptive import UGMOP_Adaptive
from hmama.evaluate import compute_basic_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='data/raw/samples.jsonl')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--lite', action='store_true', help='Use lite mode (DistilBERT + CLIP ViT-B/16)')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--eval_data', type=str, default=None, help='Optional path to eval JSONL; runs evaluation after training if provided')
    p.add_argument('--num_workers', type=int, default=0)
    return p.parse_args()


def collate(batch):
    return {k: [b[k] for b in batch] for k in batch[0].keys()}


def _is_multi_image_jsonl(jsonl_path: str) -> bool:
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                # determine by presence of list-valued 'image'
                if 'image' in obj and isinstance(obj['image'], list):
                    return True
                if 'image_path' in obj:
                    return False
                # if ambiguous, continue to next line
        return False
    except Exception:
        return False


def _encode_images_mixed(clip_enc: CLIPEncoder, images_batch, device: str):
    """
    Accepts either:
      - list of PIL.Image (single-image dataset collate)
      - list of list[PIL.Image] (multi-image dataset collate)
    Returns a tensor of shape (batch, dim) by averaging multiple image embeddings per sample.
    """
    # Detect if first element is list (multi-image)
    if len(images_batch) > 0 and isinstance(images_batch[0], list):
        embs = []
        for imgs in images_batch:
            if len(imgs) == 0:
                # fallback zero vector if no images
                embs.append(torch.zeros(1, clip_enc.model.visual_projection.out_features if hasattr(clip_enc.model, 'visual_projection') else 512, device=device))
                continue
            zi_each = clip_enc.encode_image(imgs).to(device)  # (n_i, dim)
            zi = zi_each.mean(dim=0, keepdim=True)  # (1, dim)
            embs.append(zi)
        return torch.cat(embs, dim=0)
    else:
        # Single-image per sample
        return clip_enc.encode_image(images_batch).to(device)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device
    cfg = ModelConfigSmall(device=device) if args.lite else ModelConfigSmall(device=device)

    # Data
    is_multi = _is_multi_image_jsonl(args.data)
    if is_multi:
        ds = LocalMultimodalDatasetWithImages(args.data)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=reddit_collate_fn, num_workers=args.num_workers)
    else:
        ds = LocalMultimodalDataset(args.data)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=args.num_workers)

    # Encoders
    text_enc = PretrainedTextEncoder(model_name=cfg.text_model_name, whiten_dim=cfg.whiten_dim, device=device)
    clip_enc = CLIPEncoder(model_name=cfg.clip_model_name, device=device)

    # UGMOP (produces logits for fake/real)
    ug = UGMOP_Adaptive(in_text=cfg.whiten_dim, in_ocr=cfg.whiten_dim, in_img=512, device=device).to(device)
    opt = torch.optim.AdamW(ug.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Optional: fit whitening on the dataset corpus
    texts = [x['text'] for x in ds]
    ocrs = [x['ocr'] for x in ds]
    try:
        text_enc.fit_whitening(texts + ocrs)
    except Exception:
        pass

    for epoch in range(args.epochs):
        ug.train()
        total_loss = 0.0
        pbar = tqdm(dl, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch in pbar:
            imgs = batch['image']  # may be list[Image] or list[list[Image]]
            texts_b = batch['text']
            ocr_b = batch['ocr']
            y = torch.tensor(batch['label'], dtype=torch.long, device=device)

            # Encode modalities
            zt = text_enc.encode(texts_b).to(device)
            zo = text_enc.encode(ocr_b).to(device)
            zi = _encode_images_mixed(clip_enc, imgs, device)
            contr = clip_enc.contradiction_score(texts_b, imgs).to(device)

            # Forward
            logits, gate_w = ug(zt, zo, zi, contr)
            loss = loss_fn(logits, y)

            # Backprop
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss)
            pbar.set_postfix(loss=float(loss))

        avg_loss = total_loss / max(1, len(dl))
        print(f'Avg loss: {avg_loss:.4f}')

        # Save checkpoint
        save_path = os.path.join(args.save_dir, 'model_full.pth')
        torch.save({'ug': ug.state_dict()}, save_path)
        print(f'Saved checkpoint to {save_path}')

    # Optional evaluation
    if args.eval_data is not None and os.path.exists(args.eval_data):
        ug.eval()
        is_multi_eval = _is_multi_image_jsonl(args.eval_data)
        if is_multi_eval:
            ds_eval = LocalMultimodalDatasetWithImages(args.eval_data)
            dl_eval = DataLoader(ds_eval, batch_size=args.batch, shuffle=False, collate_fn=reddit_collate_fn, num_workers=args.num_workers)
        else:
            ds_eval = LocalMultimodalDataset(args.eval_data)
            dl_eval = DataLoader(ds_eval, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

        y_true = []
        y_pred = []
        y_prob = []
        with torch.no_grad():
            for batch in tqdm(dl_eval, desc='Evaluating'):
                imgs = batch['image']
                texts_b = batch['text']
                ocr_b = batch['ocr']
                y = torch.tensor(batch['label'], dtype=torch.long, device=device)

                zt = text_enc.encode(texts_b).to(device)
                zo = text_enc.encode(ocr_b).to(device)
                zi = _encode_images_mixed(clip_enc, imgs, device)
                contr = clip_enc.contradiction_score(texts_b, imgs).to(device)
                logits, _ = ug(zt, zo, zi, contr)
                probs = torch.softmax(logits, dim=-1)
                pred = probs.argmax(dim=-1)

                y_true.extend(y.detach().cpu().tolist())
                y_pred.extend(pred.detach().cpu().tolist())
                y_prob.extend(probs[:,1].detach().cpu().tolist())  # prob of class 1 (REAL)

        metrics = compute_basic_metrics(y_true, y_pred, y_prob)
        print('Evaluation metrics:', metrics)
        try:
            with open(os.path.join(args.save_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
        except Exception:
            pass


if __name__ == '__main__':
    main()
