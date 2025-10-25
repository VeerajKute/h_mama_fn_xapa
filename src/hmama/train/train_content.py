import torch
from torch.utils.data import DataLoader
from hmama.data.dataset import NewsDataset
from hmama.models.text_encoder import BertWhitening
from hmama.models.vision_encoder import ResNetEncoder
from hmama.models.ug_mop import UGMOP
from hmama.config import ModelConfig, TrainConfig
from tqdm import tqdm

def collate(batch):
    return {k:[b[k] for b in batch] for k in batch[0].keys()}

def main():
    print('Starting content-only training (toy)...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mc = ModelConfig(device=device)
    tc = TrainConfig()
    ds = NewsDataset('data/raw/merged_dataset.jsonl')
    dl = DataLoader(ds, batch_size=tc.batch_size, shuffle=True, collate_fn=collate)
    bertw = BertWhitening(mc.text_model_name, mc.whiten_dim, mc.device)
    clip = ResNetEncoder(device=mc.device)
    # fit whitening on small dataset
    texts = [x['text'] for x in ds]
    ocrs = [x['ocr'] for x in ds]
    bertw.fit_whitening(texts + ocrs)
    gate = UGMOP(mc.whiten_dim, mc.whiten_dim, 512)
    opt = torch.optim.AdamW(gate.parameters(), lr=tc.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(tc.epochs):
        pbar = tqdm(dl)
        for batch in pbar:
            imgs = batch['image']
            text = batch['text']
            ocrt = batch['ocr']
            y = torch.tensor(batch['label'], dtype=torch.long)
            zt = bertw.encode(text)
            zo = bertw.encode(ocrt)
            zi = clip.encode_image(imgs)
            contr = clip.contradiction(text, imgs)
            logits, w = gate(zt, zo, zi, contr)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss))
    print('Done toy training.')

if __name__ == '__main__':
    main()
