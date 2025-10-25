import torch
from torch.utils.data import DataLoader
from hmama.data.dataset import NewsDataset
from hmama.data.graph import load_cascades, build_hetero_graph
from hmama.models.text_encoder import BertWhitening
from hmama.models.vision_encoder import ResNetEncoder
from hmama.models.ug_mop import UGMOP
from hmama.models.propagation_gnn import THPGNN
from tqdm import tqdm

def collate(batch):
    return {k:[b[k] for b in batch] for k in batch[0].keys()}

class JointClassifier(torch.nn.Module):
    def __init__(self, gate, gnn_out=64):
        super().__init__()
        self.gate = gate
        self.fc = torch.nn.Sequential(torch.nn.Linear(2+gnn_out, 2))

    def forward(self, fused_logits, gnn_h):
        x = torch.cat([fused_logits, gnn_h], dim=1)
        return self.fc(x)

def main():
    print('Starting joint training (toy)...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = NewsDataset('data/raw/samples.jsonl')
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate)
    bertw = BertWhitening(device=device)
    clip = ResNetEncoder(device=device)
    texts = [x['text'] for x in ds]
    ocrs = [x['ocr'] for x in ds]
    bertw.fit_whitening(texts + ocrs)
    gate = UGMOP(bertw.whiten_dim, bertw.whiten_dim, 512)
    gdata, uid, pid, sid = build_hetero_graph(load_cascades('data/raw/propagation.jsonl'))
    gnn = THPGNN()
    clf = JointClassifier(gate, gnn_out=64)
    opt = torch.optim.AdamW(list(gate.parameters())+list(gnn.parameters())+list(clf.parameters()), lr=2e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    # compute gnn embeddings once (toy)
    gnn_h = gnn(gdata)
    post_map = {v:k for k,v in pid.items()}
    for epoch in range(1):
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
            fused_logits, w = gate(zt, zo, zi, contr)
            # map batch ids to post indices (toy: assume order)
            idx = torch.arange(0, fused_logits.size(0))
            h_sel = gnn_h[idx]
            pred = clf(fused_logits, h_sel)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss))
    print('Done joint training (toy).')

if __name__ == '__main__':
    main()
