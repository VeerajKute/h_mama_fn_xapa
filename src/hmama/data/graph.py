import json
from collections import defaultdict
import torch
from torch_geometric.data import HeteroData

def load_cascades(jsonl_path):
    events = []
    with open(jsonl_path) as f:
        for line in f:
            events.append(json.loads(line))
    return events

def build_hetero_graph(events):
    uid, pid, sid = {}, {}, {}
    def get(d, x):
        if x not in d:
            d[x] = len(d)
        return d[x]
    edges_user_post = []
    for e in events:
        u = get(uid, e['user_id'])
        p = get(pid, e['post_id'])
        s = get(sid, e['source'])
        edges_user_post.append((u,p,e['time']))
    data = HeteroData()
    data['user'].num_nodes = len(uid)
    data['post'].num_nodes = len(pid)
    data['source'].num_nodes = len(sid)
    # simple random features
    data['user'].x = torch.randn(len(uid), 32)
    data['post'].x = torch.randn(len(pid), 32)
    data['source'].x = torch.randn(len(sid), 32)
    if edges_user_post:
        up_src = torch.tensor([u for u,_,_ in edges_user_post], dtype=torch.long)
        up_dst = torch.tensor([p for _,p,_ in edges_user_post], dtype=torch.long)
        data['user','interacts','post'].edge_index = torch.stack([up_src, up_dst])
    return data, uid, pid, sid
