import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.sparse import load_npz
from tqdm.auto import tqdm
from torch.utils.checkpoint import checkpoint
from models import DySAT



def normalize_gcn(adj):
    """D^-1/2 (A+I) D^-1/2, computed per edge so nnz stays fixed."""
    A = sp.coo_matrix(adj, dtype=np.float32) + sp.eye(adj.shape[0], dtype=np.float32)
    A = A.tocoo()
    d = np.asarray(A.sum(1)).flatten()
    dinv = np.zeros_like(d)
    nz = d > 0
    dinv[nz] = 1.0 / np.sqrt(d[nz])
    data = A.data * dinv[A.row] * dinv[A.col]
    return sp.coo_matrix((data, (A.row, A.col)), shape=A.shape)


def sample_context(raw, anchors, max_positive, rng):
    """Positive partners per anchor, as minibatch.py samples from context_pairs."""
    a, p = [], []
    for t, (indptr, indices) in enumerate(raw):
        s, d = [], []
        for n in anchors:
            nb = indices[indptr[n]:indptr[n + 1]]
            if nb.size == 0:
                continue
            k = min(max_positive, nb.size)
            pick = nb if nb.size <= k else rng.choice(nb, k, replace=False)
            s.extend([n] * len(pick)); d.extend(pick.tolist())
        a.append(np.asarray(s)); p.append(np.asarray(d))
    return a, p


def graph_loss(e_t, src, dst, n_nodes, neg_sample_size=10, neg_weight=1.0):
    """_loss port. Positives are observed edges; walk context is too large to store."""
    pos = (e_t[src] * e_t[dst]).sum(-1)
    neg_dst = torch.randint(n_nodes, (src.numel() * neg_sample_size,), device=e_t.device)
    neg = (e_t[src.repeat_interleave(neg_sample_size)] * e_t[neg_dst]).sum(-1)
    return (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) +
            neg_weight * F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))


def train_dysat(source, data_dir, out_dir,
                num_features=128, structural_layer_config=(128,),
                structural_head_config=(16,), temporal_layer_config=128,
                temporal_head_config=16, spatial_drop=0.1, temporal_drop=0.5,
                learning_rate=0.001, weight_decay=5e-4, max_gradient_norm=1.0,
                neg_sample_size=10, neg_weight=1.0, max_positive=10,
                epochs=150, batch_size=256, patience=10, seed=42, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    d = Path(data_dir) / source
    years = np.load(d / "years.npy")
    T = len(years)

    raw, graphs = [], []
    for y in years:
        W = load_npz(d / f"adj_{y}.npz").tocsr()
        N = W.shape[0]
        raw.append((W.indptr.copy(), W.indices.copy()))
        A = normalize_gcn(W)
        graphs.append((torch.as_tensor(A.row, dtype=torch.long),
                       torch.as_tensor(A.col, dtype=torch.long)))
        del W, A
        
    print(f"# train nodes {N} | batches per epoch {int(np.ceil(N / batch_size))}", flush=True)

    model = DySAT(N, T, num_features, structural_layer_config,
                  structural_head_config, temporal_layer_config,
                  temporal_head_config, spatial_drop, temporal_drop).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best, best_epoch, wait, hist = float("inf"), 0, 0, []

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N).numpy()
        tot, nb, t0 = 0.0, 0, time.time()
        pbar = tqdm(range(0, N, batch_size), desc=f"{source} ep{ep}")
        for s in pbar:
            anchors = perm[s:s + batch_size]
            a_list, p_list = sample_context(raw, anchors, max_positive, rng)

            # Temporal attention is per-node, so it only needs the nodes the loss
            # touches. Partners are unioned in: restricting to batch-internal
            # edges would drop all but ~0.1% of the pairs.
            union = np.unique(np.concatenate([anchors] + a_list + p_list))
            pos_map = np.full(N, -1, dtype=np.int64)
            pos_map[union] = np.arange(len(union))
            ui = torch.as_tensor(union, dtype=torch.long, device=device)
            U = len(union)

            opt.zero_grad()
            zs = []
            for src, dst in graphs:
                z = checkpoint(model.structural_one,
                               src.to(device), dst.to(device), N,
                               use_reentrant=False)
                zs.append(z[ui]); del z
            e = model.temporal(torch.stack(zs, 1))            # [U, T, F]
            del zs

            loss, n_terms = 0.0, 0
            for t in range(T):
                if a_list[t].size == 0:
                    continue
                loss = loss + graph_loss(
                    e[:, t, :],
                    torch.as_tensor(pos_map[a_list[t]], dtype=torch.long, device=device),
                    torch.as_tensor(pos_map[p_list[t]], dtype=torch.long, device=device),
                    U, neg_sample_size, neg_weight)
                n_terms += 1

            if n_terms:
                (loss / n_terms).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
                opt.step()
                tot += float(loss.detach()) / n_terms
                nb += 1
            del e
            if device == "cuda":
                torch.cuda.empty_cache()
            pbar.set_postfix(loss=f"{tot / max(nb, 1):.4f}")

        ep_loss = tot / max(nb, 1)
        hist.append(ep_loss)
        print(f"Time for epoch {time.time() - t0:.1f} | "
              f"Mean Loss at epoch {ep} : {ep_loss:.5f}", flush=True)

        if ep_loss < best - 1e-4:
            best, best_epoch, wait = ep_loss, ep, 0
            torch.save(model.state_dict(), out_dir / f"{source}_best.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping at epoch", ep, flush=True)
                break

    print(f"Best epoch {best_epoch} loss {best}", flush=True)

    model.load_state_dict(torch.load(out_dir / f"{source}_best.pt"))
    model.eval()
    with torch.no_grad():
        Z = np.empty((N, T, temporal_layer_config), dtype=np.float32)
        for s in tqdm(range(0, N, batch_size), desc=f"{source} embed"):
            bi = torch.arange(s, min(s + batch_size, N), device=device)
            zs = [model.structural_one(src.to(device), dst.to(device), N)[bi]
                  for src, dst in graphs]
            Z[s:s + len(bi)] = model.temporal(torch.stack(zs, 1)).cpu().numpy()
            del zs

    np.save(out_dir / f"{source}_embs_all.npy", Z)
    pd.Series(hist, name="loss").to_csv(out_dir / f"{source}_loss.csv")
    print("Saved embeddings", Z.shape, flush=True)
    return Z