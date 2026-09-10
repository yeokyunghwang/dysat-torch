import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralAttentionLayer(nn.Module):
    """sp_attn_head port. conv1d(kernel 1) == Linear."""
    def __init__(self, in_dim, out_dim, n_heads, attn_drop, ffd_drop):
        super().__init__()
        self.h, self.c = n_heads, out_dim // n_heads
        self.W = nn.ModuleList([nn.Linear(in_dim, self.c, bias=False) for _ in range(n_heads)])
        self.a1 = nn.ModuleList([nn.Linear(self.c, 1) for _ in range(n_heads)])
        self.a2 = nn.ModuleList([nn.Linear(self.c, 1) for _ in range(n_heads)])
        self.attn_drop, self.ffd_drop = attn_drop, ffd_drop
        for m in list(self.W) + list(self.a1) + list(self.a2):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, src, dst,  w=None):
        outs = []
        for j in range(self.h):
            seq = self.W[j](x)
            f1, f2 = self.a1[j](seq).view(-1), self.a2[j](seq).view(-1)

            e = f1[src] + f2[dst]
            if w is not None:
                e = e * w
            e = F.leaky_relu(e, 0.2)
            e = e - e.max()
            ex = e.exp()
            denom = torch.zeros(n, device=x.device).index_add_(0, dst, ex)
            coef = ex / (denom[dst] + 1e-16)

            if self.attn_drop:
                coef = F.dropout(coef, self.attn_drop, self.training)
            if self.ffd_drop:
                seq = F.dropout(seq, self.ffd_drop, self.training)

            z = torch.zeros(n, self.c, device=x.device)
            z.index_add_(0, dst, seq[src] * coef.unsqueeze(-1))
            outs.append(F.elu(z))
        return torch.cat(outs, dim=-1)


class TemporalAttentionLayer(nn.Module):
    """Scaling by sqrt(T), matching the TF code (the paper uses sqrt(F'))."""
    def __init__(self, dim, n_heads, T, attn_drop, position_ffn=True, residual=False):
        super().__init__()
        self.h, self.T, self.d = n_heads, T, dim
        self.attn_drop = attn_drop
        self.position_ffn, self.residual = position_ffn, residual
        self.pos = nn.Parameter(torch.empty(T, dim))
        self.Wq = nn.Parameter(torch.empty(dim, dim))
        self.Wk = nn.Parameter(torch.empty(dim, dim))
        self.Wv = nn.Parameter(torch.empty(dim, dim))
        self.ffn = nn.Linear(dim, dim)
        for p in (self.pos, self.Wq, self.Wk, self.Wv):
            nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.ffn.weight)
        nn.init.zeros_(self.ffn.bias)

    def forward(self, x):                                  # [B, T, F]
        b = x.shape[0]
        ti = x + self.pos[None, :, :]
        q, k, v = ti @ self.Wq, ti @ self.Wk, ti @ self.Wv
        s = self.d // self.h
        q, k, v = (torch.cat(torch.split(z, s, dim=2), dim=0) for z in (q, k, v))

        o = (q @ k.transpose(1, 2)) / (self.T ** 0.5)
        mask = torch.tril(torch.ones(self.T, self.T, dtype=torch.bool, device=x.device))
        o = F.softmax(o.masked_fill(~mask, -2.0 ** 32 + 1), dim=2)
        o = F.dropout(o, self.attn_drop, self.training)
        o = torch.cat(torch.split(o @ v, b, dim=0), dim=2)

        if self.position_ffn:
            o = F.relu(self.ffn(o)) + o
        if self.residual:
            o = o + ti
        return o
