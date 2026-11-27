"""Check of the forward pass and of the backpropagation of a STERA projection (thesis, Chapter 5).

Compares, in double precision, the formulas of the thesis with:
  (a) PyTorch autograd applied to the same reconstruction used in training
      (training/ts68_core.py, RowMask23.reconstruct, and the form W = diag(gout) W~ diag(gin) + U V^T of ts68_v3.py);
  (b) central finite differences on randomly chosen parameters.
It also checks that the products with the block and with the transposed block computed with the
Walsh--Hadamard transform match the dense products, and the numbers of the worked example in the thesis.
Usage: python check_backward_stera.py   (writes backward_results.json next to the script)
"""
import json, pathlib, sys
import torch

R = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "training"))
import ts68_core as T

torch.manual_seed(0)
D = torch.float64
N, NC = 128, 3176
RM = T.RowMask23("cpu")
sig = lambda i, k: 1 + (13 * k + 7 * i) % 127


def phi(c):                       # 128x128 block from a vector of 3176 coordinates (same indexing as the trainer)
    return (c[RM.base] + c[RM.residual] * RM.mask[0].to(c.dtype)).reshape(N, N)


def H(v):                         # fast Walsh--Hadamard transform, unnormalized
    v = v.clone(); h = 1
    while h < N:
        w = v.reshape(N // (2 * h), 2, h, *v.shape[1:])
        a, b = w[:, 0].clone(), w[:, 1].clone()
        w[:, 0], w[:, 1] = a + b, a - b
        v = w.reshape(N, *v.shape[1:]); h *= 2
    return v


def parts(c):
    d = c[:128]; b = torch.cat([torch.zeros(1, dtype=c.dtype), c[128:255]]); e = c[255:].reshape(127, 23)
    return d, b, e


def block_times(c, x):            # y = Phi(c) x with the WHT
    d, b, e = parts(c)
    y = d * x + H(H(b) * H(x)) / N
    for i in range(1, N):
        for k in range(23):
            y[i] += e[i - 1, k] * x[i ^ sig(i, k)]
    return y


def block_T_times(c, v):          # y = Phi(c)^T v: D and B are symmetric, the corrections swap direction
    d, b, e = parts(c)
    y = d * v + H(H(b) * H(v)) / N
    for i in range(1, N):
        for k in range(23):
            y[i ^ sig(i, k)] += e[i - 1, k] * v[i]
    return y


def coordinate_grad(alpha, dt, xt):  # dt = delta~ of the block (128), xt = x~ of the block (128), summed over tokens
    g = torch.zeros(NC, dtype=D)
    g[:128] = alpha * (dt * xt).sum(-1)
    corr = sum(H(H(dt[:, t]) * H(xt[:, t])) / N for t in range(dt.shape[1]))   # XOR correlation
    g[128:255] = alpha * corr[1:]
    for i in range(1, N):
        for k in range(23):
            g[255 + 23 * (i - 1) + k] = alpha * (dt[i] * xt[i ^ sig(i, k)]).sum()
    return g


m, n, r, Tk = 256, 384, 8, 5
P, Q = m // N, n // N
coords = (torch.randn(P * Q, NC, dtype=D) * .3).requires_grad_()
scales = (1 + .3 * torch.randn(P * Q, 1, 1, dtype=D)).requires_grad_()
gout = (1 + .3 * torch.randn(m, dtype=D)).requires_grad_()
gin = (1 + .3 * torch.randn(n, dtype=D)).requires_grad_()
U = (.2 * torch.randn(m, r, dtype=D)).requires_grad_()
V = (.2 * torch.randn(n, r, dtype=D)).requires_grad_()
X = torch.randn(Tk, n, dtype=D, requires_grad=True)
Delta = torch.randn(Tk, m, dtype=D)


def projection(coords, scales, gout, gin, U, V):  # as in ts68_v3.py: reconstruction + gains + low rank
    Wt = RM.reconstruct(coords, scales, m, n)
    return Wt * gout.reshape(m, 1) * gin.reshape(1, n) + U @ V.t()


W = projection(coords, scales, gout, gin, U, V)
Y = X @ W.t()
(Y * Delta).sum().backward()      # L linear in Y with dL/dY = Delta
auto = {k: v.grad.clone() for k, v in dict(coords=coords, scales=scales, gout=gout, gin=gin, U=U, V=V, x=X).items()}

# --- formulas of the thesis, step by step (Chapter 5) ---
with torch.no_grad():
    x, dl = X.t(), Delta.t()                     # columns = tokens
    xt = gin[:, None] * x                        # forward step 1
    z = torch.zeros(m, Tk, dtype=D)              # step 2: block mosaic
    for p in range(P):
        for q in range(Q):
            for t in range(Tk):
                z[p*N:(p+1)*N, t] += scales[p*Q+q, 0, 0] * block_times(coords[p*Q+q], xt[q*N:(q+1)*N, t])
    u = gout[:, None] * z                        # step 3
    a = V.t() @ x                                # step 4: low rank
    y = u + U @ a
    ours = {}
    ours["U"] = dl @ a.t(); ours["V"] = x @ (U.t() @ dl).t()          # low-rank branch
    ours["gout"] = (dl * z).sum(1); dt = gout[:, None] * dl          # row gains
    gc, gs, wtd = torch.zeros(P*Q, NC, dtype=D), torch.zeros(P*Q, 1, 1, dtype=D), torch.zeros(n, Tk, dtype=D)
    for p in range(P):
        for q in range(Q):
            k = p*Q+q; al = scales[k, 0, 0]; dtb, xtb = dt[p*N:(p+1)*N], xt[q*N:(q+1)*N]
            gc[k] = coordinate_grad(al, dtb, xtb)
            gs[k] = sum(dtb[:, t] @ block_times(coords[k], xtb[:, t]) for t in range(Tk))
            for t in range(Tk):
                wtd[q*N:(q+1)*N, t] += al * block_T_times(coords[k], dtb[:, t])
    ours["coords"], ours["scales"] = gc, gs
    ours["gin"] = (x * wtd).sum(1)
    ours["x"] = (gin[:, None] * wtd + V @ (U.t() @ dl)).t()

    err = {k: float((ours[k] - auto[k]).abs().max() / auto[k].abs().max()) for k in auto}
    err_forward = float((y.t() - Y).abs().max() / Y.abs().max())

# --- finite differences on 20 random parameters ---
def L_of(**kw):
    a = dict(coords=coords, scales=scales, gout=gout, gin=gin, U=U, V=V); a.update(kw)
    return float(((X @ projection(**a).t()) * Delta).sum())
fd = []
g = torch.Generator().manual_seed(1)
with torch.no_grad():
    for name in ["coords", "coords", "coords", "coords", "scales", "gout", "gin", "U", "V"] * 2 + ["coords", "coords"]:
        t = dict(coords=coords, scales=scales, gout=gout, gin=gin, U=U, V=V)[name]
        idx = tuple(int(torch.randint(s, (1,), generator=g)) for s in t.shape); h = 1e-6
        tp, tm = t.detach().clone(), t.detach().clone(); tp[idx] += h; tm[idx] -= h
        num = (L_of(**{name: tp}) - L_of(**{name: tm})) / (2 * h)
        fd.append(abs(num - float(ours[name][idx])) / max(1e-12, abs(num)))

# --- worked example of the thesis (entry (1,9), rank 1) ---
al, go1, gi9, phi19, x9, d1, u11, v91 = 1.2, .8, 1.1, .15, 2.0, .5, .3, -.4
es = dict(y1=go1*al*phi19*gi9*x9 + u11*v91*x9, e=al*go1*d1*gi9*x9, alpha=go1*d1*phi19*gi9*x9,
          gout=d1*al*phi19*gi9*x9, gin=x9*al*phi19*go1*d1, U=d1*v91*x9, V=x9*u11*d1,
          x=gi9*al*phi19*go1*d1 + v91*u11*d1)

res = dict(forward_relative_error=err_forward, backward_relative_error_vs_autograd=err,
           finite_differences_max_relative_error=max(fd), thesis_example={k: round(v, 6) for k, v in es.items()},
           sizes=dict(m=m, n=n, r=r, tokens=Tk), precision="float64")
(pathlib.Path(__file__).parent / "backward_results.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
