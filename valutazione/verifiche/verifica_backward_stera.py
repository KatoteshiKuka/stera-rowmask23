"""Verifica del passo in avanti e della retropropagazione di una proiezione STERA (Capitolo 4, Sezione 4.7).

Confronta, in doppia precisione, le formule della tesi con:
  (a) l'autodifferenziazione di PyTorch applicata alla stessa ricostruzione usata in addestramento
      (addestramento/ts68_core.py, RowMask23.reconstruct, e la forma W = diag(gout) W~ diag(gin) + U V^T di ts68_v3.py);
  (b) le differenze finite centrate su coordinate scelte a caso.
Verifica anche che il prodotto con il blocco e con il blocco trasposto calcolati con la trasformata di
Walsh--Hadamard coincidano con i prodotti densi, e i numeri dell'esempio della tesi.
Uso: python verifica_backward_stera.py   (scrive risultati_backward.json accanto allo script)
"""
import json, pathlib, sys
import torch

R = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "addestramento"))
import ts68_core as T

torch.manual_seed(0)
D = torch.float64
N, NC = 128, 3176
RM = T.RowMask23("cpu")
sig = lambda i, k: 1 + (13 * k + 7 * i) % 127


def phi(c):                       # blocco 128x128 da un vettore di 3176 coordinate (stesso indice del trainer)
    return (c[RM.base] + c[RM.residual] * RM.mask[0].to(c.dtype)).reshape(N, N)


def H(v):                         # trasformata di Walsh--Hadamard veloce, senza normalizzazione
    v = v.clone(); h = 1
    while h < N:
        w = v.reshape(N // (2 * h), 2, h, *v.shape[1:])
        a, b = w[:, 0].clone(), w[:, 1].clone()
        w[:, 0], w[:, 1] = a + b, a - b
        v = w.reshape(N, *v.shape[1:]); h *= 2
    return v


def parti(c):
    d = c[:128]; b = torch.cat([torch.zeros(1, dtype=c.dtype), c[128:255]]); e = c[255:].reshape(127, 23)
    return d, b, e


def blocco_per(c, x):             # y = Phi(c) x con la WHT (Sezione 4.3)
    d, b, e = parti(c)
    y = d * x + H(H(b) * H(x)) / N
    for i in range(1, N):
        for k in range(23):
            y[i] += e[i - 1, k] * x[i ^ sig(i, k)]
    return y


def blocco_T_per(c, v):           # y = Phi(c)^T v: D e B sono simmetriche, le correzioni cambiano verso
    d, b, e = parti(c)
    y = d * v + H(H(b) * H(v)) / N
    for i in range(1, N):
        for k in range(23):
            y[i ^ sig(i, k)] += e[i - 1, k] * v[i]
    return y


def grad_coordinate(alfa, dt, xt):  # dt = delta~ del blocco (128), xt = x~ del blocco (128), somma sui token
    g = torch.zeros(NC, dtype=D)
    g[:128] = alfa * (dt * xt).sum(-1)
    corr = sum(H(H(dt[:, t]) * H(xt[:, t])) / N for t in range(dt.shape[1]))   # correlazione XOR
    g[128:255] = alfa * corr[1:]
    for i in range(1, N):
        for k in range(23):
            g[255 + 23 * (i - 1) + k] = alfa * (dt[i] * xt[i ^ sig(i, k)]).sum()
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


def proiezione(coords, scales, gout, gin, U, V):  # come ts68_v3.py: ricostruzione + guadagni + basso rango
    Wt = RM.reconstruct(coords, scales, m, n)
    return Wt * gout.reshape(m, 1) * gin.reshape(1, n) + U @ V.t()


W = proiezione(coords, scales, gout, gin, U, V)
Y = X @ W.t()
(Y * Delta).sum().backward()      # L lineare in Y con dL/dY = Delta
auto = {k: v.grad.clone() for k, v in dict(coords=coords, scales=scales, gout=gout, gin=gin, U=U, V=V, x=X).items()}

# --- formule della tesi, passo per passo (Sezione 4.7) ---
with torch.no_grad():
    x, dl = X.t(), Delta.t()                     # colonne = token
    xt = gin[:, None] * x                        # passo 1 del forward
    z = torch.zeros(m, Tk, dtype=D)              # passo 2: mosaico
    for p in range(P):
        for q in range(Q):
            for t in range(Tk):
                z[p*N:(p+1)*N, t] += scales[p*Q+q, 0, 0] * blocco_per(coords[p*Q+q], xt[q*N:(q+1)*N, t])
    u = gout[:, None] * z                        # passo 3
    a = V.t() @ x                                # passo 4: basso rango
    y = u + U @ a
    mio = {}
    mio["U"] = dl @ a.t(); mio["V"] = x @ (U.t() @ dl).t()          # ramo del basso rango
    mio["gout"] = (dl * z).sum(1); dt = gout[:, None] * dl          # guadagni di riga
    gc, gs, wtd = torch.zeros(P*Q, NC, dtype=D), torch.zeros(P*Q, 1, 1, dtype=D), torch.zeros(n, Tk, dtype=D)
    for p in range(P):
        for q in range(Q):
            k = p*Q+q; al = scales[k, 0, 0]; dtb, xtb = dt[p*N:(p+1)*N], xt[q*N:(q+1)*N]
            gc[k] = grad_coordinate(al, dtb, xtb)
            gs[k] = sum(dtb[:, t] @ blocco_per(coords[k], xtb[:, t]) for t in range(Tk))
            for t in range(Tk):
                wtd[q*N:(q+1)*N, t] += al * blocco_T_per(coords[k], dtb[:, t])
    mio["coords"], mio["scales"] = gc, gs
    mio["gin"] = (x * wtd).sum(1)
    mio["x"] = (gin[:, None] * wtd + V @ (U.t() @ dl)).t()

    err = {k: float((mio[k] - auto[k]).abs().max() / auto[k].abs().max()) for k in auto}
    err_forward = float((y.t() - Y).abs().max() / Y.abs().max())

# --- differenze finite su 20 parametri a caso ---
def L_of(**kw):
    a = dict(coords=coords, scales=scales, gout=gout, gin=gin, U=U, V=V); a.update(kw)
    return float(((X @ proiezione(**a).t()) * Delta).sum())
fd = []
g = torch.Generator().manual_seed(1)
with torch.no_grad():
    for name in ["coords", "coords", "coords", "coords", "scales", "gout", "gin", "U", "V"] * 2 + ["coords", "coords"]:
        t = dict(coords=coords, scales=scales, gout=gout, gin=gin, U=U, V=V)[name]
        idx = tuple(int(torch.randint(s, (1,), generator=g)) for s in t.shape); h = 1e-6
        tp, tm = t.detach().clone(), t.detach().clone(); tp[idx] += h; tm[idx] -= h
        num = (L_of(**{name: tp}) - L_of(**{name: tm})) / (2 * h)
        fd.append(abs(num - float(mio[name][idx])) / max(1e-12, abs(num)))

# --- esempio numerico della tesi (elemento (1,9), rango 1) ---
al, go1, gi9, phi19, x9, d1, u11, v91 = 1.2, .8, 1.1, .15, 2.0, .5, .3, -.4
es = dict(y1=go1*al*phi19*gi9*x9 + u11*v91*x9, e=al*go1*d1*gi9*x9, alfa=go1*d1*phi19*gi9*x9,
          gout=d1*al*phi19*gi9*x9, gin=x9*al*phi19*go1*d1, U=d1*v91*x9, V=x9*u11*d1,
          x=gi9*al*phi19*go1*d1 + v91*u11*d1)

ris = dict(forward_errore_relativo=err_forward, backward_errore_relativo_vs_autograd=err,
           differenze_finite_errore_relativo_max=max(fd), esempio_tesi={k: round(v, 6) for k, v in es.items()},
           dimensioni=dict(m=m, n=n, r=r, token=Tk), precisione="float64")
(pathlib.Path(__file__).parent / "risultati_backward.json").write_text(json.dumps(ris, indent=1))
print(json.dumps(ris, indent=1))
