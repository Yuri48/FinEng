"""
Rigobon & Sack (2003) heteroskedasticity-based estimator of the war-risk factor.
  x_j,t = d_j1 * z1_t + (other common factors) + idiosyncratic ;  d_11 = 1 (x1 = two-year Treasury yield change)
Identification: Var(z1) is higher on 'war news' days (H) than on nearby other days (L); all other factors homoskedastic
and orthogonal to z1. Then  dOmega = Omega_H - Omega_L = dVar(z1) * [1, d21; d21, d21^2].
Three IV estimators (paper eq. 8-12):
  nu1 = +dx1 on H, -dx1 on L  ->  d = dCov(x1,x2)/dVar(x1)      (eq. 7 / 10)
  nu2 = +dx2 on H, -dx2 on L  ->  d = dVar(x2)/dCov(x1,x2)       (eq. 6 / 12)
  nu3 = [nu1, nu2]             ->  2SLS with both instruments (the paper's preferred column)
"""
import compat  # noqa: F401  (scipy PROPACK stub, see compat.py)
import numpy as np, pandas as pd

def _iv(y, x, Z):
    Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z
    x = x.reshape(-1, 1)
    Pz = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T
    xh = Pz @ x
    b = float((xh.T @ y) / (xh.T @ x))
    u = y - x.ravel() * b
    v = float(np.sum((xh.ravel() ** 2) * (u ** 2)) / (float(xh.T @ x) ** 2))   # heteroskedasticity-robust
    n = len(y); v *= n / (n - 1)
    return b, np.sqrt(v)

def rs_estimate(ch, var, base, H, L, scale=-0.25):
    H = [d for d in H if d in ch.index]; L = [d for d in L if d in ch.index]
    sub = ch.loc[H + L, [base, var]].dropna()
    isH = sub.index.isin(H)
    x1 = sub[base].values.astype(float).copy(); x2 = sub[var].values.astype(float).copy()
    for m in (True, False):                      # demean within regime (finite-sample analogue of zero-mean shocks)
        x1[isH == m] -= x1[isH == m].mean(); x2[isH == m] -= x2[isH == m].mean()
    sgn = np.where(isH, 1.0, -1.0); nu1 = sgn * x1; nu2 = sgn * x2
    out = {}
    for name, Z in (("nu1", nu1), ("nu2", nu2), ("nu3", np.column_stack([nu1, nu2]))):
        b, se = _iv(x2, x1, Z); out[name] = scale * b; out[name + "_t"] = abs(b / se) if se > 0 else np.nan
    SH = np.cov(np.vstack([x1[isH], x2[isH]]), bias=True); SL = np.cov(np.vstack([x1[~isH], x2[~isH]]), bias=True); dO = SH - SL
    out["dVar_x1"] = dO[0, 0]; out["dCov"] = dO[0, 1]; out["dVar_x2"] = dO[1, 1]; out["nH"] = int(isH.sum()); out["nL"] = int((~isH).sum())
    d = out["nu3"] / scale
    varL = float(np.mean(sub[var].values[~isH] ** 2)); varH = float(np.mean(sub[var].values[isH] ** 2))
    dvar1 = float(np.mean(sub[base].values[isH] ** 2)) - float(np.mean(sub[base].values[~isH] ** 2))
    pred = d ** 2 * dvar1
    out.update(varL=varL, varH=varH, pred_dvar=pred, pct_H=100 * pred / varH if (varH > 0 and dvar1 > 0) else np.nan)
    return out

def table3_all_days(ch, var, base, H, L, d, nH_all, nL_all):
    """Share of the variance of the cumulative change over the window attributable to war news on the H days
    (serially independent daily changes: total = nH*VarH + nL*VarL; war part = nH * d^2 * dVar(x1))."""
    sub = ch[[base, var]].dropna(); inH = sub.index.isin(H)
    varH = float(np.mean(sub.loc[inH, var] ** 2)); varL = float(np.mean(sub.loc[~inH, var] ** 2))
    dvar1 = float(np.mean(sub.loc[inH, base] ** 2)) - float(np.mean(sub.loc[~inH, base] ** 2))
    tot = nH_all * varH + nL_all * varL
    return 100 * nH_all * d ** 2 * dvar1 / tot if (tot > 0 and dvar1 > 0) else np.nan

def pick_L_days(bdays, H, exclude=()):
    """Nearest non-H business day to each H day (paper: 'as close as possible to, but not included in' the H list)."""
    bd = list(bdays); pos = {d: i for i, d in enumerate(bd)}; used = set(H) | set(exclude); L = []
    for h in H:
        i = pos[h]; k = 1
        while k <= len(bd):
            c = [j for j in (i - k, i + k) if 0 <= j < len(bd) and bd[j] not in used]
            if c: j = min(c, key=lambda j: abs(j - i)); L.append(bd[j]); used.add(bd[j]); break
            k += 1
    return L
