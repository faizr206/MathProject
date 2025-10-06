#!/usr/bin/env python3
import os
import argparse
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score
from scipy.linalg import subspace_angles


def select_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"start_seconds", "end_seconds", "duration", "sr", "class_idx", "fold", "clip_index"}
    feat_cols = [c for c in num_cols if c not in exclude and df[c].nunique(dropna=True) > 5]
    X = df[feat_cols].to_numpy().astype(float)
    return X, feat_cols


def center_and_pca(X: np.ndarray, var_cutoff: float = 0.95, max_d: int = 150) -> Tuple[np.ndarray, int]:
    Xc = X - X.mean(axis=0, keepdims=True)
    p = PCA(svd_solver="full").fit(Xc)
    evr = p.explained_variance_ratio_
    d = int(np.searchsorted(np.cumsum(evr), var_cutoff) + 1)
    d = int(min(d, max_d, Xc.shape[1], Xc.shape[0]))
    return p.transform(Xc)[:, :d], d


def fit_bases_affine(X: np.ndarray, y: np.ndarray, d: int):
    bases = {}
    for c in np.unique(y):
        Xc = X[y == c]
        mu = Xc.mean(axis=0)
        Xc0 = Xc - mu
        p = PCA(svd_solver="full").fit(Xc0)
        de = int(min(d, Xc0.shape[0], Xc0.shape[1]))
        bases[str(c)] = {"U": p.components_[:de].T, "mu": mu, "d": de}
    return bases


def residual_to_subspace_affine(x, U, mu):
    r = x - mu
    if U.size == 0:
        return float(np.linalg.norm(r))
    return float(np.linalg.norm(r - U @ (U.T @ r)))


def predict_with_rule_affine(X, bases, rule):
    d = list(bases.values())[0]["d"] if bases else 1
    preds = []
    for x in X:
        best_c, best_s = None, -np.inf
        for c, B in bases.items():
            if rule == "residual_normed":
                s = -residual_to_subspace_affine(x, B["U"], B["mu"]) / np.sqrt(d)
            else:
                s = np.linalg.norm(B["U"].T @ (x - B["mu"])) / d
            if s > best_s:
                best_s, best_c = s, c
        preds.append(best_c)
    return np.array(preds)


def cv_select(X: np.ndarray, y: np.ndarray, dims: List[int], rules: List[str]):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    results = []
    for d in dims:
        for rule in rules:
            ys, yh = [], []
            for tr, te in skf.split(X, y):
                bases = fit_bases_affine(X[tr], y[tr], d)
                yh.append(predict_with_rule_affine(X[te], bases, rule))
                ys.append(y[te])
            ys = np.concatenate(ys); yh = np.concatenate(yh)
            results.append({"d": d, "rule": rule, "macro_f1": f1_score(ys, yh, average="macro"), "acc": accuracy_score(ys, yh)})
    return pd.DataFrame(results).sort_values(["macro_f1", "acc", "d"], ascending=[False, False, True]).iloc[0].to_dict()


def run_for_attr(df: pd.DataFrame, outdir: str, attr: str, var_cutoff: float, max_d: int, trim_frac: float, candidate_dims: List[int]):
    if attr not in df.columns:
        print(f"[warn] attribute '{attr}' not found; skipping")
        return
    X_all, feat_cols = select_feature_matrix(df)
    y_all = df[attr].astype(str).values
    mask = (y_all != "none") & pd.notna(y_all)
    X_all, y_all = X_all[mask], y_all[mask]
    # keep classes with >= 20
    vc = pd.Series(y_all).value_counts()
    keep_classes = vc[vc >= 20].index
    keep = np.isin(y_all, keep_classes)
    X_all, y_all = X_all[keep], y_all[keep]
    if len(np.unique(y_all)) < 2:
        print(f"[warn] attribute '{attr}': <2 classes after filtering; skipping")
        return

    Z, d_glob = center_and_pca(X_all, var_cutoff=var_cutoff, max_d=max_d)
    # Trim worst 10% per class using temporary d
    # (simple residual-only trim to avoid storing extra arrays)
    out_mask = np.zeros(len(y_all), dtype=bool)
    for c in np.unique(y_all):
        Xc = Z[y_all == c]
        mu = Xc.mean(axis=0)
        Xc0 = Xc - mu
        p = PCA(svd_solver="full").fit(Xc0)
        de = int(min( min(20, d_glob), Xc0.shape[0], Xc0.shape[1]))
        U = p.components_[:de].T
        res = np.array([np.linalg.norm((x - mu) - U @ (U.T @ (x - mu))) for x in Xc])
        k = int(np.floor(trim_frac * len(res)))
        if k > 0:
            worst = np.argsort(res)[-k:]
            out_mask[np.where(y_all == c)[0][worst]] = True
    keep = ~out_mask
    Zt, yt = Z[keep], y_all[keep]

    dims = [d for d in candidate_dims if d <= d_glob]
    best = cv_select(Zt, yt, dims if dims else [min(10, d_glob)], ["residual_normed", "proj_energy"])    
    shared_d = int(min(d_glob, max(1, int(best.get("d", min(10, d_glob))))))
    rule = str(best.get("rule", "residual_normed"))

    bases = fit_bases_affine(Zt, yt, shared_d)
    uniq = sorted(pd.unique(yt))
    # angles
    angle_mat = pd.DataFrame(index=uniq, columns=uniq, dtype=float)
    for c1 in uniq:
        for c2 in uniq:
            angle_mat.loc[c1, c2] = float(subspace_angles(bases[c1]["U"], bases[c2]["U"])[0])
    # stability (lite: 20 bootstraps)
    from numpy.linalg import svd
    def sub_aff(U, V):
        s = svd(U.T @ V, compute_uv=False)
        s = np.clip(s, 0, 1)
        return float(np.mean(s**2))
    stab = {c: [] for c in uniq}
    rng = np.random.default_rng(0)
    for c in uniq:
        Xc = Zt[yt == c]
        if len(Xc) < 5:
            continue
        pref = PCA(svd_solver="full").fit(Xc - Xc.mean(axis=0))
        Uref = pref.components_[:shared_d].T
        m = len(Xc)
        for _ in range(20):
            k = max(2, int(round(0.7 * m)))
            idx = rng.choice(m, size=k, replace=True)
            Xb = Xc[idx]
            pb = PCA(svd_solver="full").fit(Xb - Xb.mean(axis=0))
            Ub = pb.components_[:shared_d].T
            stab[c].append(sub_aff(Uref, Ub))

    # evaluation
    yh = predict_with_rule_affine(Z, bases, rule)
    report = classification_report(y_all, yh, digits=3, output_dict=True)

    # write
    od = os.path.join(outdir, attr)
    os.makedirs(od, exist_ok=True)
    angle_mat.to_csv(os.path.join(od, "angles.csv"))
    pd.DataFrame(stab).to_csv(os.path.join(od, "stability.csv"), index=False)
    with open(os.path.join(od, "summary.txt"), "w") as f:
        f.write(f"== Global PCA ==\nKept d_glob={d_glob}\n\n")
        f.write(f"== CV selection ==\nSHARED_D={shared_d}, RULE={rule}\n\n")
        off = [angle_mat.loc[a,b] for a in uniq for b in uniq if a!=b]
        if off:
            f.write(f"== Between-class angles (rad) ==\nmin_offdiag={float(np.min(off)):.6f}, median_offdiag={float(np.median(off)):.6f}\n\n")
        f.write("== Final evaluation on full set ==\n")
        f.write(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Run subspace analysis across multiple metadata attributes")
    ap.add_argument("--outdir", default="analysis_outputs3")
    ap.add_argument("--attrs", nargs="*", default=["engine_configuration", "engine_state", "turbo_supercharged", "is_car"])    
    ap.add_argument("--var-cutoff", type=float, default=0.95)
    ap.add_argument("--global-max-d", type=int, default=150)
    ap.add_argument("--trim-frac", type=float, default=0.10)
    ap.add_argument("--candidate-dims", nargs="*", type=int, default=[5,10,15,20,30,40,60,80,100])
    args = ap.parse_args()

    feats_path = os.path.join(args.outdir, "clip_features.parquet")
    df = pd.read_parquet(feats_path)

    attr_out = os.path.join("analysis_extras", "attr_outputs")
    os.makedirs(attr_out, exist_ok=True)
    for attr in args.attrs:
        run_for_attr(df, os.path.join(attr_out), attr, args.var_cutoff, args.global_max_d, args.trim_frac, args.candidate_dims)
        print(f"Done: {attr}")


if __name__ == "__main__":
    main()

