#!/usr/bin/env python3
import os
import math
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score
from scipy.linalg import subspace_angles


def select_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Heuristically select numeric feature columns from a mixed metadata+feature DataFrame.
    Returns X (n x D) and the column list.
    """
    # Start with numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude common meta columns if present
    exclude = {"start_seconds", "end_seconds", "duration", "sr", "class_idx", "fold", "clip_index"}
    feat_cols = [c for c in num_cols if c not in exclude]
    # Also drop near-binary columns which are likely codes (<=5 unique values)
    feat_cols = [c for c in feat_cols if df[c].nunique(dropna=True) > 5]
    X = df[feat_cols].to_numpy().astype(float)
    return X, feat_cols


def center_and_pca(X: np.ndarray, var_cutoff: float = 0.95, max_d: int = 150) -> Tuple[np.ndarray, PCA, int]:
    Xc = X - X.mean(axis=0, keepdims=True)
    p = PCA(svd_solver="full").fit(Xc)
    evr = p.explained_variance_ratio_
    cum = np.cumsum(evr)
    d_glob = int(np.searchsorted(cum, var_cutoff) + 1)
    d_glob = int(min(d_glob, max_d, Xc.shape[1], Xc.shape[0]))
    Z = p.transform(Xc)[:, :d_glob]
    return Z, p, d_glob


def fit_bases_affine(X: np.ndarray, y: np.ndarray, d: int) -> Dict[str, Dict[str, np.ndarray]]:
    bases = {}
    for c in np.unique(y):
        Xc = X[y == c]
        mu = Xc.mean(axis=0)
        Xc0 = Xc - mu
        p = PCA(svd_solver="full").fit(Xc0)
        d_eff = int(min(d, Xc0.shape[0], Xc0.shape[1]))
        U = p.components_[:d_eff].T
        bases[str(c)] = {"U": U, "mu": mu, "d": d_eff}
    return bases


def residual_to_subspace_affine(x: np.ndarray, U: np.ndarray, mu: np.ndarray) -> float:
    r = x - mu
    if U.size == 0:
        return float(np.linalg.norm(r))
    proj = U @ (U.T @ r)
    res = r - proj
    return float(np.linalg.norm(res))


def projection_energy_affine(x: np.ndarray, U: np.ndarray, mu: np.ndarray) -> float:
    r = x - mu
    if U.size == 0:
        return 0.0
    return float(np.linalg.norm(U.T @ r))


def predict_with_rule_affine(X: np.ndarray, bases: Dict[str, Dict[str, np.ndarray]], rule: str) -> np.ndarray:
    d = list(bases.values())[0]["d"] if bases else 1
    preds = []
    for x in X:
        best_c, best_score = None, -np.inf
        for c, B in bases.items():
            if rule == "residual_normed":
                s = -residual_to_subspace_affine(x, B["U"], B["mu"]) / math.sqrt(d)
            elif rule == "proj_energy":
                s = projection_energy_affine(x, B["U"], B["mu"]) / d
            else:
                raise ValueError(f"unknown rule: {rule}")
            if s > best_score:
                best_score, best_c = s, c
        preds.append(best_c)
    return np.array(preds)


def bootstrap_stability(X: np.ndarray, y: np.ndarray, d: int, n_boot: int = 40, frac: float = 0.7,
                        rng: np.random.Generator = np.random.default_rng(0)) -> Dict[str, List[float]]:
    from numpy.linalg import svd

    def subspace_affinity(U: np.ndarray, V: np.ndarray) -> float:
        # sq of cos principal angles, averaged
        s = svd(U.T @ V, compute_uv=False)
        s = np.clip(s, 0.0, 1.0)
        return float(np.mean(s**2))

    classes = np.unique(y)
    stab = {str(c): [] for c in classes}
    for c in classes:
        Xc = X[y == c]
        if len(Xc) < 5:
            continue
        # reference basis
        p_ref = PCA(svd_solver="full").fit(Xc - Xc.mean(axis=0))
        U_ref = p_ref.components_[:d].T
        m = len(Xc)
        for _ in range(n_boot):
            k = max(2, int(round(frac * m)))
            idx = rng.choice(m, size=k, replace=True)
            Xb = Xc[idx]
            p_b = PCA(svd_solver="full").fit(Xb - Xb.mean(axis=0))
            U_b = p_b.components_[:d].T
            stab[str(c)].append(subspace_affinity(U_ref, U_b))
    return stab


def trim_outliers_per_class_affine(X: np.ndarray, y: np.ndarray, trim_frac: float = 0.10, d_temp: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    if trim_frac <= 0:
        return X, y
    outlier_mask = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        Xc = X[y == c]
        mu = Xc.mean(axis=0)
        Xc0 = Xc - mu
        p = PCA(svd_solver="full").fit(Xc0)
        d_eff = int(min(d_temp, Xc0.shape[0], Xc0.shape[1]))
        U = p.components_[:d_eff].T
        res = np.array([residual_to_subspace_affine(x, U, mu) for x in Xc])
        k = int(np.floor(trim_frac * len(res)))
        if k <= 0:
            continue
        worst = np.argsort(res)[-k:]
        outlier_mask[np.where(y == c)[0][worst]] = True
    keep = ~outlier_mask
    return X[keep], y[keep]


def cv_select(X: np.ndarray, y: np.ndarray, dims: List[int], rules: List[str]) -> Dict[str, float]:
    dims = sorted(set(int(d) for d in dims))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    results = []
    for d in dims:
        for rule in rules:
            ys, yh = [], []
            for tr, te in skf.split(X, y):
                bases = fit_bases_affine(X[tr], y[tr], d)
                preds = predict_with_rule_affine(X[te], bases, rule)
                ys.append(y[te]); yh.append(preds)
            ys = np.concatenate(ys); yh = np.concatenate(yh)
            results.append({"d": d, "rule": rule,
                            "macro_f1": f1_score(ys, yh, average="macro"),
                            "acc": accuracy_score(ys, yh)})
    res_df = pd.DataFrame(results).sort_values(["macro_f1", "acc", "d"], ascending=[False, False, True])
    return res_df.iloc[0].to_dict()


def main():
    ap = argparse.ArgumentParser(description="Posthoc numeric summaries for subspace analysis")
    ap.add_argument("--outdir", default="analysis_outputs3")
    ap.add_argument("--label-col", default="turbo_supercharged",
                    help="Label column in features parquet (e.g., turbo_supercharged, engine_configuration, engine_state, is_car)")
    ap.add_argument("--trim-frac", type=float, default=0.10)
    ap.add_argument("--var-cutoff", type=float, default=0.95)
    ap.add_argument("--global-max-d", type=int, default=150)
    ap.add_argument("--candidate-dims", nargs="*", type=int, default=[5, 10, 15, 20, 30, 40, 60, 80, 100])
    ap.add_argument("--n-boot", type=int, default=40)
    ap.add_argument("--boot-frac", type=float, default=0.7)
    args = ap.parse_args()

    feats_path = os.path.join(args.outdir, "clip_features.parquet")
    if not os.path.exists(feats_path):
        raise FileNotFoundError(f"Could not find features parquet at {feats_path}")

    df = pd.read_parquet(feats_path)
    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found in {feats_path}")

    # Prepare X and y
    X_all, feat_cols = select_feature_matrix(df)
    y_all = df[args.label_col].astype(str).values
    # Filter invalid/none labels
    mask = (y_all != "none") & pd.notna(y_all)
    X_all = X_all[mask]
    y_all = y_all[mask]

    # Keep classes with minimum support
    counts = pd.Series(y_all).value_counts()
    classes = counts[counts >= 10].index.tolist()
    keep = np.isin(y_all, classes)
    X_all, y_all = X_all[keep], y_all[keep]

    # Global PCA
    Z, p_glob, d_glob = center_and_pca(X_all, var_cutoff=args.var_cutoff, max_d=args.global_max_d)

    # Trim per class by affine residuals
    Z_trim, y_trim = trim_outliers_per_class_affine(Z, y_all, trim_frac=args.trim_frac, d_temp=min(20, d_glob))

    # CV selection
    cand = [d for d in args.candidate_dims if d <= d_glob]
    best = cv_select(Z_trim, y_trim, cand, ["residual_normed", "proj_energy"]) if len(cand) > 0 else {"d": min(10, d_glob), "rule": "residual_normed", "macro_f1": float("nan"), "acc": float("nan")}
    shared_d = int(min(d_glob, max(1, int(best.get("d", min(10, d_glob))))))
    rule = str(best.get("rule", "residual_normed"))

    # Fit final bases and compute angles
    bases = fit_bases_affine(Z_trim, y_trim, shared_d)
    uniq = sorted(pd.unique(y_trim))
    angle_mat = pd.DataFrame(index=uniq, columns=uniq, dtype=float)
    for c1 in uniq:
        for c2 in uniq:
            th = subspace_angles(bases[c1]["U"], bases[c2]["U"])[0]
            angle_mat.loc[c1, c2] = float(th)

    # Stability bootstraps
    stab = bootstrap_stability(Z_trim, y_trim, shared_d, n_boot=args.n_boot, frac=args.boot_frac)

    # Final evaluation on full set
    y_pred = predict_with_rule_affine(Z, bases, rule)
    report = classification_report(y_all, y_pred, digits=3, output_dict=True)

    # Numeric summaries
    offdiag = [angle_mat.loc[c1, c2] for c1 in uniq for c2 in uniq if c1 != c2]
    min_angle = float(np.min(offdiag)) if offdiag else None
    med_angle = float(np.median(offdiag)) if offdiag else None
    stab_summary = {c: {"mean": float(np.mean(v)) if len(v) else None, "std": float(np.std(v)) if len(v) else None} for c, v in stab.items()}

    out_txt = os.path.join(args.outdir, f"posthoc_summaries_{args.label_col}.txt")
    with open(out_txt, "w") as f:
        f.write(f"== Global PCA ==\nKept d_glob={d_glob}\n\n")
        f.write(f"== CV selection ==\nSHARED_D={shared_d}, RULE={rule}\nBest (cv): macroF1={best.get('macro_f1', float('nan'))}, acc={best.get('acc', float('nan'))}\n\n")
        f.write("== Stability (bootstrap; subspace affinity) ==\n")
        for c, st in stab_summary.items():
            if st["mean"] is not None:
                f.write(f"{c}: mean={st['mean']:.3f}, std={st['std']:.3f}\n")
            else:
                f.write(f"{c}: insufficient data\n")
        f.write("\n== Between-class angles (radians) ==\n")
        if min_angle is not None:
            f.write(f"min_offdiag={min_angle:.6f}, median_offdiag={med_angle:.6f}\n\n")
        else:
            f.write("no angles computed\n\n")
        f.write("== Final evaluation on full set ==\n")
        f.write(json.dumps(report, indent=2))

    angle_mat.to_csv(os.path.join(args.outdir, f"angles_{args.label_col}.csv"))
    pd.DataFrame(stab).to_csv(os.path.join(args.outdir, f"stability_{args.label_col}.csv"), index=False)

    print(f"Wrote: {out_txt}")


if __name__ == "__main__":
    main()

