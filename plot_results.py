import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICY_LABELS = {
    "random": "Random",
    "fastest": "Fastest-only",
    "static": "Static SPD-EQ",
    "dspdeq": "D-SPD-EQ",
    "oracle": "Coverage Oracle",
}


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    mean = values.mean()
    if n < 2:
        return mean, 0.0
    ci = 1.96 * values.std(ddof=1) / np.sqrt(n)
    return mean, ci


def interp_series(g, col, grid):
    g = g.sort_values("wall")
    wall = g["wall"].values
    y = g[col].values
    return np.interp(grid, wall, y, left=y[0], right=y[-1])


def row_at_cut(g, tcut):
    g = g.sort_values("wall")
    g_cut = g[g["wall"] <= tcut]
    if len(g_cut) == 0:
        return g.iloc[0]
    return g_cut.iloc[-1]


def jain_from_counts(counts_json):
    counts = np.asarray(json.loads(counts_json), dtype=np.float64)
    denom = len(counts) * np.sum(counts ** 2) + 1e-12
    return float((counts.sum() ** 2) / denom)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/raw_results.csv")
    parser.add_argument("--tcut", type=float, default=80.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--ps", type=float, default=0.5)
    parser.add_argument("--out_prefix", type=str, default="results")
    parser.add_argument("--include_oracle", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print("Policies:", sorted(df["policy"].unique()))
    print("Seeds per policy:", df.groupby("policy")["seed"].nunique().to_dict())

    policy_order = ["random", "fastest", "static", "dspdeq"]
    if args.include_oracle and "oracle" in df["policy"].unique():
        policy_order.append("oracle")

    df_main = df[(df["alpha"] == args.alpha) & (df["ps"] == args.ps)].copy()

    # 当前 CSV 没有 wp_start 列，只有 wp_end；D-SPD-EQ 只保留 wp_end=0.8
    if "wp_start" in df_main.columns:
        df_main = df_main[
            (df_main["policy"] != "dspdeq") | (
                (df_main["wp_start"] == 0.2) & (df_main["wp_end"] == 0.8)
            )
        ]
    else:
        df_main = df_main[
            (df_main["policy"] != "dspdeq") | (df_main["wp_end"] == 0.8)
        ]

    grid = np.linspace(0.0, args.tcut, 200)

    # ---------------- Figure 1: accuracy vs wall-clock ----------------
    plt.figure(figsize=(4.0, 2.8))
    for policy in policy_order:
        curves = []
        sub = df_main[df_main["policy"] == policy]
        for seed, g in sub.groupby("seed"):
            if len(g) == 0:
                continue
            curves.append(interp_series(g, "acc", grid))
        if len(curves) == 0:
            continue
        curves = np.asarray(curves)
        mean = curves.mean(axis=0)
        ci = (1.96 * curves.std(axis=0, ddof=1) / np.sqrt(curves.shape[0])
              if curves.shape[0] >= 2 else np.zeros_like(mean))
        plt.plot(grid, mean, label=POLICY_LABELS[policy])
        plt.fill_between(grid, mean - ci, mean + ci, alpha=0.18)

    plt.xlabel("Simulated wall-clock time (s)")
    plt.ylabel("Global test accuracy")
    plt.xlim(0, args.tcut)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}/fig1_acc_vs_wall.png", dpi=300)
    plt.close()

    # ---------------- Figure 2: worst-class at cutoff ----------------
    worst_rows = []
    for policy in policy_order:
        sub = df_main[df_main["policy"] == policy]
        for seed, g in sub.groupby("seed"):
            if len(g) == 0:
                continue
            r = row_at_cut(g, args.tcut)
            worst_rows.append({"policy": policy, "seed": seed, "worst": float(r["worst"])})
    df_worst = pd.DataFrame(worst_rows)

    plt.figure(figsize=(4.0, 2.8))
    means, cis = [], []
    for policy in policy_order:
        m, c = mean_ci(df_worst.loc[df_worst["policy"] == policy, "worst"].values)
        means.append(m)
        cis.append(c)
    x = np.arange(len(policy_order))
    plt.bar(x, means, yerr=cis, capsize=3)
    plt.xticks(x, [POLICY_LABELS[p] for p in policy_order], rotation=20, ha="right", fontsize=7)
    plt.ylabel(f"Worst-class accuracy at {args.tcut:.0f}s")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}/fig2_worst_class_at_cut.png", dpi=300)
    plt.close()

    # ---------------- Figure 3: Jain fairness at cutoff ----------------
    jain_rows = []
    for policy in policy_order:
        sub = df_main[df_main["policy"] == policy]
        for seed, g in sub.groupby("seed"):
            if len(g) == 0:
                continue
            r = row_at_cut(g, args.tcut)
            jain_rows.append({"policy": policy, "seed": seed,
                              "jain": jain_from_counts(r["counts"])})
    df_jain = pd.DataFrame(jain_rows)

    plt.figure(figsize=(4.0, 2.8))
    means, cis = [], []
    for policy in policy_order:
        m, c = mean_ci(df_jain.loc[df_jain["policy"] == policy, "jain"].values)
        means.append(m)
        cis.append(c)
    x = np.arange(len(policy_order))
    plt.bar(x, means, yerr=cis, capsize=3)
    plt.xticks(x, [POLICY_LABELS[p] for p in policy_order], rotation=20, ha="right", fontsize=7)
    plt.ylabel(f"Jain fairness at {args.tcut:.0f}s")
    plt.ylim(0, 1.0)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}/fig3_jain_at_cut.png", dpi=300)
    plt.close()

    # ---------------- Summary CSV ----------------
    summary = []
    for policy in policy_order:
        w_vals = df_worst.loc[df_worst["policy"] == policy, "worst"].values
        j_vals = df_jain.loc[df_jain["policy"] == policy, "jain"].values
        wm, wc = mean_ci(w_vals)
        jm, jc = mean_ci(j_vals)
        summary.append({
            "policy": policy,
            "worst_mean": wm, "worst_ci": wc,
            "jain_mean": jm, "jain_ci": jc,
            "n_seeds": len(w_vals),
        })
    pd.DataFrame(summary).to_csv(f"{args.out_prefix}/summary_at_cut.csv", index=False)

    # ---------------- Seed-level diagnostics ----------------
    if args.diagnose:
        print("\n" + "=" * 90)
        print(f"Seed-level diagnostics at T_cut = {args.tcut}s")
        print("=" * 90)
        print(f"{'policy':10s} {'seed':>4s} {'worst':>7s} {'jain':>7s} {'acc':>7s} {'wall':>8s}")
        diag_rows = []
        for policy in policy_order:
            sub = df_main[df_main["policy"] == policy]
            for seed, g in sub.groupby("seed"):
                r = row_at_cut(g, args.tcut)
                worst_val = float(r["worst"])
                jain_val = jain_from_counts(r["counts"])
                acc_val = float(r["acc"])
                wall_val = float(r["wall"])
                print(f"{policy:10s} {int(seed):4d} {worst_val:7.4f} {jain_val:7.4f} "
                      f"{acc_val:7.4f} {wall_val:8.2f}")
                diag_rows.append({
                    "policy": policy, "seed": int(seed),
                    "worst_at_cut": worst_val,
                    "jain_at_cut": jain_val,
                    "acc_at_cut": acc_val,
                    "wall_at_cut": wall_val,
                })
        pd.DataFrame(diag_rows).to_csv(f"{args.out_prefix}/seed_diagnostics.csv", index=False)
        print(f"\nSaved seed diagnostics to {args.out_prefix}/seed_diagnostics.csv")

    print(f"\nSaved figures and summary to {args.out_prefix}/")


if __name__ == "__main__":
    main()