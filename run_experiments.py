import argparse
import copy
import csv
import itertools
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from tqdm import tqdm


# Model and basic utilities
class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def jain_index(counts):
    counts = np.asarray(counts, dtype=np.float64)
    denom = len(counts) * np.sum(counts ** 2) + 1e-12
    return float((counts.sum() ** 2) / denom)


# Data partition and link model
def dirichlet_partition(targets, num_clients, alpha, seed):
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    classes = np.unique(targets)
    client_indices = [[] for _ in range(num_clients)]

    for c in classes:
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)
        props = rng.dirichlet([alpha] * num_clients)
        counts = rng.multinomial(len(idx_c), props)
        start = 0
        for i, cnt in enumerate(counts):
            client_indices[i].extend(idx_c[start:start + cnt].tolist())
            start += cnt

    for i in range(num_clients):
        rng.shuffle(client_indices[i])
    return client_indices


def assign_rates(num_clients, slow_fraction, seed):
    rng = np.random.default_rng(seed + 10007)
    slow = int(round(slow_fraction * num_clients))
    remaining = num_clients - slow
    fast = remaining // 2
    medium = remaining - fast
    rates = [20e6] * fast + [5e6] * medium + [1e6] * slow
    rng.shuffle(rates)
    return np.asarray(rates, dtype=np.float64)


# Local training / aggregation / evaluation
def local_train(global_model, client_x, client_y, epochs, batch_size, lr, device):
    if len(client_x) == 0:
        return {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

    model = copy.deepcopy(global_model).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_samples = len(client_x)
    for _ in range(epochs):
        indices = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            x, y = client_x[batch_idx].to(device), client_y[batch_idx].to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def aggregate(global_model, local_states, weights):
    total = float(sum(weights)) + 1e-12
    new_state = copy.deepcopy(global_model.state_dict())
    for k in new_state:
        acc = torch.zeros_like(new_state[k], dtype=torch.float32)
        for state, w in zip(local_states, weights):
            acc += state[k].float() * (w / total)
        new_state[k] = acc
    global_model.load_state_dict(new_state)


def evaluate(model, test_x, test_y, device, batch_size=2048, num_classes=10):
    model.eval()
    correct = np.zeros(num_classes, dtype=np.float64)
    total = np.zeros(num_classes, dtype=np.float64)

    with torch.no_grad():
        for i in range(0, len(test_x), batch_size):
            x = test_x[i:i + batch_size].to(device)
            y = test_y[i:i + batch_size].to(device)
            pred = model(x).argmax(dim=1)
            for c in range(num_classes):
                mask = (y == c)
                total[c] += mask.sum().item()
                correct[c] += (pred[mask] == y[mask]).sum().item()

    per_class = correct / np.maximum(total, 1.0)
    acc = correct.sum() / max(total.sum(), 1.0)
    worst = float(np.min(per_class))
    return float(acc), per_class, worst


# Schedulers
def score_based_select(n, rates, t, H, K, q, wp_start, wp_end, tau, rng):
    N = len(rates)
    speed = rates / rates.max()
    debt = np.maximum(0.0, tau - n)
    debt_norm = debt / (debt.max() + 1e-9)
    wp = wp_start + (wp_end - wp_start) * (t / max(H, 1))
    score = (1.0 - wp) * speed + wp * debt_norm

    selected = []
    if q > 0:
        order = np.lexsort((rng.random(N), n))
        selected = order[:q].tolist()

    remaining = [i for i in range(N) if i not in selected]
    remaining_sorted = sorted(remaining, key=lambda i: (-score[i], rng.random()))
    selected += remaining_sorted[: K - len(selected)]
    return selected


def coverage_oracle_select(client_class_counts, K, global_counts, rng):
    N = len(client_class_counts)
    target = global_counts * (K / N)
    best_score = -1e18
    best_combo = None

    for combo in itertools.combinations(range(N), K):
        agg = np.sum([client_class_counts[i] for i in combo], axis=0)
        score = float(np.sum(np.minimum(agg, target)))
        if score > best_score + 1e-12:
            best_score = score
            best_combo = combo
        elif abs(score - best_score) <= 1e-12 and rng.random() < 0.5:
            best_combo = combo

    return list(best_combo)


# One run
def run_one(
        seed, alpha, ps, policy, wp_end, args, train_x, train_y, test_x, test_y,
        device
):
    set_seed(seed)
    N = args.num_clients
    K = args.K
    H = args.H

    client_indices = dirichlet_partition(train_y.numpy(), N, alpha, seed)
    rates = assign_rates(N, ps, seed)
    client_class_counts = [
        np.bincount(train_y[idx].numpy(), minlength=10) for idx in client_indices
    ]
    global_counts = np.bincount(train_y.numpy(), minlength=10)

    client_data = []
    for idx in client_indices:
        client_data.append((train_x[idx], train_y[idx]))

    torch.manual_seed(seed)
    global_model = MLP().to(device)
    payload_bytes = sum(p.numel() for p in global_model.parameters()) * 4

    n = np.zeros(N, dtype=np.int64)
    tau = (H * K) // N

    policy_hash = abs(hash((policy, int(wp_end * 1000)))) % 100000
    rng = np.random.default_rng(seed * 1000 + policy_hash)

    wall = 0.0
    bytes_total = 0
    rows = []

    for t in range(H):
        if policy == "random":
            S = rng.choice(N, size=K, replace=False).tolist()
        elif policy == "fastest":
            S = score_based_select(n, rates, t, H, K, 0, 0.0, 0.0, tau, rng)
        elif policy == "static":
            S = score_based_select(n, rates, t, H, K, 1, 0.5, 0.5, tau, rng)
        elif policy == "dspdeq":
            S = score_based_select(n, rates, t, H, K, 1, 0.2, wp_end, tau, rng)
        elif policy == "oracle":
            S = coverage_oracle_select(client_class_counts, K, global_counts, rng)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        local_states = []
        weights = []
        for i in S:
            cx, cy = client_data[i]
            state = local_train(
                global_model, cx, cy,
                epochs=args.local_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
            )
            local_states.append(state)
            weights.append(len(cx))

        aggregate(global_model, local_states, weights)

        round_time = max(
            args.T_COMP + 8.0 * payload_bytes / rates[i] + args.L_FIXED
            for i in S
        ) + args.T_AGG + args.T_CTRL

        wall += round_time
        bytes_total += len(S) * payload_bytes

        for i in S:
            n[i] += 1

        if (t + 1) % 5 == 0 or (t + 1) == H:
            acc, per_class, worst = evaluate(global_model, test_x, test_y, device)
            per_class_str = json.dumps([float(x) for x in per_class])
        else:
            if len(rows) > 0:
                acc = rows[-1]["acc"]
                worst = rows[-1]["worst"]
                per_class_str = rows[-1]["per_class"]
            else:
                acc, per_class, worst = evaluate(global_model, test_x, test_y, device)
                per_class_str = json.dumps([float(x) for x in per_class])

        jain = jain_index(n)

        rows.append({
            "run_id": f"{seed}_{alpha}_{ps}_{policy}_{wp_end}",
            "seed": seed,
            "alpha": alpha,
            "ps": ps,
            "policy": policy,
            "wp_end": wp_end,
            "round": t + 1,
            "wall": wall,
            "bytes": bytes_total,
            "acc": acc,
            "worst": worst,
            "jain": jain,
            "round_time": round_time,
            "per_class": per_class_str,
            "counts": json.dumps([int(x) for x in n]),
            "selected": json.dumps([int(x) for x in S]),
        })

    return rows

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--out", type=str, default="results/raw_results.csv")
    # Validation Guide: begin with 10 independent seeds per randomized scenario.
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--H", type=int, default=50)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--T_COMP", type=float, default=0.3)
    parser.add_argument("--L_FIXED", type=float, default=0.01)
    parser.add_argument("--T_AGG", type=float, default=0.01)
    parser.add_argument("--T_CTRL", type=float, default=0.05)
    parser.add_argument("--oracle_seeds", type=int, default=2)
    parser.add_argument("--include_sanity", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}, H: {args.H}, Oracle seeds: {args.oracle_seeds}")

    print("Loading MNIST data to memory...")
    train_dataset = datasets.MNIST(root=args.data_root, train=True, download=True)
    test_dataset = datasets.MNIST(root=args.data_root, train=False, download=True)

    train_x = train_dataset.data.float().unsqueeze(1) / 255.0
    train_x = (train_x - 0.1307) / 0.3081
    train_y = train_dataset.targets
    test_x = test_dataset.data.float().unsqueeze(1) / 255.0
    test_x = (test_x - 0.1307) / 0.3081
    test_y = test_dataset.targets

    policies = ["random", "fastest", "static", "dspdeq", "oracle"]
    alphas = [0.1] + ([1.0] if args.include_sanity else [])
    ps_list = [0.2, 0.5, 0.8]

    fieldnames = [
        "run_id", "seed", "alpha", "ps", "policy", "wp_end",
        "round", "wall", "bytes", "acc", "worst", "jain",
        "round_time", "per_class", "counts", "selected",
    ]

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total_runs = len(alphas) * len(ps_list) * args.seeds * len(policies)
        pbar = tqdm(total=total_runs, desc="runs")

        for seed in range(args.seeds):
            for alpha in alphas:
                for ps in ps_list:
                    for policy in policies:
                        if policy == "oracle" and seed >= args.oracle_seeds:
                            pbar.update(1)
                            continue

                        if policy == "dspdeq":
                            wp_ends = [0.5, 0.8, 1.0] if args.sweep else [0.8]
                        else:
                            wp_ends = [0.8]

                        for wp_end in wp_ends:
                            print(
                                f"\n>>> Running seed={seed}, alpha={alpha}, ps={ps}, policy={policy}, wp_end={wp_end}")

                            rows = run_one(
                                seed=seed,
                                alpha=alpha,
                                ps=ps,
                                policy=policy,
                                wp_end=wp_end,
                                args=args,
                                train_x=train_x,
                                train_y=train_y,
                                test_x=test_x,
                                test_y=test_y,
                                device=device,
                            )
                            writer.writerows(rows)
                            f.flush()

                        pbar.update(1)

        pbar.close()

    print(f"Saved raw results to {args.out}")


if __name__ == "__main__":
    main()