"""Difference of means with held-out diagnostics; never use test outcomes."""
import torch
from .evaluation import auc


def difference_of_means(activations, positive, negative):
    if sum(positive) < 2 or sum(negative) < 2:
        raise ValueError("Need at least two examples in both direction classes.")
    delta = activations[positive].mean(0) - activations[negative].mean(0)
    norms = delta.norm(dim=-1)
    # Layer 0 pre-residual is a constant final-token embedding and has zero delta.
    return delta / norms.clamp_min(1e-8).unsqueeze(-1), norms


def estimate(train, validation, train_rows, validation_rows, seed):
    risk = torch.tensor([r["risk"] == 1 for r in train_rows])
    observed = torch.tensor([r["refusal_marker"] for r in train_rows])
    val_risk = [r["risk"] for r in validation_rows]
    val_observed = [int(r["refusal_marker"]) for r in validation_rows]
    estimators = {"risk_dom": (risk, ~risk)}
    # Behavior-confirmed subset reduces risk-vs-refusal label conflation, but
    # remains dependent on an imperfect lexical refusal detector.
    if (risk & observed).sum() >= 2 and (~risk & ~observed).sum() >= 2:
        estimators["observed_dom"] = (risk & observed, ~risk & ~observed)
    directions, diagnostics = {}, []
    generator = torch.Generator().manual_seed(seed)
    for point in ("pre", "post"):
        for method, (positive, negative) in estimators.items():
            vectors, norms = difference_of_means(train[point], positive, negative)
            directions[point + "_" + method] = vectors
            pos, neg = train[point][positive], train[point][negative]
            stabilities = []
            for _ in range(40):
                p = pos[torch.randperm(len(pos), generator=generator)[:max(2, len(pos)//2)]]
                n = neg[torch.randperm(len(neg), generator=generator)[:max(2, len(neg)//2)]]
                delta = p.mean(0) - n.mean(0)
                delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                stabilities.append((delta * vectors).sum(-1))
            stability = torch.stack(stabilities).mean(0)
            for layer in range(train[point].shape[1]):
                scores = validation[point][:, layer] @ vectors[layer]
                diagnostics.append({"point": point, "method": method, "layer": layer,
                                    "mean_difference_norm": norms[layer].item(),
                                    "positive_n": int(positive.sum()), "negative_n": int(negative.sum()),
                                    "validation_risk_auc": auc(scores.numpy(), val_risk),
                                    "validation_observed_refusal_auc": auc(scores.numpy(), val_observed),
                                    "subsample_cosine_mean": stability[layer].item()})
    return directions, diagnostics
