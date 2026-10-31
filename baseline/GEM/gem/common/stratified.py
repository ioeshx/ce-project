from typing import List, Sequence, Optional
import torch


# ---------- Stratified Sampling (Isolated RNG) ----------
def stratified_sampling(
    items: Sequence[int],
    n_total: int,
    generator: torch.Generator,
    weights: Optional[torch.Tensor] = None,  # shape (len(items),)
) -> List[int]:
    items = list(items)
    n = len(items)

    if weights is None:
        weights = torch.ones(n, dtype=torch.float)
    else:
        weights = weights.float().clamp_min(0)
        weights = weights / weights.sum()

    # allocate counts proportional to weights (multinomial counts)
    counts = torch.multinomial(weights, n_total, replacement=True, generator=generator)
    # counts are indices; turn into actual items
    out = [items[i] for i in counts.tolist()]
    perm = torch.randperm(len(out), generator=generator)
    out = [out[i] for i in perm.tolist()]

    return out

