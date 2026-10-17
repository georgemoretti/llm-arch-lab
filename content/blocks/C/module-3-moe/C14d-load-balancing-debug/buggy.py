"""балансировка нагрузки экспертов — код из «Балансировка экспертов» с одной ошибкой. Найди и исправь её."""

import torch


def expert_load(indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    """indices [N, k] → доля назначений на каждого эксперта [E] (сумма 1)."""
    counts = torch.bincount(indices.reshape(-1), minlength=n_experts).float()
    return counts / counts.sum()


def load_balancing_loss(router_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Switch-style aux loss: E · Σ_i f_i · P_i. Идеальный баланс → 1, коллапс в одного эксперта → E."""
    N, E = router_logits.shape
    probs = router_logits.float().softmax(dim=-1)                  # [N, E], дифференцируемо
    indices = probs.topk(top_k, dim=-1).indices                    # [N, k]
    f = expert_load(indices, E)                                    # [E] доля назначений — БЕЗ градиента
    P = probs.mean(dim=0)                                          # [E] средняя вероятность — С градиентом
    return E * (f * P).sum()


def route_with_bias(scores: torch.Tensor, bias: torch.Tensor, top_k: int, norm_topk_prob: bool = True):
    """DeepSeek-V3: bias сдвигает только ВЫБОР экспертов, веса берутся из исходных scores."""
    indices = (scores + bias).topk(top_k, dim=-1).indices          # [N, k] — выбор с учётом bias
    weights = scores.gather(-1, indices)                           # [N, k] — веса без bias
    if norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, indices


def update_bias(bias: torch.Tensor, load: torch.Tensor, gamma: float) -> torch.Tensor:
    """Один шаг bias-балансировки по текущей нагрузке экспертов."""
    return bias + gamma * torch.sign(load - load.mean())
