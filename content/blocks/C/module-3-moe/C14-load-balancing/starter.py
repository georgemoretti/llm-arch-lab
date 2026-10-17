"""балансировка нагрузки экспертов — aux loss (Switch) и bias без градиента (DeepSeek-V3).

Реализуй:
1. expert_load — какая доля назначений досталась каждому эксперту;
2. load_balancing_loss — вспомогательный loss E · Σ f_i · P_i;
3. route_with_bias — выбор top-k с bias (веса — без bias);
4. update_bias — шаг «aux-loss-free» балансировки.
"""

import torch


def expert_load(indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    """indices [N, k] → доля назначений на каждого эксперта [E] (сумма 1)."""
    # TODO: bincount по всем назначениям (minlength!) → float → поделить на сумму.
    raise NotImplementedError("expert_load")


def load_balancing_loss(router_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Switch-style aux loss: E · Σ_i f_i · P_i. Идеальный баланс → 1, коллапс в одного эксперта → E."""
    # TODO 1: probs = softmax логитов по экспертам (float32).
    # TODO 2: f — доля назначений top-k на каждого эксперта (expert_load). Дифференцируема ли она?
    # TODO 3: P — средняя по токенам вероятность каждого эксперта.
    # TODO 4: E · Σ f_i · P_i.
    raise NotImplementedError("load_balancing_loss")


def route_with_bias(scores: torch.Tensor, bias: torch.Tensor, top_k: int, norm_topk_prob: bool = True):
    """DeepSeek-V3: bias сдвигает только ВЫБОР экспертов, веса берутся из исходных scores."""
    # TODO 1: индексы top-k — по scores + bias.
    # TODO 2: веса — исходные scores выбранных экспертов (gather), при norm_topk_prob — к сумме 1.
    raise NotImplementedError("route_with_bias")


def update_bias(bias: torch.Tensor, load: torch.Tensor, gamma: float) -> torch.Tensor:
    """Один шаг: перегруженным экспертам bias уменьшается на γ, недогруженным — увеличивается."""
    # TODO: перегружен (load > средней) → bias −γ; недогружен → +γ; ровно средний → без изменений.
    #       Шаг фиксированный (γ), а не пропорциональный перекосу. Какая функция даёт −1 / 0 / +1?
    raise NotImplementedError("update_bias")
