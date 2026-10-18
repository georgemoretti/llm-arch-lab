// Реестр визуализаций: id из content/viz/<id>.yaml → React-компонент.
// tests/test_viz_registry.py проверяет, что ключи здесь совпадают с файлами content/viz/.
// Новая визуализация: компонент в web/src/viz/, строка сюда, описание в content/viz/<id>.yaml,
// затем `:::viz <id>:::` в README задачи и/или `viz: [<id>]` в task.yaml.
import type { ComponentType } from "react";
import Activations from "./Activations";
import AutogradGraph from "./AutogradGraph";
import CausalMask from "./CausalMask";
import GqaGroups from "./GqaGroups";
import GradCheck from "./GradCheck";
import Hybrid from "./Hybrid";
import KvCache from "./KvCache";
import Masks from "./Masks";
import MlpBoundary from "./MlpBoundary";
import MoeRouting from "./MoeRouting";
import Mtp from "./Mtp";
import Nope from "./Nope";
import NormPlacement from "./NormPlacement";
import ParamBudget from "./ParamBudget";
import QkNorm from "./QkNorm";
import RmsNorm from "./RmsNorm";
import RopeRotation from "./RopeRotation";
import Sampling from "./Sampling";
import SoftmaxCe from "./SoftmaxCe";
import SsdDecay from "./SsdDecay";
import StateMemory from "./StateMemory";

export const VIZ_REGISTRY: Record<string, ComponentType> = {
  "activations": Activations,
  "autograd-graph": AutogradGraph,
  "causal-mask": CausalMask,
  "gqa-groups": GqaGroups,
  "gradcheck": GradCheck,
  "hybrid": Hybrid,
  "kv-cache": KvCache,
  "masks": Masks,
  "mlp-boundary": MlpBoundary,
  "moe-routing": MoeRouting,
  "mtp": Mtp,
  "nope": Nope,
  "norm-placement": NormPlacement,
  "param-budget": ParamBudget,
  "qk-norm": QkNorm,
  "rmsnorm": RmsNorm,
  "rope-rotation": RopeRotation,
  "sampling": Sampling,
  "softmax-ce": SoftmaxCe,
  "ssd-decay": SsdDecay,
  "state-memory": StateMemory,
};
