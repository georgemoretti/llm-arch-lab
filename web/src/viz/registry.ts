// Реестр визуализаций: id из content/viz/<id>.yaml → React-компонент.
// tests/test_viz_registry.py проверяет, что ключи здесь совпадают с файлами content/viz/.
// Новая визуализация: компонент в web/src/viz/, строка сюда, описание в content/viz/<id>.yaml,
// затем `:::viz <id>:::` в README задачи и/или `viz: [<id>]` в task.yaml.
import type { ComponentType } from "react";
import Activations from "./Activations";
import AutogradGraph from "./AutogradGraph";
import BigramTable from "./BigramTable";
import Bpe from "./Bpe";
import CausalMask from "./CausalMask";
import ChatTemplate from "./ChatTemplate";
import CrossAttention from "./CrossAttention";
import DistillTemperature from "./DistillTemperature";
import EmbeddingSpace from "./EmbeddingSpace";
import EncdecMasks from "./EncdecMasks";
import GqaGroups from "./GqaGroups";
import GradCheck from "./GradCheck";
import Hybrid from "./Hybrid";
import InitDepth from "./InitDepth";
import KvCache from "./KvCache";
import LayerNormGrad from "./LayerNormGrad";
import Masks from "./Masks";
import LrSchedule from "./LrSchedule";
import MlpBoundary from "./MlpBoundary";
import MlmMasking from "./MlmMasking";
import MoeRouting from "./MoeRouting";
import Mtp from "./Mtp";
import Nope from "./Nope";
import NormPlacement from "./NormPlacement";
import Optimizers from "./Optimizers";
import ParamBudget from "./ParamBudget";
import QkNorm from "./QkNorm";
import RmsNorm from "./RmsNorm";
import RopeRotation from "./RopeRotation";
import SftCurves from "./SftCurves";
import Sampling from "./Sampling";
import Sinusoidal from "./Sinusoidal";
import SoftmaxCe from "./SoftmaxCe";
import SsdDecay from "./SsdDecay";
import StateMemory from "./StateMemory";
import T5Buckets from "./T5Buckets";
import TrainingCurve from "./TrainingCurve";
import Unpadding from "./Unpadding";
import Wordpiece from "./Wordpiece";

export const VIZ_REGISTRY: Record<string, ComponentType> = {
  "activations": Activations,
  "autograd-graph": AutogradGraph,
  "bigram-table": BigramTable,
  "bpe": Bpe,
  "causal-mask": CausalMask,
  "chat-template": ChatTemplate,
  "cross-attention": CrossAttention,
  "distill-temperature": DistillTemperature,
  "embedding-space": EmbeddingSpace,
  "encdec-masks": EncdecMasks,
  "gqa-groups": GqaGroups,
  "gradcheck": GradCheck,
  "hybrid": Hybrid,
  "init-depth": InitDepth,
  "kv-cache": KvCache,
  "layernorm-grad": LayerNormGrad,
  "masks": Masks,
  "lr-schedule": LrSchedule,
  "mlm-masking": MlmMasking,
  "mlp-boundary": MlpBoundary,
  "moe-routing": MoeRouting,
  "mtp": Mtp,
  "nope": Nope,
  "norm-placement": NormPlacement,
  "optimizers": Optimizers,
  "param-budget": ParamBudget,
  "qk-norm": QkNorm,
  "rmsnorm": RmsNorm,
  "rope-rotation": RopeRotation,
  "sft-curves": SftCurves,
  "sampling": Sampling,
  "sinusoidal": Sinusoidal,
  "softmax-ce": SoftmaxCe,
  "ssd-decay": SsdDecay,
  "state-memory": StateMemory,
  "t5-buckets": T5Buckets,
  "training-curve": TrainingCurve,
  "unpadding": Unpadding,
  "wordpiece": Wordpiece,
};
