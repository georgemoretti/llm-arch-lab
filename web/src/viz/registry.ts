// Реестр визуализаций: id из content/viz/<id>.yaml → React-компонент.
// tests/test_viz_registry.py проверяет, что ключи здесь совпадают с файлами content/viz/.
// Новая визуализация: компонент в web/src/viz/, строка сюда, описание в content/viz/<id>.yaml,
// затем `:::viz <id>:::` в README задачи и/или `viz: [<id>]` в task.yaml.
import type { ComponentType } from "react";
import CausalMask from "./CausalMask";

export const VIZ_REGISTRY: Record<string, ComponentType> = {
  "causal-mask": CausalMask,
};
