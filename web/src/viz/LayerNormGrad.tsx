// V-layernorm-grad: из чего складывается dx у LayerNorm. g = dy·γ, затем вычитаются проекции на 1 (путь через μ)
// и на x̂ (путь через σ). Особые dy (константа, ∝ x̂) показывают, что эти направления дают нулевой градиент.
import { useMemo, useState } from "react";
import { BarRow, Bars, fmt, mean, randn, rng, Segmented, Slider } from "./common";

const C = 12;
type DyMode = "random" | "const" | "xhat" | "onehot";

export default function LayerNormGrad() {
  const [seed, setSeed] = useState(4);
  const [mode, setMode] = useState<DyMode>("random");
  const [shift, setShift] = useState(1.5);
  const [paths, setPaths] = useState<"all" | "mu" | "none">("all");

  const r = useMemo(() => {
    const next = rng(seed);
    const x = randn(next, C).map((v) => v + shift);
    const mu = mean(x);
    const varc = mean(x.map((v) => (v - mu) ** 2));
    const rstd = 1 / Math.sqrt(varc + 1e-5);
    const xhat = x.map((v) => (v - mu) * rstd);
    const rand = randn(next, C);
    const g =
      mode === "random" ? rand : mode === "const" ? Array(C).fill(0.8) : mode === "xhat" ? xhat.map((v) => 0.6 * v) : xhat.map((_, i) => (i === 3 ? 1 : 0));
    const mg = mean(g);
    const mgx = mean(g.map((v, i) => v * xhat[i]));
    const minusMu = g.map((v) => v - mg);
    const full = g.map((v, i) => v - mg - xhat[i] * mgx);
    const dx = (paths === "all" ? full : paths === "mu" ? minusMu : g).map((v) => v * rstd);
    return { x, xhat, g, minusMu, full, dx, rstd };
  }, [seed, mode, shift, paths]);

  const sum = r.dx.reduce((a, b) => a + b, 0);
  const dotXhat = r.dx.reduce((a, b, i) => a + b * r.xhat[i], 0);
  const max = Math.max(1e-9, ...r.g.map(Math.abs), ...r.minusMu.map(Math.abs), ...r.full.map(Math.abs));

  return (
    <div>
      <div className="viz-controls">
        <span>
          dy (γ = 1, поэтому g = dy):{" "}
          <Segmented
            value={mode}
            options={[
              ["random", "случайный"],
              ["const", "константа"],
              ["xhat", "∝ x̂"],
              ["onehot", "один элемент"],
            ]}
            onChange={setMode}
          />
        </span>
        <span>
          учесть пути:{" "}
          <Segmented
            value={paths}
            options={[
              ["all", "μ и σ"],
              ["mu", "только μ"],
              ["none", "никакие (ошибка)"],
            ]}
            onChange={setPaths}
          />
        </span>
      </div>
      <div className="viz-controls">
        <Slider label={<>сдвиг x = {shift.toFixed(1)}</>} value={shift} min={-3} max={3} step={0.1} onChange={setShift} />
        <button className="btn small" onClick={() => setSeed((s) => s + 1)}>
          новые x и dy
        </button>
      </div>
      <BarRow title="x̂ (нормированный вход)" stat={<>mean 0 · std 1</>}>
        <Bars values={r.xhat} color="var(--ax-model)" />
      </BarRow>
      <BarRow title="g = ∂L/∂x̂" stat={<>mean {fmt(mean(r.g))}</>}>
        <Bars values={r.g} max={max} color="var(--series-1)" />
      </BarRow>
      <BarRow title="g − mean(g)" stat={<>путь через μ</>}>
        <Bars values={r.minusMu} max={max} color="var(--series-1)" dimmed={r.minusMu.map(() => paths === "none")} />
      </BarRow>
      <BarRow title="… − x̂·mean(g·x̂)" stat={<>путь через σ</>}>
        <Bars values={r.full} max={max} color="var(--series-1)" dimmed={r.full.map(() => paths !== "all")} />
      </BarRow>
      <BarRow title="dx = rstd · (…)" stat={<>Σ dx = {fmt(sum, 4)} · ⟨dx, x̂⟩ = {fmt(dotXhat, 4)}</>}>
        <Bars values={r.dx} color={paths === "all" ? "var(--series-3)" : "var(--series-2)"} />
      </BarRow>
      <p className="viz-caption">
        LayerNorm не замечает, если ко всему x прибавить константу или растянуть его. Поэтому у верного dx (оба пути)
        сумма равна 0, а скалярное произведение с x̂ почти 0 (с точностью до eps). Выберите dy = константа или dy ∝ x̂:
        верный градиент станет нулевым, потому что такой dy просит изменить именно то, что LayerNorm всё равно
        нормирует. Режим «никакие» соответствует типичной ошибке, когда μ и σ считают константами: сумма уже не 0.
      </p>
    </div>
  );
}
