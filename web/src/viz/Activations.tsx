// V-activations: ReLU / GELU / SiLU и как гейт SwiGLU «открывает» каналы ветки up.
import { useMemo, useState } from "react";
import { BarRow, Bars, randn, rng, Slider } from "./common";
import { LineChart } from "./LineChart";

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const silu = (z: number) => z * sigmoid(z);
// tanh-аппроксимация, как в GPT-2
const gelu = (z: number) => 0.5 * z * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (z + 0.044715 * z ** 3)));

const XS = Array.from({ length: 81 }, (_, i) => -5 + i * 0.125);
const F = 12;

export default function Activations() {
  const [gain, setGain] = useState(1);
  const [seed, setSeed] = useState(7);

  const curves = useMemo(
    () => [
      { id: "relu", label: "ReLU", color: "var(--series-3)", ys: XS.map((z) => Math.max(0, z)) },
      { id: "gelu", label: "GELU", color: "var(--series-2)", ys: XS.map(gelu) },
      { id: "silu", label: "SiLU", color: "var(--series-1)", ys: XS.map(silu) },
    ],
    [],
  );

  const { g, u, h } = useMemo(() => {
    const next = rng(seed);
    const g = randn(next, F).map((v) => v * 1.5 * gain);
    const u = randn(next, F);
    return { g, u, h: g.map((v, i) => silu(v) * u[i]) };
  }, [gain, seed]);
  const max = Math.max(...u.map(Math.abs), ...h.map(Math.abs), 1);

  return (
    <div>
      <LineChart
        xs={XS}
        series={curves}
        xTicks={[-4, -2, 0, 2, 4]}
        xFormat={(x) => x.toFixed(1)}
        yFormat={(y) => y.toFixed(2)}
        xLabel="z"
        yLabel="f(z)"
        zeroLine
        height={200}
      />
      <div className="viz-subtitle">Один токен, {F} скрытых каналов SwiGLU</div>
      <div className="viz-controls">
        <Slider label={<>масштаб W_gate × {gain.toFixed(2)}</>} value={gain} min={0} max={3} step={0.05} onChange={setGain} />
        <button className="btn small" onClick={() => setSeed((s) => s + 1)}>
          новый токен
        </button>
      </div>
      <BarRow title="SiLU(gate)" stat="насколько открыт канал">
        <Bars values={g.map(silu)} max={max} color="var(--series-2)" />
      </BarRow>
      <BarRow title="up" stat="что пропустить">
        <Bars values={u} max={max} color="var(--series-3)" />
      </BarRow>
      <BarRow title="SiLU(gate) ⊙ up" stat="вход down_proj">
        <Bars values={h} max={max} color="var(--series-1)" />
      </BarRow>
      <p className="viz-caption">
        SiLU похожа на GELU: гладкая, у нуля пропускает немного отрицательных значений (минимум ≈ −0.278 при z ≈ −1.28),
        при больших z почти совпадает с z. В SwiGLU нелинейность стоит только на ветке gate: канал с отрицательным gate почти
        закрыт, какое бы значение ни несла ветка up. Сведите масштаб W_gate к 0 — выход станет ровно нулём.
      </p>
    </div>
  );
}
