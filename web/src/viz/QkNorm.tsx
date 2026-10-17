// V-qk-norm: рост весов W_Q / W_K «схлопывает» softmax; QK-Norm делает внимание к этому нечувствительным.
import { useMemo, useState } from "react";
import { BarRow, Bars, dot, entropyBits, randn, rms, rng, Slider, softmax } from "./common";

const D = 16;
const S = 12;

export default function QkNorm() {
  const [scale, setScale] = useState(1);
  const [seed, setSeed] = useState(11);

  const { plain, normed } = useMemo(() => {
    const next = rng(seed);
    const q = randn(next, D);
    const ks = Array.from({ length: S }, () => randn(next, D));
    // Веса W_Q и W_K выросли в scale раз → q и k тоже, а score — в scale² раз.
    const plain = softmax(ks.map((k) => (dot(q, k) * scale * scale) / Math.sqrt(D)));
    // RMSNorm по размеру головы (вес 1): |q| = |k| = √D при любом scale.
    const norm = (v: number[]) => v.map((x) => (x * scale) / rms(v.map((y) => y * scale)));
    const qn = norm(q);
    const normed = softmax(ks.map((k) => dot(qn, norm(k)) / Math.sqrt(D)));
    return { plain, normed };
  }, [scale, seed]);

  const stat = (p: number[]) => (
    <>
      max {Math.max(...p).toFixed(2)} · энтропия {entropyBits(p).toFixed(2)} бит
    </>
  );

  return (
    <div>
      <div className="viz-controls">
        <Slider label={<>веса W_Q, W_K × {scale.toFixed(1)}</>} value={scale} min={0.2} max={6} step={0.1} onChange={setScale} width={130} />
        <button className="btn small" onClick={() => setSeed((s) => s + 1)}>
          новые q, k
        </button>
      </div>
      <BarRow title="без QK-Norm" stat={stat(plain)}>
        <Bars values={plain} max={1} color="var(--series-2)" labels={plain.map((_, i) => `k${i}`)} />
      </BarRow>
      <BarRow title="с QK-Norm" stat={stat(normed)}>
        <Bars values={normed} max={1} color="var(--series-1)" labels={normed.map((_, i) => `k${i}`)} />
      </BarRow>
      <p className="viz-caption">
        Веса внимания одного запроса по {S} ключам. Если при обучении нормы W_Q и W_K растут в s раз, логиты растут в s² раз,
        softmax становится почти one-hot (энтропия → 0), а градиенты через него — почти нулевыми: так выглядит нестабильность
        больших моделей. QK-Norm нормирует q и k по размеру головы, |score| ≤ √D, и распределение не зависит от масштаба
        весов. Остаётся обучаемый вес нормы — им модель сама выбирает «температуру».
      </p>
    </div>
  );
}
