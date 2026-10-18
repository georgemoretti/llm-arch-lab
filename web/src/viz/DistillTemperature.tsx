// V-distill-temperature: мягкие метки учителя softmax(v/T) на одном игрушечном примере, энтропия и «тёмное знание»
// в малых вероятностях; масштаб градиента ‖T·(q − p)‖ с множителем T² и без него в зависимости от T.
import { useMemo, useState } from "react";
import { LineChart } from "./LineChart";
import { BarRow, Bars, Segmented, Slider, entropyBits, softmax } from "./common";

// Логиты придуманы для курса: «Самый большой город на юге Франции — [MASK].»
const WORDS = ["марсель", "ницца", "тулуза", "лион", "париж", "монако", "барселона", "берлин", "рыба", "банан"];
const TEACHER = [9.0, 7.4, 6.6, 5.9, 5.6, 4.6, 3.8, 1.2, -0.8, -2.0];
// ученик, ещё не обученный: верный порядок наверху, но «плоский» хвост
const STUDENT = [6.5, 6.0, 3.0, 4.0, 5.5, 1.0, 3.5, 2.5, 1.5, 1.0];

const temps = Array.from({ length: 41 }, (_, i) => Math.pow(10, -0.3 + i * (1.6 / 40))); // 0.5 … 20

function gradNorm(T: number, withT2: boolean) {
  const p = softmax(TEACHER.map((v) => v / T));
  const q = softmax(STUDENT.map((v) => v / T));
  // ∂(T²·KL)/∂z = T·(q − p); без T² — (q − p)/T
  const k = withT2 ? T : 1 / T;
  return Math.sqrt(q.reduce((s, qi, i) => s + (k * (qi - p[i])) ** 2, 0));
}

export default function DistillTemperature() {
  const [T, setT] = useState(2);
  const [view, setView] = useState<"teacher" | "both">("teacher");
  const p = useMemo(() => softmax(TEACHER.map((v) => v / T)), [T]);
  const q = useMemo(() => softmax(STUDENT.map((v) => v / T)), [T]);
  const p1 = useMemo(() => softmax(TEACHER), []);
  const tail = p.slice(1).reduce((a, b) => a + b, 0);
  const tail1 = p1.slice(1).reduce((a, b) => a + b, 0);
  const kl = p.reduce((s, pi, i) => s + pi * (Math.log(pi) - Math.log(q[i])), 0);
  const fmtP = (v: number) => (v >= 0.01 ? v.toFixed(2) : v.toExponential(0));

  return (
    <div>
      <div className="viz-controls">
        <Slider label={<>температура T = {T.toFixed(1)}</>} value={T} min={0.5} max={10} step={0.1} onChange={setT} />
        <Segmented
          value={view}
          options={[
            ["teacher", "учитель"],
            ["both", "учитель и ученик"],
          ]}
          onChange={setView}
        />
      </div>
      <div className="viz-subtitle">«Самый большой город на юге Франции — [MASK].» Логиты придуманы для примера.</div>
      <BarRow title={<>учитель, softmax(v / {T.toFixed(1)})</>} stat={<>H = {entropyBits(p).toFixed(2)} бит</>}>
        <Bars values={p} max={1} labels={WORDS.map((w) => w.slice(0, 6))} titles={WORDS.map((w, i) => `${w}: ${fmtP(p[i])}`)} barWidth={40} height={110} />
      </BarRow>
      {view === "both" && (
        <BarRow title={<>ученик, softmax(z / {T.toFixed(1)})</>} stat={<>H = {entropyBits(q).toFixed(2)} бит</>}>
          <Bars values={q} max={1} color="var(--series-2)" labels={WORDS.map((w) => w.slice(0, 6))} titles={WORDS.map((w, i) => `${w}: ${fmtP(q[i])}`)} barWidth={40} height={110} />
        </BarRow>
      )}
      <div className="kv-tiles">
        <div>
          <span className="muted">масса вне top-1</span>
          <b>{(tail * 100).toFixed(1)} %</b>
          <span className="muted">при T = 1: {(tail1 * 100).toFixed(1)} %</span>
        </div>
        <div>
          <span className="muted">p(рыба) / p(банан)</span>
          <b>{(p[8] / p[9]).toFixed(2)}</b>
          <span className="muted">отношение = e^(Δv / T): T сглаживает, но порядок тот же</span>
        </div>
        <div>
          <span className="muted">T²·KL(p ‖ q)</span>
          <b>{(kl * T * T).toFixed(3)}</b>
          <span className="muted">без T²: {kl.toFixed(4)}</span>
        </div>
      </div>
      <LineChart
        xs={temps}
        logX
        series={[
          { id: "t2", label: "с множителем T²", color: "var(--series-1)", ys: temps.map((t) => gradNorm(t, true)) },
          { id: "no", label: "без T²", color: "var(--series-2)", ys: temps.map((t) => gradNorm(t, false)) },
        ]}
        xTicks={[0.5, 1, 2, 5, 10, 20]}
        xFormat={(v) => String(v)}
        yFormat={(v) => v.toFixed(2)}
        yLabel="‖∂L/∂z‖ для этого примера"
        xLabel="температура T (лог. шкала)"
        markerX={T}
        height={200}
      />
      <p className="viz-caption">
        При T = 1 учитель почти уверен в одном ответе, и ученику достаётся та же информация, что из правильной метки.
        С ростом T вероятности «ницца», «тулуза», «лион» вырастают: видно, какие ответы учитель считает близкими, а
        «банан» и «рыба» остаются внизу. Градиент мягкой части равен T·(q − p): без множителя T² он падает как 1/T²
        (линия «без T²»), и вклад дистилляции в общий лосс пришлось бы подбирать заново для каждой T. С T² масштаб
        выходит на постоянный уровень, а при больших T ученик просто подгоняет свои логиты к логитам учителя.
      </p>
    </div>
  );
}
