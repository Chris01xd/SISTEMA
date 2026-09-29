import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { useTranslation } from "@/i18n/useTranslation";
import {
  downloadMLReport,
  type EvaluationRow,
  getMLStatus,
  type MLAnalysisResult,
  runMLAnalysis,
  type TrainedModel,
} from "@/services/mlAnalysisApi";

const fmt = (v: number | null | undefined, digits = 3) => (v === null || v === undefined ? "—" : v.toFixed(digits));
const COLORS = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2", "#ca8a04", "#4b5563"];
const TARGETS: Record<string, string[]> = { mimic_icu: ["icu_flag", "died_in_hospital"], mimic_mortality: ["died_in_hospital", "icu_flag"] };

function errorDetail(err: unknown, fallback: string): string {
  const detail = (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object" && "message" in detail) return String((detail as { message: unknown }).message);
  if (Array.isArray(detail)) return detail.map((d) => (d as { msg?: string }).msg ?? "").join("; ");
  return fallback;
}

// ------------------------------------------------------------------ building blocks

function Interpretation({ text }: { text?: string }) {
  const { t } = useTranslation();
  if (!text) return null;
  return (
    <p className="rounded-lg bg-indigo-50 p-3 text-xs leading-relaxed text-gray-700 ring-1 ring-indigo-100 dark:bg-indigo-950/30 dark:text-gray-300 dark:ring-indigo-900/50">
      <span className="font-semibold text-indigo-700 dark:text-indigo-300">{t("ml.interpretation")}: </span>
      {text}
    </p>
  );
}

function Block({ title, interpretation, children }: { title: string; interpretation?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <h4 className="text-sm font-semibold text-gray-900 dark:text-white">{title}</h4>
      {children}
      <Interpretation text={interpretation} />
    </div>
  );
}

function Phase({ title, summary, children }: { title: string; summary?: string; children: React.ReactNode }) {
  return (
    <section className="space-y-4 rounded-xl border border-gray-200 p-4 dark:border-gray-800">
      <div>
        <h3 className="text-sm font-bold uppercase tracking-widest text-gray-700 dark:text-gray-300">{title}</h3>
        {summary && <p className="text-xs text-gray-500 dark:text-gray-400">{summary}</p>}
      </div>
      {children}
    </section>
  );
}

function TypeBadge({ type }: { type: string }) {
  const { t } = useTranslation();
  const hybrid = type === "hybrid";
  return (
    <span className={`rounded px-2 py-0.5 text-xs font-semibold ${hybrid ? "bg-purple-100 text-purple-800 dark:bg-purple-900/50 dark:text-purple-200" : "bg-sky-100 text-sky-800 dark:bg-sky-900/50 dark:text-sky-200"}`}>
      {hybrid ? t("ml.hybrid") : t("ml.normal")}
    </span>
  );
}

function SimpleTable({ columns, rows }: { columns: string[]; rows: (string | number | null | undefined)[][] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead className="uppercase text-gray-500 dark:text-gray-400">
          <tr>{columns.map((c) => <th key={c} className="py-1.5 pr-3">{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-t border-gray-200 dark:border-gray-800">
              {row.map((cell, j) => (
                <td key={j} className="py-1.5 pr-3 tabular-nums text-gray-700 dark:text-gray-200">
                  {typeof cell === "number" ? fmt(cell) : cell ?? "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ------------------------------------------------------------------ CRISP-DM 2: data understanding

function EdaSection({ result }: { result: MLAnalysisResult }) {
  const { t } = useTranslation();
  const eda = result.eda;
  const it = result.interpretations;
  const [feature, setFeature] = useState(eda.numeric[0]?.feature ?? "");
  const selected = eda.numeric.find((n) => n.feature === feature);
  const hist = (selected?.histogram.positive ?? []).map((pos, i) => ({
    bin: fmt(selected?.histogram.edges[i] ?? null, 1),
    positive: pos,
    negative: selected?.histogram.negative[i] ?? 0,
  }));
  const topCat = [...eda.categorical].sort((a, b) => (b.association.cramers_v ?? 0) - (a.association.cramers_v ?? 0))[0];
  return (
    <>
      <Block title={t("ml.edaOverview")} interpretation={it.eda_overview}>
        <SimpleTable
          columns={[t("ml.metric"), t("ml.value")]}
          rows={[
            ["rows", eda.overview.rows], ["features", eda.overview.features], [`${eda.overview.target} = 1`, `${(100 * eda.overview.positive_rate).toFixed(1)}%`],
            ["complete rows", `${eda.overview.complete_rows_pct.toFixed(1)}%`], ["duplicated rows", eda.overview.duplicated_rows], ["source", result.dataset.source],
          ]}
        />
      </Block>
      <Block title={t("ml.edaMissing")} interpretation={it.eda_missing}>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={eda.missing.filter((m) => m.missing > 0)} layout="vertical" margin={{ left: 120 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" unit="%" domain={[0, 100]} />
            <YAxis type="category" dataKey="feature" width={130} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="missing_pct" fill="#f59e0b" name="% missing" />
          </BarChart>
        </ResponsiveContainer>
      </Block>
      <Block title={t("ml.edaNumeric")} interpretation={it.eda_numeric}>
        <SimpleTable
          columns={[t("ml.feature"), "mean (+)", "mean (−)", "skew", "outliers", "p (Holm)", "rank-biserial", "sig."]}
          rows={eda.numeric.map((n) => [n.feature, n.mean_positive, n.mean_negative, n.skewness, n.outliers_iqr,
            n.association.p_value_holm ?? null, n.association.effect_size_rank_biserial, n.association.significant ? "✓" : "—"])}
        />
        <div className="flex items-center gap-2 pt-2 text-xs text-gray-600 dark:text-gray-300">
          {t("ml.distribution")}
          <select value={feature} onChange={(e) => setFeature(e.target.value)} className="rounded bg-gray-200 px-2 py-1 dark:bg-gray-800">
            {eda.numeric.map((n) => <option key={n.feature}>{n.feature}</option>)}
          </select>
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={hist}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Legend />
            <Bar dataKey="positive" name={`${eda.overview.target} = 1`} fill="#dc2626" />
            <Bar dataKey="negative" name={`${eda.overview.target} = 0`} fill="#2563eb" />
          </BarChart>
        </ResponsiveContainer>
      </Block>
      {topCat && (
        <Block title={t("ml.edaCategorical")} interpretation={it.eda_categorical}>
          <SimpleTable
            columns={[t("ml.feature"), "levels", "Cramér V", "p (Holm)", "sig."]}
            rows={eda.categorical.map((c) => [c.feature, c.n_levels, c.association.cramers_v, c.association.p_value_holm ?? null, c.association.significant ? "✓" : "—"])}
          />
          <p className="pt-2 text-xs text-gray-600 dark:text-gray-300">{t("ml.positiveRateBy")} {topCat.feature}</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={topCat.levels.map((l) => ({ level: l.level, rate: 100 * (l.positive_rate ?? 0), n: l.count }))}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="level" tick={{ fontSize: 9 }} interval={0} angle={-15} height={50} />
              <YAxis unit="%" />
              <Tooltip />
              <Bar dataKey="rate" fill="#9333ea" name="% positive" />
            </BarChart>
          </ResponsiveContainer>
        </Block>
      )}
      <Block title={t("ml.edaCorrelations")} interpretation={it.eda_correlations}>
        <div className="overflow-x-auto">
          <table className="text-[10px]">
            <tbody>
              <tr><td />{eda.correlations.features.map((f) => <td key={f} className="px-1 text-gray-500 [writing-mode:vertical-rl]">{f}</td>)}</tr>
              {eda.correlations.matrix.map((row, i) => (
                <tr key={i}>
                  <td className="pr-2 text-gray-500">{eda.correlations.features[i]}</td>
                  {row.map((v, j) => (
                    <td key={j} className="h-7 w-10 text-center tabular-nums text-gray-900"
                      style={{ background: v === null ? "transparent" : v >= 0 ? `rgba(220,38,38,${Math.abs(v)})` : `rgba(37,99,235,${Math.abs(v)})` }}>
                      {v === null ? "" : v.toFixed(2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Block>
    </>
  );
}

// ------------------------------------------------------------------ CRISP-DM 4-5: modeling and evaluation

function TrainedModels({ models }: { models: TrainedModel[] }) {
  const { t } = useTranslation();
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {models.map((m) => (
        <div key={m.key} className="rounded-lg bg-gray-100 p-3 text-sm dark:bg-gray-900">
          <div className="flex items-center justify-between gap-2">
            <p className="font-semibold text-gray-900 dark:text-white">{m.name}</p>
            <TypeBadge type={m.family} />
          </div>
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">{m.description}</p>
          {m.components.length > 0 && <p className="mt-1 text-xs text-gray-600 dark:text-gray-300">{m.components.join(" + ")}</p>}
          <p className={`mt-2 text-xs font-semibold ${m.status === "ok" ? "text-status-green" : "text-status-red"}`}>
            {m.status === "ok" ? `${t("ml.trained")} · ${fmt(m.training_time_s, 2)} s` : `${m.status}: ${m.error ?? ""}`}
          </p>
        </div>
      ))}
    </div>
  );
}

function ComparisonTable({ rows, scores, bestKey }: { rows: EvaluationRow[]; scores: Record<string, number>; bestKey: string }) {
  const { t } = useTranslation();
  const cols: [keyof EvaluationRow, string][] = [
    ["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"], ["roc_auc", "ROC-AUC"],
    ["cv_mean", "CV Mean"], ["cv_std", "CV Std"], ["overfitting_gap", t("ml.overfittingGap")],
  ];
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[900px] text-left text-sm">
        <thead className="text-xs uppercase text-gray-500 dark:text-gray-400">
          <tr>
            <th className="py-2 pr-3">{t("ml.model")}</th>
            <th className="py-2 pr-3">{t("ml.type")}</th>
            {cols.map(([, label]) => <th key={label} className="py-2 pr-3 text-right">{label}</th>)}
            <th className="py-2 text-right">Final Score</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} className={`border-t border-gray-200 dark:border-gray-800 ${r.key === bestKey ? "bg-green-50 font-semibold dark:bg-green-950/30" : ""}`}>
              <td className="py-2 pr-3 text-gray-900 dark:text-white">{r.model}</td>
              <td className="py-2 pr-3"><TypeBadge type={r.type} /></td>
              {cols.map(([key]) => (
                <td key={String(key)} className="py-2 pr-3 text-right tabular-nums text-gray-700 dark:text-gray-200">
                  {r.status === "ok" ? fmt(r[key] as number | null) : "—"}
                </td>
              ))}
              <td className="py-2 text-right tabular-nums text-gray-900 dark:text-white">{r.status === "ok" ? fmt(scores[r.key], 4) : r.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Ranking({ result }: { result: MLAnalysisResult }) {
  const max = Math.max(...result.ranking.map((e) => e.final_score), 0.0001);
  return (
    <ol className="space-y-2">
      {result.ranking.map((e) => (
        <li key={e.key} className="flex items-center gap-3 text-sm">
          <span className="w-6 text-right font-bold text-gray-500">{e.rank}.</span>
          <span className="w-44 truncate text-gray-900 dark:text-white">{e.model}</span>
          <TypeBadge type={e.type} />
          <div className="h-3 flex-1 rounded bg-gray-200 dark:bg-gray-800">
            <div className={`h-3 rounded ${e.rank === 1 ? "bg-status-green" : e.type === "hybrid" ? "bg-purple-500" : "bg-sky-500"}`}
              style={{ width: `${Math.max(2, (100 * e.final_score) / max)}%` }} />
          </div>
          <span className="w-16 text-right tabular-nums text-gray-700 dark:text-gray-200">{e.final_score.toFixed(4)}</span>
        </li>
      ))}
    </ol>
  );
}

function CvFoldsChart({ result }: { result: MLAnalysisResult }) {
  const models = Object.values(result.cross_validation.models).filter((m) => m.status === "ok");
  const data = Array.from({ length: result.cross_validation.folds }, (_, i) =>
    Object.fromEntries([["fold", `fold ${i + 1}`], ...models.map((m) => [m.name, m.cv_scores[i]])]));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="fold" />
        <YAxis domain={["auto", "auto"]} />
        <Tooltip formatter={(v: number) => v.toFixed(3)} />
        <Legend />
        {models.map((m, i) => (
          <Line key={m.name} dataKey={m.name} stroke={COLORS[i % COLORS.length]} strokeWidth={m.name === result.best_model.name ? 3 : 1.5}
            strokeDasharray={m.family === "hybrid" ? "5 3" : undefined} dot />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

function RocChart({ result }: { result: MLAnalysisResult }) {
  const series = Object.entries(result.curves).filter(([, c]) => c.roc);
  return (
    <ResponsiveContainer width="100%" height={320}>
      <ScatterChart>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" dataKey="x" name="FPR" domain={[0, 1]} label={{ value: "FPR", position: "insideBottom", offset: -2 }} />
        <YAxis type="number" dataKey="y" name="TPR" domain={[0, 1]} label={{ value: "TPR", angle: -90, position: "insideLeft" }} />
        <Tooltip formatter={(v: number) => v.toFixed(3)} />
        <Legend />
        {series.map(([name, c], i) => (
          <Scatter key={name} name={name} data={c.roc!.x.map((x, j) => ({ x, y: c.roc!.y[j] }))} line={{ strokeWidth: name === result.best_model.name ? 3 : 1.2 }}
            fill={COLORS[i % COLORS.length]} stroke={COLORS[i % COLORS.length]} shape={() => <g />} />
        ))}
        <Scatter name="random" data={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} line={{ strokeDasharray: "4 4" }} stroke="#9ca3af" fill="#9ca3af" shape={() => <g />} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function StatisticsBlock({ result }: { result: MLAnalysisResult }) {
  const { t } = useTranslation();
  const st = result.statistical_tests;
  const p = (v?: number | null) => (v === null || v === undefined ? "—" : v < 0.001 ? "<0.001" : v.toFixed(3));
  return (
    <>
      <Block title={t("ml.statsTests")} interpretation={result.interpretations.statistical_tests}>
        <p className="text-xs text-gray-600 dark:text-gray-300">
          Friedman χ² = {fmt(st.friedman.statistic, 2)}, p = {p(st.friedman.p_value)} · Nemenyi CD = {fmt(st.nemenyi.critical_difference, 2)} ·
          α = {st.alpha} · {t("ml.evidence")}: <b>{st.summary.evidence}</b>
        </p>
        <SimpleTable
          columns={[`${st.best_model} vs`, "Δ CV (NB)", "NB t p(Holm)", "AUC best / other", "DeLong p(Holm)", "McNemar b/c", "McNemar p(Holm)"]}
          rows={st.pairwise_vs_best.map((r) => [r.model, r.nadeau_bengio.mean_difference, p(r.nadeau_bengio.p_value_holm),
            `${fmt(r.delong.auc_a)} / ${fmt(r.delong.auc_b)}`, p(r.delong.p_value_holm), `${r.mcnemar.a_right_b_wrong}/${r.mcnemar.a_wrong_b_right}`,
            p(r.mcnemar.p_value_holm)])}
        />
        <p className="text-xs text-gray-500 dark:text-gray-400">{t("ml.meanRanks")}: {Object.entries(st.nemenyi.mean_ranks).map(([m, r]) => `${m} ${r.toFixed(1)}`).join(" · ")}</p>
      </Block>
      <Block title={t("ml.bootstrapCi")} interpretation={result.interpretations.bootstrap_ci}>
        <SimpleTable
          columns={[t("ml.model"), "ROC-AUC 95% CI", "F1 95% CI"]}
          rows={Object.entries(st.bootstrap_ci_95).map(([m, c]) => [m, `[${fmt(c.roc_auc[0])}, ${fmt(c.roc_auc[1])}]`, `[${fmt(c.f1[0])}, ${fmt(c.f1[1])}]`])}
        />
      </Block>
    </>
  );
}

function BestModel({ result }: { result: MLAnalysisResult }) {
  const { t } = useTranslation();
  const best = result.best_model;
  const m = best.metrics;
  const cm = m.confusion_matrix ?? [[0, 0], [0, 0]];
  const stats: [string, string][] = [
    ["Final Score", best.final_score.toFixed(4)], ["Accuracy", fmt(m.accuracy)], ["Precision", fmt(m.precision)], ["Recall", fmt(m.recall)],
    ["F1", fmt(m.f1)], ["ROC-AUC", fmt(m.roc_auc)], ["CV Mean", fmt(m.cv_mean)], ["CV Std", fmt(m.cv_std)], [t("ml.overfittingGap"), fmt(m.overfitting_gap)],
  ];
  return (
    <section className="space-y-4 rounded-xl border-2 border-status-green p-4">
      <div className="flex flex-wrap items-center gap-3">
        <h3 className="text-sm font-bold uppercase tracking-widest text-status-green">{t("ml.bestModel")}</h3>
        <span className="text-xl font-bold text-gray-900 dark:text-white">{best.name}</span>
        <TypeBadge type={best.type} />
      </div>
      <div className="grid grid-cols-3 gap-3 md:grid-cols-5 lg:grid-cols-9">
        {stats.map(([label, value]) => (
          <div key={label} className="rounded-lg bg-gray-100 p-2 dark:bg-gray-900">
            <p className="text-xs text-gray-500 dark:text-gray-400">{label}</p>
            <p className="text-base font-bold tabular-nums text-gray-900 dark:text-white">{value}</p>
          </div>
        ))}
      </div>
      <Block title={t("ml.confusionMatrix")} interpretation={result.interpretations.confusion_matrix}>
        <table className="text-center text-sm">
          <thead><tr><th /><th className="px-4 text-xs text-gray-500">{t("ml.pred")} 0</th><th className="px-4 text-xs text-gray-500">{t("ml.pred")} 1</th></tr></thead>
          <tbody>
            {cm.map((row, i) => (
              <tr key={i}>
                <td className="pr-3 text-xs text-gray-500">{t("ml.actual")} {i}</td>
                {row.map((v, j) => (
                  <td key={j} className={`h-12 w-24 font-bold tabular-nums ${i === j ? "bg-green-100 text-green-900 dark:bg-green-900/40 dark:text-green-100" : "bg-red-100 text-red-900 dark:bg-red-900/40 dark:text-red-100"}`}>{v}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </Block>
      <div className="space-y-2">
        <h4 className="text-sm font-semibold text-gray-900 dark:text-white">{t("ml.whySelected")}</h4>
        <p className="text-sm text-gray-700 dark:text-gray-300">{result.selection_reason}</p>
        {result.analysis_source === "llm" && (
          <>
            <h4 className="pt-2 text-sm font-semibold text-gray-900 dark:text-white">{t("ml.llmAnalysis")}</h4>
            <p className="whitespace-pre-line text-sm text-gray-700 dark:text-gray-300">{result.analysis}</p>
          </>
        )}
        <p className="text-xs text-gray-500 dark:text-gray-400">{t("ml.artifact")}: {best.artifact_path}</p>
      </div>
    </section>
  );
}

// ------------------------------------------------------------------ page

export function ModelComparisonPage() {
  const { t, language } = useTranslation();
  const [dataset, setDataset] = useState("mimic_icu");
  const [target, setTarget] = useState("icu_flag");
  const [folds, setFolds] = useState(5);
  const [useLlm, setUseLlm] = useState(true);
  const status = useQuery({ queryKey: ["ml-status"], queryFn: getMLStatus });
  const analysis = useMutation({
    mutationFn: () => runMLAnalysis({ dataset, target, problem_type: "classification", cv_folds: folds, explain_with_llm: useLlm, language }),
  });
  const report = useMutation({ mutationFn: (format: "pdf" | "xlsx") => downloadMLReport(analysis.data!.run_id, format) });
  const result = analysis.data;
  const scores = Object.fromEntries((result?.ranking ?? []).map((e) => [e.key, e.final_score]));
  const phase = (i: number) => result?.crisp_dm[i];

  return (
    <div className="mx-auto max-w-7xl space-y-4 p-4 sm:p-6">
      <div className="space-y-1">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">{t("ml.title")}</h2>
        <p className="text-sm text-gray-600 dark:text-gray-300">{t("ml.subtitle")}</p>
        {status.data && (
          <p className="text-xs text-gray-500 dark:text-gray-400">
            {t("ml.engine")}: {status.data.mode === "langflow" ? `Langflow (${status.data.flow})` : t("ml.direct")}
            {status.data.mode === "langflow" && (
              <span className={status.data.langflow_reachable ? "text-status-green" : "text-status-red"}>
                {" · "}{status.data.langflow_reachable ? t("ml.langflowUp") : t("ml.langflowDown")}
              </span>
            )}
          </p>
        )}
      </div>

      <div className="flex flex-wrap items-end gap-3 rounded-xl border border-gray-200 p-4 dark:border-gray-800">
        <label className="text-xs text-gray-600 dark:text-gray-300">
          {t("ml.dataset")}
          <select value={dataset} onChange={(e) => { setDataset(e.target.value); setTarget(TARGETS[e.target.value][0]); }}
            className="mt-1 block rounded bg-gray-200 px-2 py-1 text-sm text-gray-900 dark:bg-gray-800 dark:text-white">
            {Object.keys(TARGETS).map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
        </label>
        <label className="text-xs text-gray-600 dark:text-gray-300">
          {t("ml.target")}
          <select value={target} onChange={(e) => setTarget(e.target.value)} className="mt-1 block rounded bg-gray-200 px-2 py-1 text-sm text-gray-900 dark:bg-gray-800 dark:text-white">
            {TARGETS[dataset].map((tg) => <option key={tg} value={tg}>{tg}</option>)}
          </select>
        </label>
        <label className="text-xs text-gray-600 dark:text-gray-300">
          {t("ml.folds")}
          <input type="number" min={3} max={10} value={folds} onChange={(e) => setFolds(Number(e.target.value))}
            className="mt-1 block w-20 rounded bg-gray-200 px-2 py-1 text-sm text-gray-900 dark:bg-gray-800 dark:text-white" />
        </label>
        <label className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-300">
          <input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
          {t("ml.useLlm")}
        </label>
        <button disabled={analysis.isPending} onClick={() => analysis.mutate()}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50">
          {analysis.isPending ? t("ml.running") : t("ml.run")}
        </button>
        {result && (
          <>
            <button onClick={() => report.mutate("pdf")} disabled={report.isPending}
              className="rounded-lg bg-gray-700 px-3 py-2 text-sm font-semibold text-white hover:bg-gray-600 disabled:opacity-50">PDF</button>
            <button onClick={() => report.mutate("xlsx")} disabled={report.isPending}
              className="rounded-lg bg-gray-700 px-3 py-2 text-sm font-semibold text-white hover:bg-gray-600 disabled:opacity-50">XLSX</button>
          </>
        )}
      </div>
      {analysis.isPending && <p className="text-xs text-gray-500">{t("ml.runningHint")}</p>}

      {(analysis.isError || report.isError) && (
        <p className="rounded-lg bg-red-100 p-3 text-sm text-red-700 ring-1 ring-red-300 dark:bg-red-950/40 dark:text-red-300 dark:ring-red-800/50">
          {errorDetail(analysis.error ?? report.error, t("ml.errorGeneric"))}
        </p>
      )}

      {result && (
        <>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            {t("ml.runInfo").replace("{run}", result.run_id).replace("{rows}", String(result.dataset.rows)).replace("{target}", result.dataset.target)
              .replace("{primary}", result.dataset.primary_metric).replace("{cv}", result.cross_validation.strategy).replace("{secs}", String(result.duration_s ?? "—"))}
            {" · "}{t("ml.engine")}: {result.engine}
          </p>
          {result.warnings.map((w) => <p key={w} className="rounded bg-yellow-100 p-2 text-xs text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-200">{w}</p>)}

          <Phase title={phase(0)?.phase ?? ""} summary={phase(0)?.summary}>
            <p className="text-xs text-gray-600 dark:text-gray-300">{result.dataset.target} · {result.dataset.source}</p>
          </Phase>
          <Phase title={phase(1)?.phase ?? ""} summary={phase(1)?.summary}><EdaSection result={result} /></Phase>
          <Phase title={phase(2)?.phase ?? ""} summary={phase(2)?.summary}>
            <Block title={t("ml.preprocessing")} interpretation={result.interpretations.preprocessing}>
              <ul className="list-disc space-y-1 pl-5 text-xs text-gray-600 dark:text-gray-300">
                {result.preprocessing.leakage_controls.map((c) => <li key={c}>{c}</li>)}
              </ul>
            </Block>
          </Phase>
          <Phase title={phase(3)?.phase ?? ""} summary={phase(3)?.summary}>
            <p className="text-xs font-semibold text-gray-500">{t("ml.trainedModels")} ({result.models_evaluated}/{result.models_attempted}) · {t("ml.normalModels")}</p>
            <TrainedModels models={result.normal_models} />
            <p className="pt-2 text-xs font-semibold text-gray-500">{t("ml.hybridModels")}</p>
            <TrainedModels models={result.hybrid_models} />
            <Block title={t("ml.hyperparameters")} interpretation={result.interpretations.hyperparameters}>
              <SimpleTable
                columns={[t("ml.model"), t("ml.grid"), t("ml.bestParams"), t("ml.innerScore"), t("ml.stability")]}
                rows={Object.entries(result.hyperparameters).map(([name, h]) => [name, h.grid_size,
                  Object.entries(h.best_params ?? {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ") || "—", h.best_inner_score, h.stability])}
              />
            </Block>
          </Phase>
          <Phase title={phase(4)?.phase ?? ""} summary={phase(4)?.summary}>
            <Block title={t("ml.cvFolds")} interpretation={result.interpretations.cross_validation}><CvFoldsChart result={result} /></Block>
            <Block title={t("ml.comparison")} interpretation={result.interpretations.comparison_table}>
              <ComparisonTable rows={result.comparison_table} scores={scores} bestKey={result.best_model.key} />
              <p className="text-xs text-gray-500 dark:text-gray-400">{t("ml.metricsNote")} {result.score_config.formula}</p>
            </Block>
            <Block title={t("ml.ranking")} interpretation={result.interpretations.ranking}><Ranking result={result} /></Block>
            <Block title={t("ml.crossReference")} interpretation={result.interpretations.cross_reference}>
              <SimpleTable
                columns={[t("ml.model"), "train", "CV", "test", "CV→test", t("ml.stability"), t("ml.generalization"), t("ml.overfitting")]}
                rows={Object.values(result.cross_reference_analysis.models).map((m) => [m.model, m.train_score, m.cv_mean, m.test_score,
                  m.cv_vs_test_difference, m.stability, m.generalization, m.overfitting ? "⚠" : "—"])}
              />
            </Block>
            <Block title={t("ml.rocCurves")} interpretation={result.interpretations.roc_curves}><RocChart result={result} /></Block>
            <StatisticsBlock result={result} />
          </Phase>
          <Phase title={phase(5)?.phase ?? ""} summary={phase(5)?.summary}>
            <BestModel result={result} />
          </Phase>
        </>
      )}
    </div>
  );
}
