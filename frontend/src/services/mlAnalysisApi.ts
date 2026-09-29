// Typed wrappers for backend/app/api/v1/ml_analysis.py (Langflow model-comparison flow).
import { api } from "@/services/api";

export type ModelFamily = "normal" | "hybrid";

export interface MLAnalysisRequest {
  dataset?: string;
  target?: string;
  problem_type?: "classification";
  cv_folds?: number;
  test_size?: number;
  random_state?: number;
  explain_with_llm?: boolean;
  language?: "es" | "en";
}

export interface SetMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number | null;
  pr_auc: number | null;
}

export interface EvaluationRow {
  model: string;
  key: string;
  type: ModelFamily;
  status: "ok" | "failed" | "skipped";
  error: string | null;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  roc_auc: number | null;
  pr_auc: number | null;
  cv_mean: number | null;
  cv_std: number | null;
  cv_metrics: Record<string, number | null>;
  train_score: number | null;
  test_score: number | null;
  overfitting_gap: number | null;
  cv_overfitting_gap: number | null;
  training_time_s: number | null;
  inference_ms_per_sample: number | null;
  confusion_matrix: number[][] | null;
}

export interface TrainedModel {
  key: string;
  name: string;
  family: ModelFamily;
  status: "ok" | "failed" | "skipped";
  error: string | null;
  description: string;
  components: string[];
  training_time_s: number | null;
  final_score: number | null;
}

export interface RankingEntry {
  rank: number;
  model: string;
  key: string;
  type: ModelFamily;
  final_score: number;
  score_breakdown: Record<string, number>;
}

export interface CrossReferenceModel {
  model: string;
  train_score: number | null;
  cv_mean: number | null;
  test_score: number | null;
  stability: "stable" | "unstable";
  generalization: "consistent" | "inconsistent";
  overfitting: boolean;
  cv_vs_test_difference: number;
  fold_scores: number[];
  flags: string[];
}

export interface MLAnalysisResult {
  status: "success";
  run_id: string;
  models_evaluated: number;
  models_attempted: number;
  normal_models: TrainedModel[];
  hybrid_models: TrainedModel[];
  comparison_table: EvaluationRow[];
  ranking: RankingEntry[];
  best_model: {
    name: string;
    key: string;
    type: ModelFamily;
    final_score: number;
    rank: number;
    metrics: EvaluationRow;
    artifact_path: string;
  };
  cross_validation: { strategy: string; folds: number; primary_metric: string; nested: boolean; inner_folds: number | null;
    models: Record<string, CvModel> };
  statistical_tests: StatisticalTests;
  hyperparameters: Record<string, { grid_size: number; best_params: Record<string, unknown>; best_inner_score: number | null;
    stability: number | null; fold_choices: Record<string, unknown>[] }>;
  curves: Record<string, { roc?: { x: number[]; y: number[] }; pr?: { x: number[]; y: number[] } }>;
  eda: Eda;
  interpretations: Record<string, string>;
  crisp_dm: { phase: string; summary: string }[];
  preprocessing: { split: { train_rows: number; test_rows: number; group_overlap: number; method: string }; leakage_controls: string[] };
  cross_reference_analysis: {
    models: Record<string, CrossReferenceModel>;
    cross_model: { cv_vs_holdout_rank_spearman?: number | null; cv_and_holdout_agree_on_best?: boolean };
  };
  score_config: { formula: string; weights: Record<string, number>; instability_penalty: number; overfitting_penalty: number };
  analysis: string;
  analysis_source: "llm" | "deterministic";
  selection_reason: string;
  dataset: { dataset: string; target: string; rows: number; primary_metric: string; imbalanced: boolean; source: string };
  warnings: string[];
  errors: { model: string; status: string; error: string | null }[];
  duration_s: number | null;
  engine: "langflow" | "direct";
  langflow: { flow: string; session_id: string } | null;
}

export interface NumericEda {
  feature: string;
  missing_pct: number | null;
  mean: number | null;
  median: number | null;
  skewness: number | null;
  outliers_iqr: number;
  mean_positive: number | null;
  mean_negative: number | null;
  association: { p_value_holm?: number | null; effect_size_rank_biserial: number | null; significant?: boolean };
  histogram: { edges: (number | null)[]; positive: number[]; negative: number[] };
}

export interface CategoricalEda {
  feature: string;
  n_levels: number;
  levels: { level: string; count: number; positive_rate: number | null }[];
  association: { p_value_holm?: number | null; cramers_v: number | null; significant?: boolean };
}

export interface Eda {
  overview: { rows: number; features: number; positive_rate: number; duplicated_rows: number; complete_rows_pct: number; target: string };
  missing: { feature: string; missing: number; missing_pct: number }[];
  numeric: NumericEda[];
  categorical: CategoricalEda[];
  correlations: { features: string[]; matrix: (number | null)[][]; top_pairs: { a: string; b: string; spearman: number }[] };
}

export interface PairwiseTest {
  model: string;
  nadeau_bengio: { mean_difference: number | null; p_value: number | null; p_value_holm?: number; significant?: boolean };
  delong: { auc_a: number | null; auc_b: number | null; p_value: number | null; p_value_holm?: number; significant?: boolean };
  mcnemar: { a_right_b_wrong: number; a_wrong_b_right: number; p_value: number | null; p_value_holm?: number; significant?: boolean };
}

export interface StatisticalTests {
  alpha: number;
  best_model: string;
  friedman: { statistic: number | null; p_value: number | null; significant: boolean | null };
  nemenyi: { critical_difference: number; mean_ranks: Record<string, number> };
  pairwise_vs_best: PairwiseTest[];
  bootstrap_ci_95: Record<string, { roc_auc: (number | null)[]; f1: (number | null)[] }>;
  summary: { models_significantly_worse_than_best: string[]; evidence: "strong" | "partial" | "weak" };
}

export interface CvModel {
  name: string;
  family: ModelFamily;
  status: string;
  cv_scores: number[];
  cv_mean: number | null;
  cv_std: number | null;
  tuning_stability: number | null;
}

export interface MLStatus {
  mode: "langflow" | "direct";
  flow: string;
  langflow_url: string;
  langflow_reachable: boolean | null;
}

export const runMLAnalysis = (payload: MLAnalysisRequest) =>
  api.post<MLAnalysisResult>("/ml/analysis", payload, { timeout: 15 * 60 * 1000 }).then((r) => r.data);
export const getMLStatus = () => api.get<MLStatus>("/ml/status").then((r) => r.data);

export async function downloadMLReport(runId: string, format: "pdf" | "xlsx"): Promise<void> {
  const response = await api.get(`/ml/analysis/${runId}/report`, { params: { format }, responseType: "blob" });
  const url = URL.createObjectURL(response.data as Blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `ml_analysis_${runId}.${format}`;
  link.click();
  URL.revokeObjectURL(url);
}
