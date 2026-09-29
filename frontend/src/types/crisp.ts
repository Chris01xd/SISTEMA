// Mirrors backend/app/schemas/{mining,crisp_*}.py — kept in this one file
// (unlike types/index.ts's hospital types) since the CRISP-DM domain is
// self-contained and large; splitting it further isn't worth the import
// overhead for how these types are actually consumed (whole panels at a time).

export type PhaseStatus = "NOT_STARTED" | "IN_PROGRESS" | "IN_REVIEW" | "APPROVED" | "NEEDS_CORRECTION";
export type PhaseApprovalAction = "APPROVE" | "REJECT" | "RETURN_TO_PREVIOUS";
export type ProjectOverallStatus = "NOT_STARTED" | "IN_PROGRESS" | "ON_HOLD" | "COMPLETED" | "CANCELLED";

export interface LoadingUnit {
  id: string;
  code: string;
  unit_type: "EXCAVATOR" | "WHEEL_LOADER" | "HAUL_TRUCK" | "SHOVEL";
  brand: string;
  model: string;
  manufacture_year: number | null;
  capacity_tons: number;
  fleet: string;
  current_location: string;
  operational_status: "OPERATIONAL" | "MAINTENANCE" | "OUT_OF_SERVICE";
}

export interface DigitalTwin {
  id: string;
  loading_unit_id: string;
  model_reference: string;
  calibration_parameters: Record<string, unknown>;
  current_state: "NORMAL" | "WARNING" | "CRITICAL" | "OUT_OF_SERVICE";
  risk_level: number;
  last_prediction: Record<string, unknown>;
}

export interface Alert {
  id: string;
  loading_unit_id: string;
  severity: "NORMAL" | "WARNING" | "CRITICAL" | "OUT_OF_SERVICE";
  source: "MODEL_PREDICTION" | "THRESHOLD_RULE" | "MANUAL";
  message: string;
  status: "OPEN" | "ACKNOWLEDGED" | "RESOLVED" | "DISMISSED";
  triggered_at: string;
  acknowledged_by_id: string | null;
  acknowledged_at: string | null;
}

export interface MaintenanceOrder {
  id: string;
  loading_unit_id: string;
  order_type: "PREVENTIVE" | "PREDICTIVE" | "CORRECTIVE";
  status: "PENDING" | "SCHEDULED" | "IN_PROGRESS" | "COMPLETED" | "CANCELLED";
  scheduled_date: string | null;
  executed_date: string | null;
  observations: string;
  cost: number | null;
}

export interface KpiBaselineTarget {
  indicator: string;
  unit?: string;
  baseline_value?: number | null;
  baseline_date?: string | null;
  target_value?: number | null;
  actual_value?: number | null;
  actual_measured_at?: string | null;
}

export interface BusinessUnderstandingData {
  business_problem: string;
  affected_loading_unit_ids: string[];
  maintenance_processes_involved: string[];
  general_objective: string;
  specific_objectives: string[];
  interested_users: string[];
  constraints: string[];
  initial_risks: string[];
  available_resources: string[];
  kpis: KpiBaselineTarget[];
  expected_goals: string;
  business_success_criteria: string[];
}

export interface CrispProject {
  id: string;
  name: string;
  description: string;
  digital_twin_id: string | null;
  responsible_id: string | null;
  start_date: string;
  estimated_end_date: string | null;
  actual_end_date: string | null;
  current_phase_number: number;
  overall_status: ProjectOverallStatus;
  progress_percent: number;
  status: string;
}

export interface CrispPhase {
  id: string;
  project_id: string;
  phase_number: number;
  status: PhaseStatus;
  phase_data: Record<string, unknown>;
  started_at: string | null;
  completed_at: string | null;
  approved_by_id: string | null;
  approved_at: string | null;
  last_rejection_reason: string;
}

export interface CrispPhaseActivity {
  id: string;
  phase_id: string;
  description: string;
  result: string;
  observations: string;
  performed_by_id: string | null;
  performed_at: string;
}

export interface DataSource {
  id: string;
  project_id: string;
  name: string;
  source_type: "IOT_SENSOR" | "SCADA" | "ERP" | "MAINTENANCE_ORDERS" | "FAILURE_HISTORY" | "MANUFACTURER_DATA" | "MANUAL_ENTRY";
  loading_unit_id: string | null;
  measured_variable: string;
  unit_of_measure: string;
  sampling_frequency: string;
  period_start: string | null;
  period_end: string | null;
  observations: string;
  record_count: number;
  missing_values_pct: number;
  duplicate_records: number;
  outlier_count: number;
  quality_level: "HIGH" | "MEDIUM" | "LOW" | "UNASSESSED";
  eda_report: Record<string, any>;
}

export interface CrispDataset {
  id: string;
  project_id: string;
  name: string;
  description: string;
}

export interface DatasetVersion {
  id: string;
  dataset_id: string;
  name: string;
  responsible_id: string | null;
  source_data_source_id: string | null;
  operations_applied: { step_type: string; parameters?: Record<string, unknown> }[];
  record_count: number;
  variables: string[];
  storage_uri: string;
  split_config: Record<string, [string, string]>;
  validation_status: "DRAFT" | "VALID" | "INVALID";
}

export interface EngineeredFeature {
  id: string;
  dataset_version_id: string;
  feature_name: string;
  definition: string;
  source_variables: string[];
  created_by_id: string | null;
}

export type Algorithm = "RANDOM_FOREST" | "XGBOOST" | "LSTM" | "ISOLATION_FOREST" | "REGRESSION";
export type ModelObjective =
  | "FAILURE_CLASSIFICATION"
  | "ANOMALY_DETECTION"
  | "FAILURE_PREDICTION"
  | "REMAINING_USEFUL_LIFE"
  | "OPERATIONAL_VARIABLE_FORECAST";

export interface MlExperiment {
  id: string;
  project_id: string;
  dataset_version_id: string;
  objective: ModelObjective;
  algorithm: Algorithm;
  input_variables: string[];
  target_variable: string;
  hyperparameters: Record<string, unknown>;
  validation_method: string;
  responsible_id: string | null;
  trained_at: string | null;
  duration_seconds: number | null;
  status: "DRAFT" | "QUEUED" | "TRAINING" | "COMPLETED" | "FAILED";
  model_artifact_uri: string;
  observations: string;
}

export interface ModelMetric {
  metric_name: string;
  metric_value: number;
  dataset_split: "TRAIN" | "VALIDATION" | "TEST";
}

export interface MlModel {
  id: string;
  experiment_id: string;
  name: string;
  version: string;
  algorithm: Algorithm;
  artifact_uri: string;
  is_candidate: boolean;
  is_published: boolean;
}

export interface ExperimentComparisonRow {
  experiment: MlExperiment;
  metrics: ModelMetric[];
}

export type EvaluationDecision = "APPROVED" | "REJECTED" | "RETURN_TO_DATA_UNDERSTANDING" | "RETURN_TO_DATA_PREPARATION";

export interface ModelEvaluation {
  id: string;
  model_id: string;
  evaluator_id: string;
  operational_metrics: Record<string, number | null>;
  business_metrics_estimate: Record<string, number | null>;
  specialist_opinion: string;
  decision: EvaluationDecision | null;
  justification: string;
  evaluated_at: string | null;
}

export interface ModelDeployment {
  id: string;
  model_id: string;
  project_id: string;
  version: string;
  deployed_at: string;
  api_endpoint: string;
  service_status: "ACTIVE" | "PAUSED" | "RETIRED";
  alert_thresholds: { warning?: number; critical?: number };
  inference_frequency: string;
  responsible_id: string | null;
  rollback_plan: string;
  next_retraining_date: string | null;
}

export interface ModelPrediction {
  id: string;
  deployment_id: string;
  loading_unit_id: string;
  predicted_at: string;
  equipment_state: "NORMAL" | "WARNING" | "CRITICAL" | "OUT_OF_SERVICE";
  risk_level: number;
  anomaly_type: string | null;
  failure_probability: number | null;
  affected_component: string | null;
  estimated_rul_hours: number | null;
  severity: string;
  maintenance_recommendation: string;
  linked_alert_id: string | null;
}

export interface ModelMonitoring {
  id: string;
  deployment_id: string;
  recorded_at: string;
  current_accuracy: number | null;
  inference_latency_ms: number | null;
  prediction_count: number;
  alerts_generated_count: number;
  false_alarm_count: number;
  needs_retraining: boolean;
  notes: string;
}

export interface DataDriftRecord {
  id: string;
  deployment_id: string;
  detected_at: string;
  drift_type: "DATA_DRIFT" | "MODEL_DRIFT";
  variable_name: string | null;
  drift_score: number;
  method: string;
  threshold_used: number;
  is_significant: boolean;
}

export interface CrispAuditLog {
  id: string;
  user_id: string | null;
  action: string;
  occurred_at: string;
  project_id: string | null;
  phase_id: string | null;
  entity_type: string;
  entity_id: string;
  old_value: Record<string, unknown>;
  new_value: Record<string, unknown>;
  ip_address: string | null;
}

export const PHASE_NAMES: Record<number, string> = {
  1: "Comprensión del negocio",
  2: "Comprensión de los datos",
  3: "Preparación de los datos",
  4: "Modelado",
  5: "Evaluación",
  6: "Despliegue",
};
