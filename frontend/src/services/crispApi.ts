// Thin typed wrappers around the shared `api` axios client (services/api.ts)
// for every backend/app/api/v1/{crisp,mining}.py endpoint the frontend uses.
import { api } from "@/services/api";
import type {
  Alert,
  CrispAuditLog,
  CrispDataset,
  CrispPhase,
  CrispPhaseActivity,
  CrispProject,
  DataDriftRecord,
  DataSource,
  DatasetVersion,
  DigitalTwin,
  EngineeredFeature,
  LoadingUnit,
  MaintenanceOrder,
  MlExperiment,
  MlModel,
  ModelDeployment,
  ModelEvaluation,
  ModelMetric,
  ModelMonitoring,
  ModelPrediction,
  ExperimentComparisonRow,
} from "@/types/crisp";

// ---- Mining domain -------------------------------------------------------
export const listLoadingUnits = () => api.get<LoadingUnit[]>("/crisp/mining/loading-units").then((r) => r.data);
export const getDigitalTwin = (unitId: string) =>
  api.get<DigitalTwin>(`/crisp/mining/loading-units/${unitId}/digital-twin`).then((r) => r.data);
export const listAlerts = (params?: { loading_unit_id?: string; status_filter?: string }) =>
  api.get<Alert[]>("/crisp/mining/alerts", { params }).then((r) => r.data);
export const acknowledgeAlert = (alertId: string, comments: string) =>
  api.post<Alert>(`/crisp/mining/alerts/${alertId}/acknowledge`, { comments }).then((r) => r.data);
export const listMaintenanceOrders = (loading_unit_id?: string) =>
  api.get<MaintenanceOrder[]>("/crisp/mining/maintenance-orders", { params: { loading_unit_id } }).then((r) => r.data);
export const createMaintenanceOrder = (payload: Partial<MaintenanceOrder> & { loading_unit_id: string; order_type: string }) =>
  api.post<MaintenanceOrder>("/crisp/mining/maintenance-orders", payload).then((r) => r.data);

// ---- Projects -------------------------------------------------------------
export const listProjects = () => api.get<CrispProject[]>("/crisp/projects").then((r) => r.data);
export const getProject = (id: string) => api.get<CrispProject>(`/crisp/projects/${id}`).then((r) => r.data);
export const createProject = (payload: {
  name: string;
  description?: string;
  start_date: string;
  estimated_end_date?: string;
  loading_unit_ids: string[];
}) => api.post<CrispProject>("/crisp/projects", payload).then((r) => r.data);
export const updateProject = (id: string, payload: Record<string, unknown>) =>
  api.put<CrispProject>(`/crisp/projects/${id}`, payload).then((r) => r.data);
export const deleteProject = (id: string) => api.delete(`/crisp/projects/${id}`);

// ---- Phases -----------------------------------------------------------------
export const listPhases = (projectId: string) => api.get<CrispPhase[]>(`/crisp/projects/${projectId}/phases`).then((r) => r.data);
export const updatePhaseData = (phaseId: string, data: Record<string, unknown>) =>
  api.put<CrispPhase>(`/crisp/phases/${phaseId}`, { data }).then((r) => r.data);
export const listPhaseActivities = (phaseId: string) =>
  api.get<CrispPhaseActivity[]>(`/crisp/phases/${phaseId}/activities`).then((r) => r.data);
export const logPhaseActivity = (phaseId: string, payload: { description: string; result?: string; observations?: string }) =>
  api.post<CrispPhaseActivity>(`/crisp/phases/${phaseId}/activities`, payload).then((r) => r.data);
export const decidePhase = (phaseId: string, payload: { action: string; comments?: string; target_phase_number?: number }) =>
  api.post<CrispPhase>(`/crisp/phases/${phaseId}/approve`, payload).then((r) => r.data);

// ---- Data understanding -----------------------------------------------------
export const listDataSources = (projectId: string) =>
  api.get<DataSource[]>(`/crisp/projects/${projectId}/datasources`).then((r) => r.data);
export const createDataSource = (payload: Partial<DataSource> & { project_id: string; name: string; source_type: string }) =>
  api.post<DataSource>("/crisp/datasources", payload).then((r) => r.data);
export const computeEda = (sourceId: string) =>
  api.post<DataSource>(`/crisp/datasources/${sourceId}/compute-eda`).then((r) => r.data);

// ---- Data preparation ---------------------------------------------------------
export const listDatasets = (projectId: string) => api.get<CrispDataset[]>(`/crisp/projects/${projectId}/datasets`).then((r) => r.data);
export const createDataset = (payload: { project_id: string; name: string; description?: string }) =>
  api.post<CrispDataset>("/crisp/datasets", payload).then((r) => r.data);
export const listDatasetVersions = (datasetId: string) =>
  api.get<DatasetVersion[]>(`/crisp/datasets/${datasetId}/versions`).then((r) => r.data);
export const createDatasetVersion = (
  datasetId: string,
  payload: { dataset_id: string; name: string; operations_applied: { step_type: string; parameters?: Record<string, unknown> }[]; split_config?: Record<string, unknown> }
) => api.post<DatasetVersion>(`/crisp/datasets/${datasetId}/versions`, payload).then((r) => r.data);
export const listEngineeredFeatures = (versionId: string) =>
  api.get<EngineeredFeature[]>(`/crisp/dataset-versions/${versionId}/features`).then((r) => r.data);
export const createEngineeredFeature = (
  versionId: string,
  payload: { dataset_version_id: string; feature_name: string; definition?: string; source_variables?: string[] }
) => api.post<EngineeredFeature>(`/crisp/dataset-versions/${versionId}/features`, payload).then((r) => r.data);

// ---- Modeling ---------------------------------------------------------------
export const listExperiments = (projectId: string) =>
  api.get<MlExperiment[]>(`/crisp/projects/${projectId}/experiments`).then((r) => r.data);
export const createExperiment = (payload: Omit<MlExperiment, "id" | "trained_at" | "duration_seconds" | "status" | "model_artifact_uri" | "observations" | "responsible_id">) =>
  api.post<MlExperiment>("/crisp/experiments", payload).then((r) => r.data);
export const trainExperiment = (experimentId: string) =>
  api.post<MlExperiment>(`/crisp/experiments/${experimentId}/train`).then((r) => r.data);
export const getExperimentMetrics = (experimentId: string) =>
  api.get<ModelMetric[]>(`/crisp/experiments/${experimentId}/metrics`).then((r) => r.data);
export const compareExperiments = (projectId: string) =>
  api.get<ExperimentComparisonRow[]>(`/crisp/projects/${projectId}/experiments/compare`).then((r) => r.data);
export const listModelsForExperiment = (experimentId: string) =>
  api.get<MlModel[]>(`/crisp/experiments/${experimentId}/models`).then((r) => r.data);
export const publishModel = (modelId: string) => api.post<MlModel>(`/crisp/models/${modelId}/publish`).then((r) => r.data);

// ---- Evaluation ---------------------------------------------------------------
export const evaluateModel = (modelId: string, payload: Record<string, unknown>) =>
  api.post<ModelEvaluation>(`/crisp/models/${modelId}/evaluate`, { model_id: modelId, ...payload }).then((r) => r.data);
export const listEvaluations = (modelId: string) =>
  api.get<ModelEvaluation[]>(`/crisp/models/${modelId}/evaluations`).then((r) => r.data);

// ---- Deployment & monitoring ----------------------------------------------------
export const deployModel = (modelId: string, payload: Record<string, unknown>) =>
  api.post<ModelDeployment>(`/crisp/models/${modelId}/deploy`, { model_id: modelId, ...payload }).then((r) => r.data);
export const listDeployments = (projectId: string) =>
  api.get<ModelDeployment[]>(`/crisp/projects/${projectId}/deployments`).then((r) => r.data);
export const createPrediction = (deployment_id: string, loading_unit_id: string) =>
  api.post<ModelPrediction>("/crisp/predictions", { deployment_id, loading_unit_id }).then((r) => r.data);
export const listMonitoring = (deploymentId: string) =>
  api.get<ModelMonitoring[]>(`/crisp/deployments/${deploymentId}/monitoring`).then((r) => r.data);
export const refreshMonitoring = (deploymentId: string) =>
  api.post<ModelMonitoring>(`/crisp/deployments/${deploymentId}/monitoring/refresh`).then((r) => r.data);
export const detectDataDrift = (deploymentId: string) =>
  api.post<DataDriftRecord>(`/crisp/deployments/${deploymentId}/drift/data`).then((r) => r.data);
export const detectModelDrift = (deploymentId: string) =>
  api.post<DataDriftRecord>(`/crisp/deployments/${deploymentId}/drift/model`).then((r) => r.data);
export const listDrift = (deploymentId: string) => api.get<DataDriftRecord[]>(`/crisp/deployments/${deploymentId}/drift`).then((r) => r.data);

// ---- Audit & reports -----------------------------------------------------------
export const getAuditLog = (projectId: string) => api.get<CrispAuditLog[]>(`/crisp/projects/${projectId}/audit`).then((r) => r.data);
export const finalReportUrl = (projectId: string, fmt: "pdf" | "xlsx") => `/api/v1/crisp/projects/${projectId}/report?fmt=${fmt}`;
