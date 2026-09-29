// Typed wrappers for backend/app/api/v1/datasets.py.
import { api } from "@/services/api";

export interface CatalogEntry {
  key: string;
  name: string;
  version: string;
  description: string;
  license: string;
  homepage: string;
  citation: string;
  ingested: boolean;
  dataset_id: string | null;
}

export interface IngestedDataset {
  id: string;
  name: string;
  source: string;
  version: string;
  schema_hash: string;
  ingested_at: string;
  row_count: number | null;
}

export interface DatasetSummary {
  rows: number;
  patients?: number;
  period?: { start: string; end: string };
  age?: { mean: number; median: number; min: number; max: number };
  sex?: Record<string, number>;
  los_hours?: { mean: number; median: number; p90: number } | null;
  icu_rate_pct?: number;
  mortality_rate_pct?: number;
  severity_distribution?: Record<string, number>;
  severity_coverage_pct?: number;
  admissions_by_hour?: number[];
  top_diagnoses?: { diagnosis: string; count: number }[];
  missing_pct?: Record<string, number>;
}

export interface DatasetPreview {
  columns: string[];
  rows: Record<string, unknown>[];
  total_rows: number;
}

export const listCatalog = () => api.get<CatalogEntry[]>("/datasets/catalog").then((r) => r.data);
export const ingestDataset = (key: string) => api.post<IngestedDataset>("/datasets/ingest", { key }).then((r) => r.data);
export const getDatasetSummary = (id: string) => api.get<DatasetSummary>(`/datasets/${id}/summary`).then((r) => r.data);
export const getDatasetPreview = (id: string, limit = 15) =>
  api.get<DatasetPreview>(`/datasets/${id}/preview`, { params: { limit } }).then((r) => r.data);
