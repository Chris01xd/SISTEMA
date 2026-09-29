import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { useTranslation } from "@/i18n/useTranslation";
import { extractErrorMessage } from "@/lib/errorMessage";
import {
  createDataset,
  createDatasetVersion,
  createEngineeredFeature,
  listDatasets,
  listDatasetVersions,
  listEngineeredFeatures,
} from "@/services/crispApi";

const OPERATIONS = [
  "CLEANING",
  "DEDUPLICATION",
  "MISSING_VALUE_IMPUTATION",
  "OUTLIER_TREATMENT",
  "TEMPORAL_SYNCHRONIZATION",
  "SOURCE_INTEGRATION",
  "NORMALIZATION",
  "ENCODING",
  "FAILURE_LABELING",
  "CLASS_BALANCING",
  "TIME_WINDOWING",
];

const STATUS_COLOR: Record<string, string> = { VALID: "text-status-green", DRAFT: "text-gray-500", INVALID: "text-status-red" };

interface Props {
  projectId: string;
  canEdit: boolean;
}

export function DataPreparationPanel({ projectId, canEdit }: Props) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [datasetName, setDatasetName] = useState("");
  const [versionDatasetId, setVersionDatasetId] = useState("");
  const [versionName, setVersionName] = useState("v1");
  const [selectedOps, setSelectedOps] = useState<string[]>(["CLEANING", "DEDUPLICATION", "MISSING_VALUE_IMPUTATION"]);
  const [expandedVersion, setExpandedVersion] = useState<string | null>(null);
  const [featureName, setFeatureName] = useState("");
  const [featureDef, setFeatureDef] = useState("");
  const [error, setError] = useState<string | null>(null);

  const datasets = useQuery({ queryKey: ["crisp-datasets", projectId], queryFn: () => listDatasets(projectId) });

  const createDatasetMut = useMutation({
    mutationFn: () => createDataset({ project_id: projectId, name: datasetName }),
    onSuccess: () => {
      setDatasetName("");
      queryClient.invalidateQueries({ queryKey: ["crisp-datasets", projectId] });
    },
    onError: (err) => setError(extractErrorMessage(err, t("crisp.common.error"))),
  });

  const createVersionMut = useMutation({
    mutationFn: () =>
      createDatasetVersion(versionDatasetId, {
        dataset_id: versionDatasetId,
        name: versionName,
        operations_applied: selectedOps.map((step_type) => ({ step_type })),
      }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["crisp-dataset-versions", versionDatasetId] }),
    onError: (err) => setError(extractErrorMessage(err, t("crisp.common.error"))),
  });

  const createFeatureMut = useMutation({
    mutationFn: (versionId: string) => createEngineeredFeature(versionId, { dataset_version_id: versionId, feature_name: featureName, definition: featureDef }),
    onSuccess: (_, versionId) => {
      setFeatureName("");
      setFeatureDef("");
      queryClient.invalidateQueries({ queryKey: ["crisp-features", versionId] });
    },
    onError: (err) => setError(extractErrorMessage(err, t("crisp.common.error"))),
  });

  const fieldClass = "rounded bg-gray-200 px-2 py-1.5 text-sm text-gray-900 dark:bg-gray-800 dark:text-white";

  return (
    <div className="space-y-5">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{t("crisp.prep.title")}</h3>

      {canEdit && (
        <div className="flex flex-wrap items-end gap-2 rounded-lg border border-gray-300 bg-gray-100 p-3 dark:border-gray-700 dark:bg-gray-900/60">
          <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
            {t("crisp.prep.datasets")}
            <input value={datasetName} onChange={(e) => setDatasetName(e.target.value)} className={fieldClass} />
          </label>
          <button
            onClick={() => createDatasetMut.mutate()}
            disabled={!datasetName || createDatasetMut.isPending}
            className="rounded bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-50"
          >
            {t("crisp.prep.addDataset")}
          </button>
        </div>
      )}

      {error && (
        <p className="rounded-lg bg-red-100 p-2 text-xs text-red-700 ring-1 ring-red-300 dark:bg-red-950/40 dark:text-red-300 dark:ring-red-800/50">
          {error}
        </p>
      )}

      <div className="space-y-4">
        {datasets.data?.map((dataset) => (
          <DatasetVersionsBlock
            key={dataset.id}
            datasetId={dataset.id}
            datasetName={dataset.name}
            canEdit={canEdit}
            expandedVersion={expandedVersion}
            setExpandedVersion={setExpandedVersion}
            featureName={featureName}
            setFeatureName={setFeatureName}
            featureDef={featureDef}
            setFeatureDef={setFeatureDef}
            createFeatureMut={createFeatureMut}
          />
        ))}
      </div>

      {canEdit && datasets.data && datasets.data.length > 0 && (
        <div className="space-y-2 rounded-lg border border-gray-300 bg-gray-100 p-3 dark:border-gray-700 dark:bg-gray-900/60">
          <div className="flex flex-wrap items-end gap-2">
            <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
              {t("crisp.prep.datasets")}
              <select value={versionDatasetId} onChange={(e) => setVersionDatasetId(e.target.value)} className={fieldClass}>
                <option value="">{t("crisp.common.none")}</option>
                {datasets.data.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col text-xs text-gray-600 dark:text-gray-300">
              {t("crisp.prep.addVersion")}
              <input value={versionName} onChange={(e) => setVersionName(e.target.value)} className={fieldClass} />
            </label>
          </div>
          <div>
            <p className="mb-1 text-xs font-semibold text-gray-600 dark:text-gray-300">{t("crisp.prep.operations")}</p>
            <div className="flex flex-wrap gap-2">
              {OPERATIONS.map((op) => (
                <label key={op} className="flex items-center gap-1 text-xs text-gray-600 dark:text-gray-300">
                  <input
                    type="checkbox"
                    checked={selectedOps.includes(op)}
                    onChange={(e) =>
                      setSelectedOps((prev) => (e.target.checked ? [...prev, op] : prev.filter((o) => o !== op)))
                    }
                  />
                  {op}
                </label>
              ))}
            </div>
          </div>
          <button
            onClick={() => createVersionMut.mutate()}
            disabled={!versionDatasetId || createVersionMut.isPending}
            className="rounded bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-50"
          >
            {createVersionMut.isPending ? "..." : t("crisp.prep.addVersion")}
          </button>
        </div>
      )}
    </div>
  );
}

function DatasetVersionsBlock({
  datasetId,
  datasetName,
  canEdit,
  expandedVersion,
  setExpandedVersion,
  featureName,
  setFeatureName,
  featureDef,
  setFeatureDef,
  createFeatureMut,
}: {
  datasetId: string;
  datasetName: string;
  canEdit: boolean;
  expandedVersion: string | null;
  setExpandedVersion: (id: string | null) => void;
  featureName: string;
  setFeatureName: (v: string) => void;
  featureDef: string;
  setFeatureDef: (v: string) => void;
  createFeatureMut: ReturnType<typeof useMutation<unknown, unknown, string>>;
}) {
  const { t } = useTranslation();
  const versions = useQuery({ queryKey: ["crisp-dataset-versions", datasetId], queryFn: () => listDatasetVersions(datasetId) });

  return (
    <div className="rounded-lg border border-gray-300 bg-gray-100 p-3 dark:border-gray-700 dark:bg-gray-900/60">
      <p className="text-sm font-semibold text-gray-900 dark:text-white">{datasetName}</p>
      <div className="mt-2 space-y-2">
        {versions.data?.map((v) => (
          <div key={v.id} className="rounded bg-gray-200 p-2 text-xs dark:bg-gray-800">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="font-semibold text-gray-800 dark:text-gray-100">{v.name}</span>
              <span className={STATUS_COLOR[v.validation_status]}>{v.validation_status}</span>
            </div>
            <p className="mt-1 text-gray-600 dark:text-gray-300">
              {t("crisp.prep.records")}: {v.record_count} · {t("crisp.prep.variables")}: {v.variables.length}
            </p>
            <p className="mt-1 break-words text-gray-500 dark:text-gray-400">{v.variables.join(", ")}</p>
            <button onClick={() => setExpandedVersion(expandedVersion === v.id ? null : v.id)} className="mt-1 text-indigo-600 dark:text-indigo-400">
              {t("crisp.prep.features")}
            </button>
            {expandedVersion === v.id && (
              <FeaturesBlock
                versionId={v.id}
                canEdit={canEdit}
                featureName={featureName}
                setFeatureName={setFeatureName}
                featureDef={featureDef}
                setFeatureDef={setFeatureDef}
                createFeatureMut={createFeatureMut}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function FeaturesBlock({
  versionId,
  canEdit,
  featureName,
  setFeatureName,
  featureDef,
  setFeatureDef,
  createFeatureMut,
}: {
  versionId: string;
  canEdit: boolean;
  featureName: string;
  setFeatureName: (v: string) => void;
  featureDef: string;
  setFeatureDef: (v: string) => void;
  createFeatureMut: ReturnType<typeof useMutation<unknown, unknown, string>>;
}) {
  const { t } = useTranslation();
  const features = useQuery({ queryKey: ["crisp-features", versionId], queryFn: () => listEngineeredFeatures(versionId) });
  const fieldClass = "rounded bg-gray-300 px-2 py-1 text-xs text-gray-900 dark:bg-gray-700 dark:text-white";

  return (
    <div className="mt-2 space-y-1 border-t border-gray-300 pt-2 dark:border-gray-700">
      {features.data?.map((f) => (
        <p key={f.id} className="text-gray-600 dark:text-gray-300">
          <span className="font-semibold">{f.feature_name}</span>: {f.definition}
        </p>
      ))}
      {canEdit && (
        <div className="flex flex-wrap gap-1">
          <input placeholder={t("crisp.prep.featureName")} value={featureName} onChange={(e) => setFeatureName(e.target.value)} className={fieldClass} />
          <input placeholder={t("crisp.prep.featureDefinition")} value={featureDef} onChange={(e) => setFeatureDef(e.target.value)} className={fieldClass} />
          <button
            onClick={() => createFeatureMut.mutate(versionId)}
            disabled={!featureName || createFeatureMut.isPending}
            className="rounded bg-indigo-600 px-2 py-1 text-xs font-semibold text-white disabled:opacity-50"
          >
            {t("crisp.prep.addFeature")}
          </button>
        </div>
      )}
    </div>
  );
}
