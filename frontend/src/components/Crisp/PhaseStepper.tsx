import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { useTranslation } from "@/i18n/useTranslation";
import { extractErrorMessage } from "@/lib/errorMessage";
import { decidePhase } from "@/services/crispApi";
import type { CrispPhase, PhaseStatus } from "@/types/crisp";
import { PHASE_NAMES } from "@/types/crisp";

const STATUS_DOT: Record<PhaseStatus, string> = {
  NOT_STARTED: "bg-gray-400",
  IN_PROGRESS: "bg-status-yellow",
  IN_REVIEW: "bg-indigo-500",
  APPROVED: "bg-status-green",
  NEEDS_CORRECTION: "bg-status-red",
};

interface PhaseStepperProps {
  projectId: string;
  phases: CrispPhase[];
  activePhaseNumber: number;
  onSelectPhase: (phaseNumber: number) => void;
}

/** The six-phase visual required by the spec: 5 states per phase, approve /
 * reject / return-to-previous controls, and a mandatory reason when
 * returning to an earlier phase. */
export function PhaseStepper({ projectId, phases, activePhaseNumber, onSelectPhase }: PhaseStepperProps) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [decisionOpenFor, setDecisionOpenFor] = useState<string | null>(null);
  const [comments, setComments] = useState("");
  const [targetPhase, setTargetPhase] = useState<number | "">("");
  const [error, setError] = useState<string | null>(null);

  const decide = useMutation({
    mutationFn: (vars: { phaseId: string; action: string }) =>
      decidePhase(vars.phaseId, {
        action: vars.action,
        comments,
        target_phase_number: vars.action === "RETURN_TO_PREVIOUS" ? Number(targetPhase) : undefined,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["crisp-phases", projectId] });
      queryClient.invalidateQueries({ queryKey: ["crisp-project", projectId] });
      setDecisionOpenFor(null);
      setComments("");
      setTargetPhase("");
      setError(null);
    },
    onError: (err) => setError(extractErrorMessage(err, t("crisp.common.error"))),
  });

  const openPhase = phases.find((p) => decisionOpenFor === p.id);

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-gray-600 dark:text-gray-300">
        {t("crisp.phase.stepperTitle")}
      </h3>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
        {phases
          .slice()
          .sort((a, b) => a.phase_number - b.phase_number)
          .map((phase) => (
            <button
              key={phase.id}
              onClick={() => onSelectPhase(phase.phase_number)}
              className={`flex flex-col items-start gap-1 rounded-lg border p-3 text-left transition ${
                activePhaseNumber === phase.phase_number
                  ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-950/40"
                  : "border-gray-300 bg-gray-100 hover:border-gray-400 dark:border-gray-700 dark:bg-gray-900/60"
              }`}
            >
              <div className="flex items-center gap-1.5">
                <span className={`h-2 w-2 rounded-full ${STATUS_DOT[phase.status]}`} />
                <span className="text-xs font-semibold text-gray-500 dark:text-gray-400">
                  {t("crisp.projects.phase")} {phase.phase_number}
                </span>
              </div>
              <span className="text-sm font-medium text-gray-900 dark:text-white">{PHASE_NAMES[phase.phase_number]}</span>
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {t(`crisp.phase.status.${phase.status}`)}
              </span>
            </button>
          ))}
      </div>

      {phases.map(
        (phase) =>
          activePhaseNumber === phase.phase_number && (
            <div key={phase.id} className="flex flex-wrap items-center gap-2 rounded-lg bg-gray-100 p-3 dark:bg-gray-900/60">
              <button
                onClick={() => decide.mutate({ phaseId: phase.id, action: "APPROVE" })}
                disabled={decide.isPending}
                className="rounded bg-status-green px-3 py-1.5 text-xs font-semibold text-white hover:opacity-90 disabled:opacity-50"
              >
                {t("crisp.phase.approve")}
              </button>
              <button
                onClick={() => setDecisionOpenFor(decisionOpenFor === phase.id ? null : phase.id)}
                className="rounded bg-gray-300 px-3 py-1.5 text-xs font-semibold text-gray-800 hover:bg-gray-400 dark:bg-gray-700 dark:text-gray-100"
              >
                {t("crisp.phase.reject")} / {t("crisp.phase.return")}
              </button>
              {phase.last_rejection_reason && (
                <span className="text-xs text-status-red">
                  {t("crisp.phase.comments")}: {phase.last_rejection_reason}
                </span>
              )}
            </div>
          )
      )}

      {openPhase && (
        <div className="space-y-2 rounded-lg border border-gray-300 bg-white p-3 dark:border-gray-700 dark:bg-gray-950">
          <textarea
            value={comments}
            onChange={(e) => setComments(e.target.value)}
            placeholder={t("crisp.phase.comments")}
            className="w-full rounded bg-gray-200 px-2 py-1.5 text-sm text-gray-900 dark:bg-gray-800 dark:text-white"
            rows={2}
          />
          <div className="flex flex-wrap items-center gap-2">
            <select
              value={targetPhase}
              onChange={(e) => setTargetPhase(e.target.value ? Number(e.target.value) : "")}
              className="rounded bg-gray-200 px-2 py-1.5 text-sm text-gray-900 dark:bg-gray-800 dark:text-white"
            >
              <option value="">{t("crisp.phase.targetPhase")}</option>
              {Array.from({ length: openPhase.phase_number - 1 }, (_, i) => i + 1).map((n) => (
                <option key={n} value={n}>
                  {t("crisp.projects.phase")} {n} — {PHASE_NAMES[n]}
                </option>
              ))}
            </select>
            <button
              onClick={() => decide.mutate({ phaseId: openPhase.id, action: "REJECT" })}
              disabled={decide.isPending || !comments}
              className="rounded bg-status-red px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-50"
            >
              {t("crisp.phase.reject")}
            </button>
            <button
              onClick={() => decide.mutate({ phaseId: openPhase.id, action: "RETURN_TO_PREVIOUS" })}
              disabled={decide.isPending || !comments || !targetPhase}
              className="rounded bg-amber-600 px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-50"
              title={!comments || !targetPhase ? t("crisp.phase.commentsRequired") : undefined}
            >
              {t("crisp.phase.return")}
            </button>
          </div>
        </div>
      )}

      {error && (
        <p className="rounded-lg bg-red-100 p-2 text-xs text-red-700 ring-1 ring-red-300 dark:bg-red-950/40 dark:text-red-300 dark:ring-red-800/50">
          {error}
        </p>
      )}
    </div>
  );
}
