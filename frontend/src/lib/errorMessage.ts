/** Surfaces a FastAPI HTTPException's `detail` (already human-readable —
 * e.g. "No se puede aprobar la fase: faltan campos obligatorios (...)")
 * instead of a generic message, falling back only when none is present. */
export function extractErrorMessage(err: unknown, fallback: string): string {
  const anyErr = err as { response?: { status?: number; data?: { detail?: string } } };
  const detail = anyErr?.response?.data?.detail;
  return typeof detail === "string" ? detail : fallback;
}
