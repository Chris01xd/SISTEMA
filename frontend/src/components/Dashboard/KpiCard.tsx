interface KpiCardProps {
  label: string;
  value: string | number;
  accent?: "green" | "yellow" | "red" | "neutral";
}

const ACCENT_CLASSES: Record<string, string> = {
  green: "border-status-green/50 text-status-green",
  yellow: "border-status-yellow/50 text-status-yellow",
  red: "border-status-red/50 text-status-red",
  neutral: "border-gray-300 text-gray-900 dark:border-gray-700 dark:text-gray-100",
};

export function KpiCard({ label, value, accent = "neutral" }: KpiCardProps) {
  return (
    <div className={`rounded-xl border bg-gray-100 p-4 dark:bg-gray-900/60 ${ACCENT_CLASSES[accent]}`}>
      <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">{label}</p>
      <p className="mt-1 text-2xl font-bold">{value}</p>
    </div>
  );
}
