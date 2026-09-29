import { ExecutiveDashboard } from "@/components/Dashboard/ExecutiveDashboard";
import { useTwinWebSocket } from "@/hooks/useTwinWebSocket";

export function DashboardPage() {
  useTwinWebSocket();
  return <ExecutiveDashboard />;
}
