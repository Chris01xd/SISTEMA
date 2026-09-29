import type { StatusColor } from "@/types";

export const STATUS_HEX: Record<StatusColor, string> = {
  GREEN: "#16a34a",
  YELLOW: "#eab308",
  RED: "#dc2626",
};

export function bedStatusToColor(status: string): string {
  switch (status) {
    case "AVAILABLE":
      return STATUS_HEX.GREEN;
    case "CLEANING":
      return STATUS_HEX.YELLOW;
    case "OCCUPIED":
    case "OUT_OF_SERVICE":
      return STATUS_HEX.RED;
    default:
      return "#6b7280";
  }
}
