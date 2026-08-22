import { clsx } from "clsx";

interface Props {
  value: string | number;
  unit?: string;
  tone: "primary" | "secondary";
}

export default function DataValue({ value, unit, tone }: Props) {
  return (
    <span
      className={clsx(
        "font-mono font-medium whitespace-nowrap",
        tone === "primary" ? "text-accent-primary" : "text-accent-secondary"
      )}
    >
      {value}
      {unit && <span className="ml-1 text-sm">{unit}</span>}
    </span>
  );
}
