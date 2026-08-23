import { Gauge, Ruler, Move, Camera, Video, Target, LucideIcon } from "lucide-react";
import { clsx } from "clsx";

interface Props {
  label: string;
  className?: string;
}

const iconMap: Record<string, LucideIcon> = {
  Velocity: Gauge,
  "Scale Recovery": Ruler,
  "Object Size": Ruler,
  Tracking: Move,
  Vision: Camera,
  Video: Video,
  Prediction: Target,
};

export default function TaskTag({ label, className }: Props) {
  const Icon = iconMap[label] || Target;

  return (
    <div
      className={clsx(
        "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full",
        "bg-surface border border-grid-line/60 text-ink-muted text-xs font-medium font-sans",
        className
      )}
    >
      <Icon className="w-3.5 h-3.5" />
      <span>{label}</span>
    </div>
  );
}
