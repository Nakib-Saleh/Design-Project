import { ReactNode } from "react";

interface Props {
  children: ReactNode;
  label?: string;
}

export default function InProgressPanel({ children, label = "Results pending — data collection in progress." }: Props) {
  return (
    <div className="relative border-2 border-dashed border-grid-line p-6 my-8 overflow-hidden group">
      {/* Subtle diagonal hatch background fill */}
      <div
        className="absolute inset-0 opacity-[0.03] pointer-events-none"
        style={{
          backgroundImage: "repeating-linear-gradient(45deg, var(--color-ink) 0, var(--color-ink) 1px, transparent 1px, transparent 12px)",
        }}
      />
      
      {/* Label tag */}
      <div className="absolute top-0 right-0 bg-surface border-b border-l border-grid-line px-3 py-1 text-xs text-ink-muted font-medium font-sans z-10">
        {label}
      </div>

      {/* Content wrapper */}
      <div className="relative z-0 opacity-80 group-hover:opacity-100 transition-opacity duration-300">
        {children}
      </div>
    </div>
  );
}
