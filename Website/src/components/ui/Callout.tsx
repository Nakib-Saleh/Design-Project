import { ReactNode } from "react";

export default function Callout({ children }: { children: ReactNode }) {
  return (
    <div className="bg-surface border-l-[3px] border-accent-primary p-8 my-10 text-xl md:text-2xl font-display font-medium text-ink">
      {children}
    </div>
  );
}
