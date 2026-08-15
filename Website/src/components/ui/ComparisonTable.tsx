import { literatureEntries } from "@/data/literature";

export default function ComparisonTable() {
  return (
    <div className="overflow-x-auto my-12 border border-grid-line">
      <table className="w-full border-collapse text-left whitespace-nowrap">
        <thead>
          <tr className="bg-surface">
            <th className="font-semibold text-base text-ink py-4 px-6 border border-grid-line" style={{ fontFamily: "var(--font-display)" }}>Work</th>
            <th className="font-semibold text-base text-ink py-4 px-6 border border-grid-line" style={{ fontFamily: "var(--font-display)" }}>Year</th>
            <th className="font-semibold text-base text-ink py-4 px-6 border border-grid-line" style={{ fontFamily: "var(--font-display)" }}>Task</th>
            <th className="font-semibold text-base text-ink py-4 px-6 border border-grid-line" style={{ fontFamily: "var(--font-display)" }}>Grounding</th>
            <th className="font-semibold text-base text-ink py-4 px-6 border border-grid-line" style={{ fontFamily: "var(--font-display)" }}>Limitation</th>
          </tr>
        </thead>
        <tbody className="text-ink-muted text-sm md:text-base">
          {literatureEntries.map((entry, idx) => (
            <tr key={idx} className="hover:bg-surface/50 transition-colors">
              <td className="py-4 px-6 text-ink border border-grid-line font-medium">{entry.work}</td>
              <td className="py-4 px-6 border border-grid-line font-mono text-sm">{entry.year}</td>
              <td className="py-4 px-6 border border-grid-line">{entry.task}</td>
              <td className="py-4 px-6 border border-grid-line">{entry.grounding}</td>
              <td className="py-4 px-6 border border-grid-line">{entry.limitation}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
