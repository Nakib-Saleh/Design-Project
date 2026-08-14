"use client";

export default function Footer() {
  const scrollToTop = () => window.scrollTo({ top: 0, behavior: "smooth" });

  return (
    <footer className="bg-surface border-t border-grid-line py-12 mt-24">
      <div className="max-w-[1024px] mx-auto px-6 lg:px-12 flex flex-col sm:flex-row justify-between items-start sm:items-center text-ink-muted text-sm gap-6">
        <div>
          <p className="font-semibold text-ink mb-1" style={{ fontFamily: "var(--font-display)" }}>Nakib</p>
          <p>CSE, IUT · Thesis Pre-Defence</p>
          <p>Supervisor: [Supervisor Name]</p>
        </div>
        <button
          onClick={scrollToTop}
          className="hover:text-ink transition-colors px-4 py-2 border border-grid-line rounded-[6px] focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-primary"
        >
          Back to top ↑
        </button>
      </div>
    </footer>
  );
}
