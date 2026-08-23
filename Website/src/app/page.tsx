import Hero from "@/components/sections/Hero";
import OverviewCard from "@/components/ui/OverviewCard";
import Link from "next/link";

export default function Home() {
  return (
    <div className="max-w-[1024px] mx-auto px-6 lg:px-12 pb-32">
      <Hero />

      <div className="py-20 border-t border-grid-line">
        <h2
          className="text-3xl lg:text-4xl leading-tight text-ink font-semibold max-w-[28ch]"
          style={{ fontFamily: "var(--font-display)" }}
        >
          This project tests whether LLMs can go beyond describing motion to
          actually <span className="underline decoration-accent-brand decoration-4 underline-offset-4">measuring it</span>.
        </h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[3fr_2fr_2fr_3fr] gap-4 mb-24">
        <OverviewCard
          title="Literature Review"
          description="Where existing benchmarks stop at qualitative description."
          href="/literature-review"
          tagLabel="Vision"
        />
        <OverviewCard
          title="Methodology"
          description="A three-stage pipeline: prior extraction, pixel-to-world scaling, kinematic inference."
          href="/methodology"
          tagLabel="Velocity"
        />
        <OverviewCard
          title="Findings"
          description="Where the model's predictions hold up, and where they collapse."
          href="/findings"
          tagLabel="Prediction"
        />
        <OverviewCard
          title="Demo"
          description="An interactive sample walkthrough. Coming soon."
          href="/demo"
          tagLabel="Tracking"
        />
      </div>

      <div className="flex flex-col sm:flex-row gap-4 items-start border-t border-grid-line pt-12">
        <Link
          href="#"
          className="text-accent-brand border border-accent-brand rounded-[6px] px-6 py-3 font-medium hover:bg-accent-brand/10 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-brand"
        >
          Read the full report
        </Link>
        <Link
          href="/methodology"
          className="text-accent-brand border border-accent-brand rounded-[6px] px-6 py-3 font-medium hover:bg-accent-brand/10 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-brand"
        >
          Explore the methodology
        </Link>
      </div>
    </div>
  );
}
