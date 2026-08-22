"use client";
import CompareStat from "@/components/ui/CompareStat";
import TaskTag from "@/components/ui/TaskTag";

export default function Demo() {
  return (
    <div className="max-w-[1280px] mx-auto px-6 lg:px-12 py-24 lg:py-32">

      <div className="prose-col mb-16">
        <p className="font-mono text-xs text-accent-primary mb-4 uppercase tracking-widest">
          Interactive Walkthrough
        </p>
        <h1
          className="text-4xl lg:text-6xl mb-8 text-ink font-bold"
          style={{ fontFamily: "var(--font-display)" }}
        >
          Demo
        </h1>
        <p className="text-ink-muted leading-relaxed">
          Select a sample below to see the extraction, scaling, and inference pipeline in action.
          (Live inference coming soon).
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
        {/* Panel 1: Input */}
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-8 h-8 rounded-full bg-surface border border-grid-line flex items-center justify-center font-mono text-sm font-bold text-ink">1</div>
            <h2 className="text-xl font-semibold text-ink" style={{ fontFamily: "var(--font-display)" }}>Input</h2>
            <TaskTag label="Vision" />
          </div>
          
          <div className="bg-surface border border-grid-line p-2 rounded-lg">
            <video
              autoPlay
              loop
              muted
              playsInline
              className="w-full aspect-video object-cover rounded bg-base"
              src="/Free%20Sport%20Videos-%204K%20&%20HD%20-%20No%20Watermark%20-%20Download%20Now.mp4"
            />
          </div>
          
          <div className="bg-surface border border-grid-line p-4 rounded-lg">
            <h4 className="text-xs uppercase tracking-wider text-ink-muted font-mono mb-2">Physical Prior</h4>
            <p className="text-ink text-sm">"The diameter of the soccer ball is exactly 22 cm."</p>
          </div>
        </div>

        {/* Panel 2: Processing */}
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-8 h-8 rounded-full bg-surface border border-grid-line flex items-center justify-center font-mono text-sm font-bold text-ink">2</div>
            <h2 className="text-xl font-semibold text-ink" style={{ fontFamily: "var(--font-display)" }}>Processing</h2>
            <TaskTag label="Tracking" />
          </div>
          
          <div className="bg-surface border border-grid-line p-2 rounded-lg relative overflow-hidden group">
            <video
              autoPlay
              loop
              muted
              playsInline
              className="w-full aspect-video object-cover rounded bg-base relative z-0"
              src="/Free%20Tracking%20Videos-%204K%20&%20HD%20-%20No%20Watermark%20-%20Download%20Now.mp4"
            />
            {/* Mock tracking overlay */}
            <div className="absolute inset-2 border border-accent-primary/40 rounded z-10 pointer-events-none" />
            <div className="absolute top-1/2 left-1/2 w-12 h-12 border-2 border-accent-secondary rounded-sm -translate-x-1/2 -translate-y-1/2 z-20 shadow-[0_0_10px_rgba(201,117,43,0.5)]" />
            <div className="absolute bottom-4 left-4 bg-ink/80 text-base px-2 py-1 rounded text-xs font-mono backdrop-blur-sm z-30 text-[#F7F6F1]">
              Scale: 1px = 0.45cm
            </div>
          </div>
        </div>

        {/* Panel 3: Output */}
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-8 h-8 rounded-full bg-surface border border-grid-line flex items-center justify-center font-mono text-sm font-bold text-ink">3</div>
            <h2 className="text-xl font-semibold text-ink" style={{ fontFamily: "var(--font-display)" }}>Output</h2>
            <TaskTag label="Prediction" />
          </div>
          
          <div className="bg-surface border border-grid-line p-6 rounded-lg flex flex-col justify-center min-h-[220px]">
            <h4 className="text-xs uppercase tracking-wider text-ink-muted font-mono mb-6">Kinematic Result</h4>
            <CompareStat groundTruth="18.5" prediction="18.1" unit="m/s" />
            <p className="text-sm text-ink-muted mt-6">
              The model successfully tracked the object trajectory and applied the scaling factor to recover velocity within a 2.1% error margin.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
