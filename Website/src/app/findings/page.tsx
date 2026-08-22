"use client";
import FindingsAnimation from "@/components/ui/FindingsAnimation";
import Callout from "@/components/ui/Callout";
import { motion, useReducedMotion } from "framer-motion";
import { transition, variants } from "@/lib/motion";
import InProgressPanel from "@/components/ui/InProgressPanel";

const sceneData = [
  { label: "Clean background", val: 3.1 },
  { label: "Cluttered", val: 7.4 },
  { label: "Low contrast", val: 11.2 },
  { label: "Inconsistent prior", val: 19.8 },
];

const clipData = [
  { pred: 12.4, truth: 11.9 },
  { pred: 18.7, truth: 17.5 },
  { pred: 9.1, truth: 10.2 },
  { pred: 24.3, truth: 22.8 },
  { pred: 15.6, truth: 16.1 },
  { pred: 8.9, truth: 9.4 },
];

export default function Findings() {
  const shouldReduceMotion = useReducedMotion();

  return (
    <div className="max-w-[1024px] mx-auto px-6 lg:px-12 py-24 lg:py-32">
      <FindingsAnimation />

      <motion.div
        initial={shouldReduceMotion ? "visible" : "hidden"}
        whileInView="visible"
        viewport={{ once: true, margin: "-100px" }}
        variants={variants.contentReveal}
        transition={transition.contentReveal}
        className="prose-col mb-20"
      >
        <p className="font-mono text-xs text-accent-primary mb-4 uppercase tracking-widest">
          What the results show
        </p>
        <h1
          className="text-4xl lg:text-6xl mb-8 text-ink font-bold"
          style={{ fontFamily: "var(--font-display)" }}
        >
          Findings
        </h1>
      </motion.div>

      {/* Chart 1 */}
      <motion.div
        initial={shouldReduceMotion ? "visible" : "hidden"}
        whileInView="visible"
        viewport={{ once: true, margin: "-100px" }}
        variants={variants.contentReveal}
        transition={transition.contentReveal}
        className="mb-24"
      >
        <h3
          className="text-2xl text-ink mb-1 font-semibold"
          style={{ fontFamily: "var(--font-display)" }}
        >
          Velocity prediction error by scene type
        </h3>

        <InProgressPanel>
          <div className="w-full bg-surface border border-grid-line p-8 mb-8">
          <div className="flex flex-col gap-5 max-w-[600px]">
            {sceneData.map((item) => (
              <div key={item.label} className="flex items-center gap-4">
                <div className="w-[148px] text-sm text-ink-muted flex-shrink-0">{item.label}</div>
                <div className="flex-grow h-7 bg-base relative border border-grid-line overflow-hidden">
                  <div
                    className="absolute top-0 left-0 h-full bg-accent-primary opacity-80"
                    style={{ width: `${(item.val / 25) * 100}%` }}
                  />
                  {/* Threshold line at 5cm/s = 20% */}
                  <div className="absolute top-0 h-full border-l-2 border-accent-secondary border-dashed z-10" style={{ left: "20%" }} />
                </div>
                <div className="w-[44px] text-right font-mono text-accent-primary text-sm flex-shrink-0">{item.val}</div>
              </div>
            ))}
            <div className="flex pl-[164px] text-xs text-ink-muted justify-between font-mono mt-1">
              <span>0</span>
              <span className="text-accent-secondary relative -left-2">5.0 ← threshold</span>
              <span>25 cm/s</span>
            </div>
          </div>
        </div>
        </InProgressPanel>

        <Callout>
          <span className="font-semibold text-ink">Predictions stay close to ground truth under clean visual conditions,</span>{" "}
          but degrade sharply when the physical prior is inconsistent with the scene.
        </Callout>
      </motion.div>

      {/* Chart 2 */}
      <motion.div
        initial={shouldReduceMotion ? "visible" : "hidden"}
        whileInView="visible"
        viewport={{ once: true, margin: "-100px" }}
        variants={variants.contentReveal}
        transition={transition.contentReveal}
      >
        <h3
          className="text-2xl text-ink mb-1 font-semibold"
          style={{ fontFamily: "var(--font-display)" }}
        >
          Predicted vs. ground-truth speed
        </h3>

        <InProgressPanel>
          <div className="w-full bg-surface border border-grid-line p-8 mb-8 overflow-x-auto">
          <div className="flex items-end gap-2 min-w-[480px] max-w-[560px] h-[220px] border-b border-l border-grid-line relative px-4 pb-0">
            {/* Y-axis labels */}
            <div className="absolute left-[-36px] top-0 font-mono text-xs text-ink-muted">30</div>
            <div className="absolute left-[-36px] bottom-0 font-mono text-xs text-ink-muted">0</div>
            {clipData.map((clip, i) => (
              <div key={i} className="flex gap-1 items-end h-full flex-1">
                <div
                  className="flex-1 bg-accent-secondary opacity-80"
                  style={{ height: `${(clip.pred / 30) * 100}%` }}
                  title={`Predicted: ${clip.pred}`}
                />
                <div
                  className="flex-1 bg-accent-primary opacity-80"
                  style={{ height: `${(clip.truth / 30) * 100}%` }}
                  title={`Ground truth: ${clip.truth}`}
                />
              </div>
            ))}
          </div>
          <div className="flex gap-6 mt-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-accent-secondary opacity-80" />
              <span className="text-ink-muted">Predicted (LLM)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-accent-primary opacity-80" />
              <span className="text-ink-muted">Ground truth</span>
            </div>
          </div>
        </div>
        </InProgressPanel>

        <Callout>
          The model systematically tracks the target object well, but overall accuracy
          depends heavily on the robustness of the recovered pixel-to-world scale factor.
        </Callout>
      </motion.div>
    </div>
  );
}
