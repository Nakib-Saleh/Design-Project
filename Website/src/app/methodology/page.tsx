"use client";
import { motion, useReducedMotion } from "framer-motion";
import { transition, variants } from "@/lib/motion";
import PipelineDiagram from "@/components/ui/PipelineDiagram";
import TaskTag from "@/components/ui/TaskTag";
import CompareStat from "@/components/ui/CompareStat";

export default function Methodology() {
  const shouldReduceMotion = useReducedMotion();

  return (
    <div className="max-w-[1024px] mx-auto px-6 lg:px-12 py-24 lg:py-32">

      <motion.div
        initial={shouldReduceMotion ? "visible" : "hidden"}
        whileInView="visible"
        viewport={{ once: true, margin: "-100px" }}
        variants={variants.contentReveal}
        transition={transition.contentReveal}
        className="prose-col mb-16"
      >
        <p className="font-mono text-xs text-accent-primary mb-4 uppercase tracking-widest">
          How the system measures motion
        </p>
        <h1
          className="text-4xl lg:text-6xl mb-8 text-ink font-bold"
          style={{ fontFamily: "var(--font-display)" }}
        >
          Methodology
        </h1>
      </motion.div>

      {/* Architecture Diagram */}
      <PipelineDiagram />

      {/* Numbered steps */}
      <div className="flex flex-col gap-12 max-w-[68ch]">
        {[
          { num: "01", title: "Prior extraction", tag: "Vision", desc: "Parse the physical prior given in text (e.g. object size, known velocity)." },
          { num: "02", title: "Pixel-to-world scaling", tag: "Scale Recovery", desc: "Recover a scale factor between pixel-space and world units using the prior." },
          { num: "03", title: "Trajectory measurement", tag: "Tracking", desc: "Track the target object across frames in pixel space." },
          { num: "04", title: "Kinematic inference", tag: "Velocity", desc: "Apply the recovered scale to compute the requested quantity." },
          { num: "05", title: "Output", tag: "Prediction", desc: "Return a single numeric value with units." },
        ].map((step, i) => (
          <motion.div
            key={step.num}
            className="flex gap-6 items-start"
            initial={shouldReduceMotion ? "visible" : "hidden"}
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={variants.contentReveal}
            transition={{ ...transition.contentReveal, delay: (i % 3) * 0.08 }}
          >
            <div className="font-mono text-accent-primary font-medium mt-1 flex-shrink-0">{step.num}</div>
            <div>
              <div className="flex items-center gap-3 mb-2">
                <h3
                  className="text-xl text-ink font-semibold"
                  style={{ fontFamily: "var(--font-display)" }}
                >
                  {step.title}
                </h3>
                <TaskTag label={step.tag} />
              </div>
              {step.num === "05" ? (
                <div className="mt-4">
                  <p className="text-ink-muted leading-relaxed mb-4">{step.desc}</p>
                  <CompareStat groundTruth="25.00" prediction="24.99" unit="cm/s" />
                </div>
              ) : (
                <p className="text-ink-muted leading-relaxed">{step.desc}</p>
              )}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
