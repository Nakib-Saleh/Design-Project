"use client";
import { motion, useReducedMotion } from "framer-motion";

const steps = [
  "Video + Prior",
  "Prior Extraction",
  "Track & Scale",
  "Kinematic Inference",
  "Output"
];

export default function PipelineDiagram() {
  const shouldReduceMotion = useReducedMotion();

  return (
    <div className="w-full overflow-x-auto py-12 mb-8 no-scrollbar">
      <div className="min-w-[700px] flex items-center justify-between relative px-8">
        
        {/* The Animated Connecting Line */}
        <div className="absolute left-[10%] right-[10%] top-1/2 -translate-y-1/2 h-[2px] z-0">
          <svg width="100%" height="2" className="overflow-visible block">
            <motion.line
              x1="0"
              y1="1"
              x2="100%"
              y2="1"
              stroke="var(--color-grid-line)"
              strokeWidth="2"
              strokeDasharray="4 4"
              initial={shouldReduceMotion ? { pathLength: 1 } : { pathLength: 0 }}
              whileInView={{ pathLength: 1 }}
              viewport={{ once: true, margin: "-100px" }}
              transition={{ duration: 1.5, ease: "easeInOut" }}
            />
          </svg>
        </div>

        {/* The Nodes */}
        {steps.map((step, i) => (
          <motion.div
            key={i}
            className="relative z-10 flex flex-col items-center gap-3"
            initial={shouldReduceMotion ? { opacity: 1, y: 0 } : { opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.4, delay: i * 0.15 }}
          >
            <div className="w-12 h-12 rounded-full bg-surface border-2 border-accent-primary flex items-center justify-center text-accent-primary font-mono font-bold shadow-sm">
              {i + 1}
            </div>
            <span className="text-sm font-medium text-ink max-w-[100px] text-center leading-tight">
              {step}
            </span>
          </motion.div>
        ))}

      </div>
    </div>
  );
}
