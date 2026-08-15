"use client";
import { motion, useReducedMotion } from "framer-motion";

export default function LitReviewAnimation() {
  const shouldReduceMotion = useReducedMotion();

  return (
    <div className="w-full bg-surface border border-grid-line p-8 mb-12 flex flex-col md:flex-row gap-8 justify-around overflow-hidden rounded-xl">
      
      {/* Qualitative VQA Side */}
      <div className="flex flex-col items-center gap-4 w-full md:w-1/2">
        <h4 className="font-mono text-xs text-ink-muted uppercase tracking-widest">Qualitative VQA</h4>
        
        <svg viewBox="0 0 200 150" className="w-full max-w-[200px] border border-grid-line/50 bg-base rounded bg-[linear-gradient(var(--color-grid-line)_1px,transparent_1px),linear-gradient(90deg,var(--color-grid-line)_1px,transparent_1px)] bg-[size:20px_20px]">
          {/* Static Bounding Box */}
          <rect x="50" y="50" width="60" height="40" fill="none" stroke="var(--color-ink-muted)" strokeWidth="2" strokeDasharray="4 2" />
          <text x="50" y="45" fill="var(--color-ink-muted)" fontSize="10" fontFamily="var(--font-mono)">"car"</text>
          
          {/* Static motion blur indicator */}
          <line x1="20" y1="70" x2="40" y2="70" stroke="var(--color-ink-muted)" strokeWidth="1" opacity="0.5" />
          <line x1="30" y1="60" x2="45" y2="60" stroke="var(--color-ink-muted)" strokeWidth="1" opacity="0.5" />
        </svg>
      </div>

      {/* Vertical Divider */}
      <div className="w-full h-[1px] md:w-[1px] md:h-auto bg-grid-line" />

      {/* Kinematic VQA Side */}
      <div className="flex flex-col items-center gap-4 w-full md:w-1/2">
        <h4 className="font-mono text-xs text-accent-primary uppercase tracking-widest font-bold">Kinematic VQA</h4>
        
        <svg viewBox="0 0 200 150" className="w-full max-w-[200px] border border-grid-line/50 bg-base rounded bg-[linear-gradient(var(--color-grid-line)_1px,transparent_1px),linear-gradient(90deg,var(--color-grid-line)_1px,transparent_1px)] bg-[size:20px_20px] shadow-[inset_0_0_20px_rgba(15,107,92,0.05)]">
          {/* Static Bounding Box */}
          <rect x="50" y="50" width="60" height="40" fill="none" stroke="var(--color-ink-muted)" strokeWidth="2" strokeDasharray="4 2" />
          
          {/* Animated Vector Arrow */}
          <motion.line
            x1="110" y1="70" x2="160" y2="70"
            stroke="var(--color-accent-primary)"
            strokeWidth="2.5"
            initial={shouldReduceMotion ? { pathLength: 1 } : { pathLength: 0 }}
            whileInView={{ pathLength: 1 }}
            viewport={{ once: false, margin: "-50px" }}
            transition={{ duration: 1.5, ease: "easeInOut", repeat: Infinity, repeatDelay: 1 }}
          />
          <motion.polygon
            points="160,70 152,65 152,75"
            fill="var(--color-accent-primary)"
            initial={shouldReduceMotion ? { opacity: 1 } : { opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: false, margin: "-50px" }}
            transition={{ duration: 0.2, delay: 1.4, repeat: Infinity, repeatDelay: 2.3 }}
          />

          {/* Animated Value Text */}
          <motion.text
            x="135" y="60"
            fill="var(--color-accent-primary)"
            fontSize="12"
            fontFamily="var(--font-mono)"
            textAnchor="middle"
            initial={shouldReduceMotion ? { opacity: 1 } : { opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: false, margin: "-50px" }}
            transition={{ duration: 0.5, delay: 1.5, repeat: Infinity, repeatDelay: 2 }}
          >
            v = 15m/s
          </motion.text>
        </svg>
      </div>

    </div>
  );
}
