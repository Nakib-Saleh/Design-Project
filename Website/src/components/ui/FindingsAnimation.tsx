"use client";
import { motion, useReducedMotion } from "framer-motion";

export default function FindingsAnimation() {
  const shouldReduceMotion = useReducedMotion();

  return (
    <div className="w-full bg-surface border border-grid-line p-8 mb-12 flex justify-center overflow-hidden rounded-xl">
      <svg viewBox="0 0 400 150" className="w-full max-w-[500px]">
        {/* Axes */}
        <line x1="20" y1="130" x2="380" y2="130" stroke="var(--color-grid-line)" strokeWidth="1" />
        <line x1="20" y1="20" x2="20" y2="130" stroke="var(--color-grid-line)" strokeWidth="1" />

        {/* Labels */}
        <text x="380" y="145" fill="var(--color-ink-muted)" fontSize="10" fontFamily="var(--font-mono)" textAnchor="end">Time</text>
        <text x="5" y="30" fill="var(--color-ink-muted)" fontSize="10" fontFamily="var(--font-mono)" transform="rotate(-90 10,30)">Velocity</text>

        {/* Ground Truth Curve (Teal) */}
        <motion.path
          d="M 20 120 Q 150 20 280 80 T 380 40"
          fill="none"
          stroke="var(--color-accent-primary)"
          strokeWidth="3"
          initial={shouldReduceMotion ? { pathLength: 1 } : { pathLength: 0 }}
          whileInView={{ pathLength: 1 }}
          viewport={{ once: false, margin: "-50px" }}
          transition={{ duration: 2, ease: "easeInOut", repeat: Infinity, repeatDelay: 1 }}
        />
        
        {/* Prediction Curve (Amber) - slightly off at the end */}
        <motion.path
          d="M 20 120 Q 148 22 280 78 T 380 55"
          fill="none"
          stroke="var(--color-accent-secondary)"
          strokeWidth="2"
          strokeDasharray="4 4"
          initial={shouldReduceMotion ? { pathLength: 1 } : { pathLength: 0 }}
          whileInView={{ pathLength: 1 }}
          viewport={{ once: false, margin: "-50px" }}
          transition={{ duration: 2, ease: "easeInOut", delay: 0.2, repeat: Infinity, repeatDelay: 1 }}
        />

        {/* Legends */}
        <g transform="translate(40, 30)">
          <line x1="0" y1="0" x2="15" y2="0" stroke="var(--color-accent-primary)" strokeWidth="2" />
          <text x="20" y="4" fill="var(--color-ink)" fontSize="10" fontFamily="var(--font-sans)">Ground Truth</text>
        </g>
        <g transform="translate(40, 45)">
          <line x1="0" y1="0" x2="15" y2="0" stroke="var(--color-accent-secondary)" strokeWidth="2" strokeDasharray="2 2" />
          <text x="20" y="4" fill="var(--color-ink)" fontSize="10" fontFamily="var(--font-sans)">Predicted</text>
        </g>
      </svg>
    </div>
  );
}
