"use client";
import { motion, useReducedMotion, Variants } from "framer-motion";

export default function Hero() {
  const shouldReduceMotion = useReducedMotion();

  const arcVariants: Variants = {
    hidden: { pathLength: 0 },
    visible: { pathLength: 1, transition: { duration: 1.2, ease: "easeInOut" } },
  };

  const arrowVariants: Variants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { duration: 0.3, delay: 1.0 } },
  };

  const textVariants: Variants = {
    hidden: { opacity: 0, y: 8 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.4, delay: 1.3 } },
  };

  return (
    <div className="relative pt-12 pb-24 lg:pt-20 lg:pb-28">
      {/* Trajectory SVG */}
      <div
        className="w-full h-[200px] lg:h-[280px] mb-10 relative pointer-events-none overflow-hidden rounded-xl"
        aria-hidden="true"
      >
        {/* Background Video overlay behind the math animation */}
        <video
          autoPlay
          loop
          muted
          playsInline
          className="absolute inset-0 w-full h-full object-cover opacity-[0.08]"
          src="/Ball%20Rolling.mp4"
        />
        
        <svg
          viewBox="0 0 1000 280"
          preserveAspectRatio="none"
          className="absolute inset-0 w-full h-full overflow-visible"
        >
          {/* Grid lines for the "physics paper" feel */}
          {[...Array(5)].map((_, i) => (
            <line
              key={i}
              x1="0"
              y1={56 * i}
              x2="1000"
              y2={56 * i}
              stroke="var(--color-grid-line)"
              strokeWidth="1"
              opacity="0.6"
            />
          ))}
          {[...Array(11)].map((_, i) => (
            <line
              key={i}
              x1={100 * i}
              y1="0"
              x2={100 * i}
              y2="280"
              stroke="var(--color-grid-line)"
              strokeWidth="1"
              opacity="0.6"
            />
          ))}

          {/* The parabolic trajectory arc */}
          <motion.path
            d="M 50 240 Q 500 -60 950 240"
            fill="none"
            stroke="var(--color-accent-primary)"
            strokeWidth="2.5"
            initial={shouldReduceMotion ? "visible" : "hidden"}
            animate="visible"
            variants={arcVariants}
          />

          {/* Velocity vector at ~25% along arc */}
          <motion.g
            initial={shouldReduceMotion ? "visible" : "hidden"}
            animate="visible"
            variants={arrowVariants}
            transform="translate(230, 100)"
          >
            <line x1="0" y1="0" x2="28" y2="-16" stroke="var(--color-accent-primary)" strokeWidth="2" />
            <polygon points="28,-16 20,-18 28,-8" fill="var(--color-accent-primary)" />
          </motion.g>

          {/* Velocity vector at ~75% along arc */}
          <motion.g
            initial={shouldReduceMotion ? "visible" : "hidden"}
            animate="visible"
            variants={arrowVariants}
            transform="translate(760, 100)"
          >
            <line x1="0" y1="0" x2="28" y2="16" stroke="var(--color-accent-primary)" strokeWidth="2" />
            <polygon points="28,16 20,18 28,8" fill="var(--color-accent-primary)" />
          </motion.g>

          {/* Ground-truth reference line at bottom */}
          <motion.line
            x1="0" y1="240" x2="1000" y2="240"
            stroke="var(--color-accent-secondary)"
            strokeWidth="1.5"
            strokeDasharray="8 5"
            initial={shouldReduceMotion ? { opacity: 1 } : { opacity: 0 }}
            animate={{ opacity: 0.7 }}
            transition={{ delay: 1.1, duration: 0.4 }}
          />
          <motion.text
            x="960" y="234"
            fill="var(--color-accent-secondary)"
            fontSize="11"
            textAnchor="end"
            fontFamily="var(--font-mono)"
            initial={shouldReduceMotion ? { opacity: 1 } : { opacity: 0 }}
            animate={{ opacity: 0.7 }}
            transition={{ delay: 1.2, duration: 0.3 }}
          >
            ground
          </motion.text>
        </svg>
      </div>

      {/* Headline text */}
      <motion.div
        className="prose-col"
        initial={shouldReduceMotion ? "visible" : "hidden"}
        animate="visible"
        variants={textVariants}
      >
        <p className="font-mono text-xs text-ink-muted mb-4 uppercase tracking-widest">
          Nakib · CSE, IUT · Thesis Pre-Defence
        </p>
        <h1
          className="text-4xl lg:text-6xl leading-[1.1] mb-6 text-ink font-bold"
          style={{ fontFamily: "var(--font-display)" }}
        >
          Can a language model tell you <span className="text-accent-brand">how fast</span> something is moving?
        </h1>
        <p className="text-lg text-ink-muted leading-relaxed">
          An undergraduate thesis on quantitative kinematic reasoning in Visual
          Question Answering via LLMs.
        </p>
      </motion.div>
    </div>
  );
}
