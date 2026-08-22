"use client";
import LitReviewAnimation from "@/components/ui/LitReviewAnimation";
import ComparisonTable from "@/components/ui/ComparisonTable";
import Callout from "@/components/ui/Callout";
import { motion, useReducedMotion } from "framer-motion";
import { transition, variants } from "@/lib/motion";

export default function LiteratureReview() {
  const shouldReduceMotion = useReducedMotion();

  return (
    <div className="max-w-[1024px] mx-auto px-6 lg:px-12 py-24 lg:py-32">
      <LitReviewAnimation />

      <motion.div
        initial={shouldReduceMotion ? "visible" : "hidden"}
        whileInView="visible"
        viewport={{ once: true, margin: "-100px" }}
        variants={variants.contentReveal}
        transition={transition.contentReveal}
        className="prose-col"
      >
        <p className="font-mono text-xs text-accent-primary mb-4 uppercase tracking-widest">
          Where prior work stops
        </p>
        <h1
          className="text-4xl lg:text-6xl mb-8 text-ink font-bold"
          style={{ fontFamily: "var(--font-display)" }}
        >
          Literature Review
        </h1>

        <p className="text-lg text-ink-muted leading-relaxed mb-16">
          Existing Visual Question Answering (VQA) benchmarks focus overwhelmingly on qualitative description—identifying objects, describing actions, or evaluating spatial relations. This thesis targets a gap in the literature: numeric, physically-grounded answers.
        </p>

        <h3
          className="text-2xl text-ink mb-4 font-semibold"
          style={{ fontFamily: "var(--font-display)" }}
        >
          Video-grounded question answering
        </h3>
        <p className="text-ink-muted leading-relaxed mb-8">
          Current state-of-the-art models excel at narrative reasoning over video frames but falter when asked to extract continuous physical quantities without explicit supervision.
        </p>
      </motion.div>

      <ComparisonTable />

      <motion.div
        initial={shouldReduceMotion ? "visible" : "hidden"}
        whileInView="visible"
        viewport={{ once: true, margin: "-100px" }}
        variants={variants.contentReveal}
        transition={transition.contentReveal}
      >
        <Callout>
          No existing work evaluates whether an LLM can recover a target kinematic
          quantity from a video and a single physical prior — this thesis addresses
          that directly.
        </Callout>
      </motion.div>
    </div>
  );
}
