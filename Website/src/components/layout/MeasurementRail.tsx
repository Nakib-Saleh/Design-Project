"use client";
import { motion, useScroll, useTransform, useReducedMotion } from "framer-motion";

export default function MeasurementRail() {
  const { scrollYProgress } = useScroll();
  const shouldReduceMotion = useReducedMotion();

  const scaleFill = useTransform(scrollYProgress, [0, 1], [0, 1]);
  const markerPos = useTransform(scrollYProgress, [0, 1], ["0%", "100%"]);

  if (shouldReduceMotion) return null;

  return (
    <>
      {/* Mobile horizontal progress bar */}
      <div className="lg:hidden fixed top-0 left-0 w-full h-[3px] bg-grid-line z-[60]">
        <motion.div
          className="h-full bg-accent-primary opacity-40 origin-left"
          style={{ scaleX: scaleFill }}
        />
        <motion.div
          className="absolute top-1/2 -translate-y-1/2 w-2 h-2 bg-accent-primary rounded-full"
          style={{ left: markerPos, marginLeft: "-4px" }}
        />
      </div>

      {/* Desktop vertical measurement rail */}
      <div className="hidden lg:block fixed top-0 right-12 w-[2px] h-full bg-grid-line z-40">
        <div
          className="absolute inset-0 overflow-hidden opacity-50"
          style={{
            backgroundSize: "100% 80px",
            backgroundImage: "linear-gradient(to bottom, var(--color-grid-line) 1px, transparent 1px)",
            width: "6px",
            left: "-2px",
          }}
        />
        <motion.div
          className="absolute top-0 left-0 w-full bg-accent-primary opacity-20 origin-top"
          style={{ scaleY: scaleFill }}
        />
        <motion.div
          className="absolute left-1/2 -translate-x-1/2 w-2 h-2 bg-accent-primary rounded-full"
          style={{ top: markerPos, marginTop: "-4px" }}
        />
      </div>
    </>
  );
}
