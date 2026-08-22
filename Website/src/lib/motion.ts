import { Transition, Variants } from "framer-motion";

export const transition: Record<string, Transition> = {
  sectionLineDraw: { duration: 0.5, ease: [0.4, 0, 0.2, 1] },
  contentReveal: { duration: 0.35, delay: 0.45, ease: "easeOut" },
  hoverBorder: { duration: 0.15, ease: "linear" }
};

export const variants: Record<string, Variants> = {
  contentReveal: {
    hidden: { opacity: 0, y: 8 },
    visible: { opacity: 1, y: 0 }
  }
};
