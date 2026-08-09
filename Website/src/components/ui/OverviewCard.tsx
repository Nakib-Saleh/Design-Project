import Link from "next/link";
import TaskTag from "./TaskTag";

interface Props {
  title: string;
  description: string;
  href: string;
  tagLabel?: string;
}

export default function OverviewCard({ title, description, href, tagLabel }: Props) {
  return (
    <Link
      href={href}
      className="block group h-full focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-brand focus-visible:ring-offset-2 focus-visible:ring-offset-base transition-transform duration-300 hover:-translate-y-1 hover:shadow-lg hover:shadow-ink/5"
    >
      <div className="h-full border border-grid-line p-6 lg:p-8 bg-surface flex flex-col justify-between transition-colors duration-150 ease-linear group-hover:border-accent-brand">
        <div>
          {tagLabel && <TaskTag label={tagLabel} className="mb-4" />}
          <h3
            className="text-xl lg:text-2xl mb-4 text-ink font-semibold"
            style={{ fontFamily: "var(--font-display)" }}
          >
            {title}
          </h3>
        </div>
        <p className="text-ink-muted leading-relaxed text-sm lg:text-base">{description}</p>
      </div>
    </Link>
  );
}
