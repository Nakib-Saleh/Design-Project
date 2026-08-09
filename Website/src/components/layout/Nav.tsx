"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { clsx } from "clsx";

const links = [
  { name: "Home", href: "/" },
  { name: "Literature Review", href: "/literature-review" },
  { name: "Methodology", href: "/methodology" },
  { name: "Findings", href: "/findings" },
  { name: "Demo", href: "/demo" },
];

export default function Nav() {
  const pathname = usePathname();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <nav
      className={clsx(
        "sticky top-0 z-50 bg-base border-b border-grid-line transition-all duration-200 flex items-center px-6 lg:px-12",
        scrolled ? "h-[60px]" : "h-[80px]"
      )}
    >
      <div className="flex gap-8 overflow-x-auto w-full no-scrollbar">
        {links.map((link) => {
          const isActive = pathname === link.href;
          return (
            <Link
              key={link.name}
              href={link.href}
              className={clsx(
                "relative whitespace-nowrap text-sm font-medium py-1 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-brand",
                isActive ? "text-accent-brand" : "text-ink hover:text-ink-muted"
              )}
            >
              {link.name}
              {isActive && (
                <span className="absolute left-0 -bottom-1 w-full h-[2px] bg-accent-brand" />
              )}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
