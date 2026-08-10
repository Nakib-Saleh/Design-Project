import type { Metadata } from "next";
import { Space_Grotesk, Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import Nav from "@/components/layout/Nav";
import Footer from "@/components/layout/Footer";
import MeasurementRail from "@/components/layout/MeasurementRail";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
  weight: ["400", "500", "600", "700"],
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  weight: ["400", "500"],
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  weight: ["400"],
});

export const metadata: Metadata = {
  title: "Kinematics on VQA via LLM",
  description: "An undergraduate thesis on quantitative kinematic reasoning in Visual Question Answering via LLMs.",
  openGraph: {
    title: "Kinematics on VQA via LLM",
    description: "An undergraduate thesis on quantitative kinematic reasoning in Visual Question Answering via LLMs.",
    type: "website",
    images: ["/og-image.png"],
  },
  twitter: {
    card: "summary_large_image",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${spaceGrotesk.variable} ${inter.variable} ${jetbrainsMono.variable}`}
    >
      <body className="antialiased selection:bg-accent-primary/20 selection:text-accent-primary">
        <Nav />
        <MeasurementRail />
        <main>{children}</main>
        <Footer />
      </body>
    </html>
  );
}
