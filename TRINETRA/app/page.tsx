"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import GradientText from "@/components/GradientText";

export default function Home() {
  return (
    <main className="relative h-screen w-full overflow-hidden">
      {/* Background Video */}
      <video
        className="absolute inset-0 h-full w-full object-cover"
        autoPlay
        loop
        muted
        playsInline
      >
        <source src="/background/hero.mp4" type="video/mp4" />
      </video>

      {/* Dark overlay */}
      <div className="absolute inset-0 bg-black/30" />

      {/* Foreground Content */}
      <div className="relative z-10 flex flex-col items-center justify-end h-full text-white text-center px-4 pb-12">
        {/* Subtext */}
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="text-sm tracking-wide uppercase opacity-90"
        >
          GeoNLI: Geographic Natural Language Interface
        </motion.p>

        {/* Title */}
        <motion.h1
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 1, ease: "easeOut" }}
          className="text-4xl md:text-6xl font-bold mt-4"
        >
          <GradientText
            colors={["#9CA3AF", "#6B7280", "#4B5563", "#6B7280", "#9CA3AF"]}
            animationSpeed={6}
            showBorder={false}
            className="custom-class"
          >
            TRINETRA
          </GradientText>
        </motion.h1>

        {/* Tagline */}
        <motion.p
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35, duration: 0.9, ease: "easeOut" }}
          className="text-sm md:text-base mt-3 opacity-90"
        >
          AI powered chatbot for Geospatial Analysis
        </motion.p>

        {/* Button */}
        <Link href="/map">
          <motion.button
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{
              delay: 0.55,
              duration: 0.7,
              ease: "easeOut",
            }}
            className="mt-6 flex items-center gap-2 rounded-full border border-white/80 px-6 py-3 text-sm uppercase tracking-wide hover:bg-white hover:text-black transition shadow-lg shadow-white/10"
          >
            <span className="text-lg">→</span>
          </motion.button>
        </Link>
      </div>
    </main>
  );
}
