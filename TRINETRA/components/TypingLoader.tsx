"use client";

import { motion } from "framer-motion";

export default function TypingLoader() {
  return (
    <div className="flex items-center gap-1 py-1">
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          className="w-2 h-2 bg-gray-400 rounded-full"
          animate={{
            opacity: [0.2, 1, 0.2],
            y: [0, -3, 0],
          }}
          transition={{
            duration: 0.6,
            repeat: Infinity,
            delay: i * 0.15,
          }}
        />
      ))}
    </div>
  );
}
