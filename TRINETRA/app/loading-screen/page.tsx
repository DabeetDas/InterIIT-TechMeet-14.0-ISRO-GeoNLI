"use client";

import { motion } from "framer-motion";

export default function UploadLoadingScreen() {
  return (
    <div className="fixed inset-0 bg-black flex flex-col items-center justify-center text-white">
      {/* GLOW BACKDROP */}
      <motion.div
        className="absolute w-[300px] h-[300px] rounded-full bg-blue-600/20 blur-[120px]"
        animate={{ opacity: [0.3, 0.6, 0.3] }}
        transition={{ duration: 3, repeat: Infinity }}
      />

      {/* SPINNER */}
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ repeat: Infinity, duration: 1.2, ease: "linear" }}
        className="
          w-20 h-20 rounded-full 
          border-[5px] 
          border-[#4F7CFF] 
          border-t-transparent 
          shadow-[0_0_25px_#4F7CFF80]
        "
      />

      {/* TEXT */}
      <motion.p
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="mt-8 text-lg text-gray-300 tracking-wide"
      >
        Uploading your satellite map…
      </motion.p>

      {/* SUBTEXT */}
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.8 }}
        transition={{ delay: 1, duration: 0.8 }}
        className="mt-2 text-sm text-gray-500"
      >
        Preparing analysis suite…
      </motion.p>
    </div>
  );
}
