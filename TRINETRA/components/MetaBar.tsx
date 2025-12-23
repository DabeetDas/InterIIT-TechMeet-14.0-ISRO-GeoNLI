"use client";

import { FileText, Ruler, Image } from "lucide-react";

export default function MetaBar({
  filename,
  imageClass,
  length,
}: {
  filename?: string;
  imageClass?: string;
  length?: number;
}) {
  // CHIP WIDTH — smaller so they fit in one line
  const CHIP_WIDTH = "w-[180px]";

  // Normal (dark mode) chip styling
  const chip = {
    bg: "bg-[#1f2227]",
    border: "border-[#3a3d42]",
    text: "text-gray-300",
    icon: "text-gray-400",
  };

  // border
  // border-[#222426] bg-[#0e0f11]
  return (
    <div
      className="
        rounded-2xl p-3 border-[#222426] bg-[#0e0f11]
      "
    >
      {/* HEADER */}
      <div className="mb-3 text-left text-[10px] uppercase tracking-widest text-gray-500">
        Metadata
      </div>

      {/* SINGLE LINE — NO WRAP */}
      <div className="flex gap-2 flex-nowrap overflow-hidden">
        {/* FILE NAME */}
        {filename && (
          <div
            className={`
              flex items-center gap-1.5 px-2.5 py-1 rounded-xl border truncate ${CHIP_WIDTH}
              text-[10px] font-medium
              ${chip.bg} ${chip.border} ${chip.text}
            `}
          >
            <FileText size={11} className={chip.icon} />
            <span className="truncate">{filename}</span>
          </div>
        )}

        {/* IMAGE CLASS */}
        {imageClass && (
          <div
            className={`
              flex items-center gap-1.5 px-2.5 py-1 rounded-xl border truncate ${CHIP_WIDTH}
              text-[10px] font-medium
              ${chip.bg} ${chip.border} ${chip.text}
            `}
          >
            <Image size={11} className={chip.icon} />
            <span className="truncate">{imageClass}</span>
          </div>
        )}

        {/* LENGTH */}
        {length !== undefined && (
          <div
            className={`
              flex items-center gap-1.5 px-2.5 py-1 rounded-xl border truncate ${CHIP_WIDTH}
              text-[10px] font-medium
              ${chip.bg} ${chip.border} ${chip.text}
            `}
          >
            <Ruler size={11} className={chip.icon} />
            <span className="truncate">{length} m/px</span>
          </div>
        )}
      </div>
    </div>
  );
}
