"use client";

import { useRef, useState, useEffect } from "react";
import { useImageStore } from "@/stores/imageStore";
import GroundedImage from "@/components/GroundedImage";
import { useDetectionStore } from "@/stores/useDetectionStore";
import { useAnalysisStore } from "@/stores/apiCallAnalysisStore";
import { usePathname } from "next/navigation";
import { getImageMetadata } from "@/lib/getImageMetadata/getImageMetadata";
import { useImageMetadataStore } from "@/lib/imageMetadataStore";
import MetaBar from "@/components/MetaBar";
import GradientText from "@/components/GradientText";

export default function AnalysisSection() {
  const imageClass = useImageMetadataStore((s) => s.imageClass);
  const setImageClass = useImageMetadataStore((s) => s.setImageClass);
  const clearImageClass = useImageMetadataStore((s) => s.clearImageClass);

  const pathname = usePathname();

  const { imageUrl, filename, heightKm: gsd } = useImageStore();
  const { result } = useAnalysisStore();

  const {
    categories,
    currentCategory,
    setCategory,
    data,
    mergeDetections,
    clearDetections,
  } = useDetectionStore();

  const containerRef = useRef<HTMLDivElement | null>(null);
  const [containerSize, setContainerSize] = useState({ w: 0, h: 0 });

  useEffect(() => {
    clearDetections();
    clearImageClass();
  }, [pathname, clearDetections, clearImageClass]);

  useEffect(() => {
    if (!result?.groundingData) return;
    mergeDetections(result.groundingData);
  }, [result, mergeDetections]);

  useEffect(() => {
    if (!imageUrl) return;
    const url = imageUrl; // <-- NOW TS knows it's a guaranteed string

    async function load() {
      const meta = await getImageMetadata(url);
      if (meta?.image_class) {
        setImageClass(meta.image_class);
      }
    }

    load();
  }, [imageUrl, setImageClass]);

  useEffect(() => {
    if (!containerRef.current) return;
    const update = () => {
      const rect = containerRef.current!.getBoundingClientRect();
      setContainerSize({ w: rect.width, h: rect.height });
    };

    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  return (
    <div className="flex-[1.4] bg-[#000000] rounded-2xl p-6 flex flex-col gap-4 overflow-hidden">
      {/* CATEGORY CHIPS */}
      {categories.length > 0 && (
        <div
          className="
              flex gap-3 mb-2 overflow-x-auto flex-nowrap
              scrollbar-thin scrollbar-thumb-gray-700
              whitespace-nowrap pr-2
            "
        >
          <button
            onClick={() => setCategory(null)}
            className={`px-4 py-2 rounded-xl text-sm shrink-0 transition 
                ${
                  currentCategory === null
                    ? "bg-blue-600 text-white"
                    : "bg-[#1a2333] text-gray-300"
                }
              `}
          >
            None
          </button>

          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setCategory(cat)}
              className={`
                  px-4 py-2 rounded-xl text-sm shrink-0 transition
                  ${
                    currentCategory === cat
                      ? "bg-blue-600 text-white"
                      : "bg-[#1a2333] text-gray-300"
                  }
                `}
            >
              {cat} ({data[cat].length})
            </button>
          ))}
        </div>
      )}

      {/* IMAGE PANEL */}
      <div
        ref={containerRef}
        className="flex-1 rounded-2xl bg-[#000000] overflow-hidden flex items-center justify-center"
      >
        {!imageUrl && <div className="text-gray-500">No image uploaded</div>}

        {imageUrl && containerSize.w > 0 && (
          <GroundedImage
            imageUrl={imageUrl}
            containerWidth={containerSize.w}
            containerHeight={containerSize.h}
          />
        )}
      </div>

      {/* METADATA */}
      <div className="w-full">
        <MetaBar
          filename={filename ?? undefined}
          imageClass={imageClass ?? undefined}
          length={gsd ?? undefined}
        />
      </div>
    </div>
  );
}
