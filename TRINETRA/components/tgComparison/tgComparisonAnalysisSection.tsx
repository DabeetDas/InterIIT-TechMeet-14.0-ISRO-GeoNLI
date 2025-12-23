"use client";

import { DropZoneReact } from "./dropZone";
import { useTgComparisonStore } from "@/stores/tgComparison/useTgComparisonStore";
import {
  uploadFile,
  getPublicUrl,
} from "@/lib/tgComparison/supbaseClientTsStorage";
import { prettyLog } from "@/lib/devLogger";

export default function TgComparisonAnalysisSection() {
  // ✅ Correct Zustand usage — triggers rerender when store updates
  const tg = useTgComparisonStore((s) => s.tgComparisonStore);
  const setTg = useTgComparisonStore((s) => s.setTgComparisonStore);

  function getBaseObject() {
    return {
      id: crypto.randomUUID(),
      image1_url: null,
      image2_url: null,
      image1_gsd: 0,
      image2_gsd: 0,
      created_at: new Date().toISOString(),
    };
  }

  /*************************************************
   * LEFT IMAGE UPLOADER
   *************************************************/
  async function updateLeft(file: File, gsd: number = 0) {
    console.log("⬅️ updateLeft CALLED");

    const { path } = await uploadFile(file);
    const publicUrl = await getPublicUrl(path);
    console.log("⬅️ LEFT URL =", publicUrl);

    // ❗ Always read the latest store state (no stale closure)
    const current =
      useTgComparisonStore.getState().tgComparisonStore ?? getBaseObject();

    // ❗ Now update
    setTg({
      ...current,
      image1_url: publicUrl,
      image1_gsd: gsd,
    });

    console.log("⬅️ UPDATED LEFT STORE =", useTgComparisonStore.getState());
  }

  /*************************************************
   * RIGHT IMAGE UPLOADER
   *************************************************/
  async function updateRight(file: File, gsd: number = 0) {
    console.log("➡️ updateRight CALLED");

    const { path } = await uploadFile(file);
    const publicUrl = await getPublicUrl(path);
    console.log("➡️ RIGHT URL =", publicUrl);

    const current =
      useTgComparisonStore.getState().tgComparisonStore ?? getBaseObject();

    setTg({
      ...current,
      image2_url: publicUrl,
      image2_gsd: gsd,
    });

    console.log("➡️ UPDATED RIGHT STORE =", useTgComparisonStore.getState());
  }

  /*************************************************
   * DEBUGGING (optional)
   *************************************************/
  console.log("📦 TG STORE (RENDER) =", tg);

  return (
    <div className="flex-[1.4] bg-[#0f1624] rounded-2xl p-6 flex flex-col gap-6">
      <h2 className="text-gray-200 text-xl font-semibold">
        Terrain Geo Analysis
      </h2>

      <div className="flex-1 grid grid-cols-2 gap-4 items-start">
        {/* LEFT IMAGE */}
        <div className="w-full">
          <div className="aspect-square rounded-2xl overflow-hidden bg-[#0b1724] flex items-center justify-center">
            {tg?.image1_url ? (
              <img
                src={tg.image1_url}
                alt="Left comparison"
                className="max-w-full max-h-full object-contain object-center"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center p-4">
                <DropZoneReact
                  label="Upload Left Image"
                  onFile={(file, gsd) => updateLeft(file, gsd)}
                />
              </div>
            )}
          </div>
        </div>

        {/* RIGHT IMAGE */}
        <div className="w-full">
          <div className="aspect-square rounded-2xl overflow-hidden bg-[#0b1724] flex items-center justify-center">
            {tg?.image2_url ? (
              <img
                src={tg.image2_url}
                alt="Right comparison"
                className="max-w-full max-h-full object-contain object-center"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center p-4">
                <DropZoneReact
                  label="Upload Right Image"
                  onFile={(file, gsd) => updateRight(file, gsd)}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
