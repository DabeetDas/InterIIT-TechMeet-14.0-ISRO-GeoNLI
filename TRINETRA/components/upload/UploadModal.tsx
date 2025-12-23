"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import UploadDropzone from "./UploadDropzone";
import UploadFilePreview from "./UploadFilePreview";

import {
  uploadFile,
  getPublicUrl,
  createSignedUrl,
} from "@/lib/supabaseStorage";

import { supabase } from "@/lib/supabaseClient";
import { useImageStore } from "@/stores/imageStore";
import { useSessionStore } from "@/stores/forHistory/sessionStore";
import { useDetectionStore } from "@/stores/useDetectionStore";

interface UploadModalProps {
  open: boolean;
  onClose: () => void;
  initialFile?: File | null;
}

export default function UploadModal({
  open,
  onClose,
  initialFile,
}: UploadModalProps) {
  const router = useRouter();

  const setImage = useImageStore((s) => s.setImage);
  const setSessions = useSessionStore((s) => s.setSessions);

  const [step, setStep] = useState<1 | 2>(1);
  const [file, setFile] = useState<File | null>(initialFile || null);
  const [gsd, setGsd] = useState("");

  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // ⭐ Auto-advance to Step 2 when a screenshot is sent
  useEffect(() => {
    if (initialFile) {
      setFile(initialFile);
      setStep(2);
    }
  }, [initialFile]);

  if (!open) return null;

  // -----------------------------
  // Step 1 → File selected
  // -----------------------------
  function handleFileSelect(f: File) {
    setFile(f);
    setStep(2);
  }

  // -----------------------------
  // Validation for dimensions
  // -----------------------------
  function isValidInput() {
    const g = Number(gsd);
    return g >= 0; // allow 0 also
  }

  // -----------------------------
  // Upload to Supabase
  // -----------------------------
  async function handleNext() {
    if (!file || !isValidInput()) return;

    setUploadError(null);
    setUploading(true);

    // ⭐ Redirect to loading animation screen
    router.replace("/loading-screen");

    try {
      const { path } = await uploadFile(file);

      let publicUrl = getPublicUrl(path);
      if (!publicUrl) publicUrl = await createSignedUrl(path, 3600);

      const lengthNum = Number(gsd) || 0;

      const { data: inserted } = await supabase
        .from("images")
        .insert({
          filename: file.name,
          width_km: 0,
          height_km: lengthNum,
          image_url: publicUrl,
        })
        .select()
        .single();

      const parentId = inserted.id;

      useDetectionStore.getState().clearDetections();

      setImage({
        parentId,
        imageUrl: publicUrl,
        widthKm: 0,
        heightKm: lengthNum,
        filename: file.name,
      });

      await supabase.from("chat_sessions").insert({
        id: parentId,
        title: file.name,
        image_url: publicUrl,
        width_km: 0,
        height_km: lengthNum,
      });

      const { data: latest } = await supabase
        .from("chat_sessions")
        .select("*")
        .order("created_at", { ascending: false });

      if (latest) setSessions(latest);

      // ⭐ Finally go to chat
      router.replace("/chat");
    } catch (err: any) {
      setUploadError(err.message || "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black/40 backdrop-blur-sm z-50">
      <div className="w-[520px] bg-[#061C35] rounded-3xl px-8 py-10 shadow-xl">
        <h2 className="text-center text-2xl font-semibold text-white mb-6">
          Upload Map
        </h2>

        {/* -------------------------
            STEP 1: file select
        -------------------------- */}
        {step === 1 && (
          <>
            <UploadDropzone onFileSelect={handleFileSelect} />

            <div className="flex justify-between mt-10">
              <button onClick={onClose} className="text-gray-300">
                Close
              </button>

              <button
                onClick={handleNext}
                disabled={!isValidInput() || uploading}
                className={`
    px-6 py-2 rounded-xl bg-white text-black font-medium 
    transition-all duration-200
    ${
      !isValidInput() || uploading
        ? "opacity-40 cursor-not-allowed"
        : "hover:bg-gray-100 hover:scale-[1.03] hover:shadow-lg cursor-pointer"
    }
  `}
              >
                {uploading ? "Uploading..." : "Next"}
              </button>
            </div>
          </>
        )}

        {/* -------------------------
            STEP 2: preview + input
        -------------------------- */}
        {step === 2 && file && (
          <>
            <UploadFilePreview file={file} onRemove={() => setStep(1)} />

            <div className="mt-6">
              <label className="text-sm text-gray-300">
                Ground Sample Distance (GSD)
              </label>
              <input
                type="number"
                value={gsd}
                onChange={(e) => setGsd(e.target.value)}
                placeholder="Enter GSD or leave empty for 0"
                className="bg-[#0F1F33] w-full px-4 py-3 rounded-xl text-gray-200 mt-2"
              />
            </div>

            <p className="text-gray-400 text-xs mt-3">
              GSD indicates the real-world distance between pixel centers in
            </p>

            <div className="flex justify-between items-center mt-10">
              <button onClick={onClose} className="text-gray-300">
                Close
              </button>

              <button
                onClick={handleNext}
                disabled={!isValidInput() || uploading}
                className="px-6 py-2 rounded-xl bg-white text-black transition"
              >
                {uploading ? "Uploading..." : "Next"}
              </button>
            </div>

            {uploadError && (
              <p className="text-sm text-red-400 mt-4">{uploadError}</p>
            )}
          </>
        )}
      </div>
    </div>
  );
}
