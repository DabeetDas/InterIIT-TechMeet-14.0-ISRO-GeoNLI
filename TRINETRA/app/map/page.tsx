"use client";

import { useRef, useState } from "react";
import MapContainer from "@/components/map/MapTilerMap";
import UploadModal from "@/components/upload/UploadModal";
import CaptureButton, {
  CaptureButtonHandle,
} from "@/components/capture/CaptureButton";
import { Upload, Crop } from "lucide-react";

export default function MapPage() {
  const [openUpload, setOpenUpload] = useState(false);
  const [capturedFile, setCapturedFile] = useState<File | null>(null);

  const captureRef = useRef<CaptureButtonHandle | null>(null);

  function handleCapture(file: File) {
    setCapturedFile(file);
    setOpenUpload(true);
  }

  function handleOpenEmptyUpload() {
    setCapturedFile(null);
    setOpenUpload(true);
  }

  return (
    <div className="relative w-full h-screen">
      <MapContainer />
      <div
        className="
    absolute bottom-6 left-1/2 -translate-x-1/2 z-50
    flex items-center gap-6 
    px-6 py-3
    rounded-2xl
    bg-white/70 backdrop-blur-md
    shadow-xl
    border border-white/40
  "
      >
        {/* Capture Button */}
        <button
          onClick={() => captureRef.current?.triggerCapture()}
          className="
      flex items-center gap-2 
      text-blue-600
      font-medium
      transition-all duration-200
      hover:text-blue-800
      hover:scale-[1.07]
      active:scale-[0.95]
    "
        >
          <Crop size={18} className="transition-colors" />
          Capture
        </button>

        <div className="w-px h-5 bg-black/10"></div>

        <button
          onClick={handleOpenEmptyUpload}
          className="
      flex items-center gap-2 
      text-blue-600 
      font-medium
      transition-all duration-200
      hover:text-blue-800
      hover:scale-[1.07]
      active:scale-[0.95]
    "
        >
          <Upload size={18} className="transition-colors" />
          Upload
        </button>
      </div>

      {/* Hidden capture logic component */}
      <CaptureButton ref={captureRef} onCapture={handleCapture} />

      <UploadModal
        open={openUpload}
        onClose={() => setOpenUpload(false)}
        initialFile={capturedFile}
      />
    </div>
  );
}
