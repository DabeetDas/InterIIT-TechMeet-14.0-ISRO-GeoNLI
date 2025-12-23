"use client";

import { useState } from "react";

export default function UploadDropzone({
  onFileSelect,
}: {
  onFileSelect: (file: File) => void;
}) {
  const [dragActive, setDragActive] = useState(false);

  // no confirmation badge — drop should immediately advance
  const [isValidDrag, setIsValidDrag] = useState<boolean>(false);

  function handleFile(file: File | null) {
    if (!file) return;
    // Immediately notify parent — parent will advance to preview
    onFileSelect(file);
  }

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    handleFile(e.currentTarget.files?.[0] || null);
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    setDragActive(true);
    // detect if the dragged item is an image so we can show a lighter blue
    try {
      const dt = e.dataTransfer;
      let valid = false;
      if (dt.files && dt.files.length > 0) {
        const f = dt.files[0];
        valid = !!(f && f.type && f.type.startsWith("image/"));
      } else if (dt.items && dt.items.length > 0) {
        const it = dt.items[0];
        valid = it.kind === "file" && (it.type || "").startsWith("image/");
      }
      setIsValidDrag(valid);
    } catch (e) {
      setIsValidDrag(false);
    }
  }

  function handleDragEnter(e: React.DragEvent) {
    e.preventDefault();
    setDragActive(true);
    // mirror the same detection logic here
    try {
      const dt = e.dataTransfer;
      let valid = false;
      if (dt.files && dt.files.length > 0) {
        const f = dt.files[0];
        valid = !!(f && f.type && f.type.startsWith("image/"));
      } else if (dt.items && dt.items.length > 0) {
        const it = dt.items[0];
        valid = it.kind === "file" && (it.type || "").startsWith("image/");
      }
      setIsValidDrag(valid);
    } catch (e) {
      setIsValidDrag(false);
    }
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault();
    setDragActive(false);
    setIsValidDrag(false);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    // Some browsers expose files on dataTransfer.files; others only expose items.
    let file: File | null = null;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      file = e.dataTransfer.files[0];
    } else if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      const item = e.dataTransfer.items[0];
      if (item.kind === "file" && typeof item.getAsFile === "function") {
        file = item.getAsFile();
      }
    }

    handleFile(file);
  }

  return (
    <div className="w-full">
      <div
        onDragOver={handleDragOver}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative w-full rounded-2xl py-10 text-center cursor-pointer transition-all duration-150 ease-in-out
          bg-[#0F1F33] border border-[#2A3B51] overflow-hidden
          ${dragActive ? "border-blue-400" : ""}`}
      >
        {/* Drag overlay: visible only while dragging */}
        <div
          aria-hidden={!dragActive}
          className={`pointer-events-none absolute inset-0 flex items-center justify-center rounded-2xl transition-opacity duration-150
                ${dragActive ? "opacity-100" : "opacity-0"}`}
        >
          <div
            className={`w-full h-full rounded-2xl backdrop-blur-sm border-4 border-dashed flex items-center justify-center
                ${
                  isValidDrag
                    ? "bg-blue-300/12 border-blue-300"
                    : "bg-red-600/6 border-red-400"
                }`}
          >
            <div className="text-white text-lg font-semibold select-none">
              {isValidDrag
                ? "Drop file to upload"
                : "Only image files are accepted"}
            </div>
          </div>
        </div>

        <label htmlFor="uploadImage" className="relative z-10">
          <div
            className={`flex flex-col gap-1 items-center justify-center ${
              dragActive ? "text-white" : "text-gray-200"
            }`}
          >
            <div className="text-2xl">📁</div>
            <div className="font-medium text-lg pointer-events-none">
              Drop files here
            </div>
            <span className="text-sm mt-1">
              or{" "}
              <span className="text-blue-400 underline pointer-events-auto">
                browse from device
              </span>
            </span>
          </div>
        </label>

        <input
          id="uploadImage"
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleChange}
        />
        {/* no inline confirmation — drop immediately advances to preview */}
      </div>
    </div>
  );
}
