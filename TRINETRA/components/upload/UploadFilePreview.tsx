"use client";

import { X, Check } from "lucide-react";

export default function UploadFilePreview({
  file,
  onRemove,
}: {
  file: File;
  onRemove: () => void;
}) {
  return (
    <div className="flex items-center bg-[#0F1F33] border border-[#2b3643] rounded-2xl px-4 py-3 gap-4">
      {/* Yellow Thumbnail */}
      <div className="w-12 h-12 rounded-lg bg-yellow-400 flex items-center justify-center text-black font-bold">
        📄
      </div>

      {/* Name + size */}
      <div className="flex-1 text-gray-200">
        <p className="font-medium">{file.name}</p>
        <p className="text-sm text-gray-400">
          {(file.size / 1024).toFixed(1)} kb
        </p>
      </div>

      {/* Tick */}
      <Check className="text-green-400" />

      {/* Remove */}
      <button onClick={onRemove}>
        <X className="text-gray-400 hover:text-red-400" />
      </button>
    </div>
  );
}
