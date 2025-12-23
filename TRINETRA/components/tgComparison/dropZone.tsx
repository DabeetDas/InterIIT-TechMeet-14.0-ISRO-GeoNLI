import { useState } from "react";
import { useDropzone } from "react-dropzone";

interface DropZoneProps {
  onFile: (file: File, gsd: number) => void;
  label: string;
}

export function DropZoneReact({ onFile, label }: DropZoneProps) {
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [gsdInput, setGsdInput] = useState<string>("");
  const [showGsdPopup, setShowGsdPopup] = useState(false);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { "image/*": [] },
    onDrop: (files: File[]) => {
      if (!files[0]) return;
      setPendingFile(files[0]);
      setGsdInput("");
      setShowGsdPopup(true);
    },
  });

  function handleConfirm() {
    if (!pendingFile) return;

    const parsed = Number(gsdInput);
    const gsd = isNaN(parsed) ? 0 : parsed;

    onFile(pendingFile, gsd);
    setPendingFile(null);
    setShowGsdPopup(false);
    setGsdInput("");
  }

  function handleSkip() {
    if (!pendingFile) return;

    onFile(pendingFile, 0);
    setPendingFile(null);
    setShowGsdPopup(false);
    setGsdInput("");
  }

  return (
    <>
      {/* DROP ZONE */}
      <div
        {...getRootProps()}
        role="button"
        aria-label={label}
        className={`w-full min-h-[260px] bg-[#0F1F33] border rounded-2xl flex flex-col 
        items-center justify-center cursor-pointer text-gray-200 transition-all shadow-sm
        hover:scale-[1.01] transform-gpu p-6 box-border
        ${
          isDragActive
            ? "border-blue-400 bg-blue-900/20 ring-2 ring-blue-500/20"
            : "border-[#2A3B51]"
        }`}
      >
        <input {...getInputProps()} />

        <div className="flex flex-col items-center gap-3">
          <div className="text-gray-400 p-2 rounded-md">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="currentColor"
              className="w-10 h-10"
              aria-hidden="true"
            >
              <path d="M10.414 6L12 7.586 13.586 6H19a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h5.414z" />
            </svg>
          </div>

          <div className="text-2xl font-semibold text-gray-100">{label}</div>
          <div className="text-sm text-gray-400">Drop or click to upload</div>
        </div>
      </div>

      {/* SIMPLE GSD POPUP */}
      {showGsdPopup && (
        <div className="fixed inset-0 z-50 flex items-center justify-center backdrop-blur-sm bg-black/50">
          <div
            className="bg-[#0B1120] border border-[#1f2937] rounded-2xl p-6 
                 w-[90%] max-w-md shadow-xl 
                 flex flex-col gap-4"
          >
            <div>
              <h2 className="text-gray-100 text-lg font-semibold">
                Ground Sampling Distance
              </h2>
              {/* <p className="text-gray-400 text-sm mt-1 leading-relaxed">
                Enter GSD for this image (optional). If left empty, it will be
                set to <span className="text-gray-200 font-semibold">0</span>.
              </p> */}
            </div>

            <input
              type="number"
              value={gsdInput}
              onChange={(e) => setGsdInput(e.target.value)}
              placeholder="e.g. 0.5"
              className="w-full rounded-xl bg-[#0F1A2F] border border-[#334155] text-gray-200
                   px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/40"
            />

            <div className="flex justify-end gap-2 mt-2">
              <button
                type="button"
                onClick={handleSkip}
                className="px-3 py-1.5 text-sm rounded-xl border border-[#374151] text-gray-300"
              >
                Skip (use 0)
              </button>

              <button
                type="button"
                onClick={handleConfirm}
                className="px-3 py-1.5 text-sm rounded-xl bg-white text-black font-medium"
              >
                Continue
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
