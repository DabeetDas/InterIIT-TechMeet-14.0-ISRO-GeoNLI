"use client";

import { forwardRef, useImperativeHandle } from "react";

export interface CaptureButtonHandle {
  triggerCapture: () => Promise<void>;
}

interface Props {
  onCapture: (file: File) => void;
}

const CaptureButton = forwardRef<CaptureButtonHandle, Props>(
  ({ onCapture }, ref) => {
    async function handleCapture() {
      try {
        const stream = await navigator.mediaDevices.getDisplayMedia({
          video: true,
        });

        const video = document.createElement("video");
        video.style.position = "fixed";
        video.style.top = "-10000px";
        video.style.left = "-10000px";
        document.body.appendChild(video);

        video.srcObject = stream;

        await new Promise<void>((resolve) => {
          video.onloadedmetadata = () => {
            video.play().then(() => resolve());
          };
        });

        // Base canvas (full screenshot)
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext("2d");
        if (!ctx) {
          stream.getTracks().forEach((t) => t.stop());
          document.body.removeChild(video);
          return;
        }

        // Draw entire frame
        ctx.drawImage(video, 0, 0);

        // --- APPLY CROPPING ---

        const cropTop = 220; // remove 160px from the top
        const cropBottom = 200; // remove 150px from the bottom

        const croppedHeight = canvas.height - cropTop - cropBottom;

        const croppedCanvas = document.createElement("canvas");
        croppedCanvas.width = canvas.width;
        croppedCanvas.height = croppedHeight;

        const croppedCtx = croppedCanvas.getContext("2d")!;

        croppedCtx.drawImage(
          canvas,
          0,
          cropTop, // source x,y
          canvas.width,
          croppedHeight, // source w,h
          0,
          0, // destination x,y
          canvas.width,
          croppedHeight // destination w,h
        );

        // Cleanup
        stream.getTracks().forEach((t) => t.stop());
        document.body.removeChild(video);

        // Export as PNG
        const blob: Blob | null = await new Promise((resolve) =>
          croppedCanvas.toBlob(resolve, "image/png")
        );

        if (!blob) return;

        onCapture(new File([blob], "capture.png", { type: "image/png" }));
      } catch (err) {
        console.error("Screen capture failed:", err);
      }
    }

    // Expose triggerCapture() to parent
    useImperativeHandle(ref, () => ({
      triggerCapture: handleCapture,
    }));

    return null; // Component renders NOTHING
  }
);

export default CaptureButton;
