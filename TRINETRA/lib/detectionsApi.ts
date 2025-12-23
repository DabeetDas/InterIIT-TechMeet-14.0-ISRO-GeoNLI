// src/lib/detectionsApi.ts
import type { Detection } from "@/stores/useDetectionStore";

export type DetectionsResponse = {
  detections: Record<string, Detection[]>;
};

export async function fetchDetectionsForImage(
  imageId: string
): Promise<DetectionsResponse> {
  // For now, you can mock this or call your real API
  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ imageId }),
  });

  if (!res.ok) {
    throw new Error("Failed to fetch detections");
  }

  return res.json();
}
