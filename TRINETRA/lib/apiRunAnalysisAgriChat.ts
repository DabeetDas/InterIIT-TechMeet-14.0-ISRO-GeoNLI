// src/lib/apiRunAnalysis.ts
import { useAnalysisStore } from "@/stores/apiCallAnalysisStore";
import { useDetectionStore } from "@/stores/useDetectionStore";
import type { ChatMessageForModel } from "@/types/grounding/conversation";

export async function runAnalysis(payload: {
  conversation: ChatMessageForModel[];
}) {
  const analysisStore = useAnalysisStore.getState();
  const detectionStore = useDetectionStore.getState();

  analysisStore.setLoading(true);
  analysisStore.setError(null);
  console.log("🚀 Running analysis with payload:", payload);
  console.log(JSON.stringify(payload));

  try {
    const res = await fetch("/api/agriculture/analysisApi", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error("Request failed");
    const data = await res.json();
    console.log("✅ Analysis response data:", data);

    // -----------------------------------------
    // Normalize groundingData (backend returns array)
    // -----------------------------------------
    const normalizedGrounding =
      Array.isArray(data.groundingData) && data.groundingData.length > 0
        ? data.groundingData[0] // first object only
        : null;

    // -----------------------------------------
    // Store analysis result
    // -----------------------------------------
    analysisStore.setResult({
      answer: data.answer ?? null,
      graph: data.graph ?? null,
      groundingData: normalizedGrounding,
    });

    // -----------------------------------------
    // Merge detections into detection store
    // -----------------------------------------
    if (normalizedGrounding) {
      detectionStore.mergeDetections(normalizedGrounding);
    }
  } catch (err: any) {
    console.error("❌ runAnalysis error:", err);
    analysisStore.setError(err.message || "Unknown error");
  } finally {
    analysisStore.setLoading(false);
  }
}
