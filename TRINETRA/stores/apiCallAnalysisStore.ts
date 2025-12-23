"use client";

import { create } from "zustand";
import type { AnalysisResult } from "@/types/grounding/ApiDataanalysis";

interface AnalysisStore {
  result: AnalysisResult | null;
  loading: boolean;
  error: string | null;

  setResult: (res: AnalysisResult | null) => void;
  setLoading: (bool: boolean) => void;
  setError: (msg: string | null) => void;
  reset: () => void;
}

export const useAnalysisStore = create<AnalysisStore>((set) => ({
  result: null,
  loading: false,
  error: null,

  setResult: (res) => set({ result: res }),
  setLoading: (bool) => set({ loading: bool }),
  setError: (msg) => set({ error: msg }),

  reset: () => set({ result: null, loading: false, error: null }),
}));
