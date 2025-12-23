// src/stores/useDetectionStore.ts
import { create } from "zustand";

export type Detection = {
  id: string;
  category: string;
  center: { x: number; y: number };
  size: { width: number; height: number };
  angle: number;
  score: number;
};

interface DetectionState {
  loading: boolean;

  currentCategory: string | null;
  categories: string[];
  data: Record<string, Detection[]>;

  setLoading: (val: boolean) => void;
  setCategory: (cat: string | null) => void;
  setDetections: (cat: string, detections: Detection[]) => void;
  mergeDetections: (payload: Record<string, Detection[]>) => void;

  clearDetections: () => void; // 👈 NEW
}

export const useDetectionStore = create<DetectionState>((set) => ({
  loading: false,

  currentCategory: null,
  categories: [],
  data: {},

  setLoading: (val) => set({ loading: val }),

  setCategory: (cat) => set({ currentCategory: cat }),

  setDetections: (cat, detections) =>
    set((state) => ({
      data: { ...state.data, [cat]: detections },
      categories: Array.from(new Set([...state.categories, cat])),
      currentCategory: state.currentCategory ?? cat,
    })),

  mergeDetections: (payload) =>
    set((state) => {
      const mergedData = { ...state.data, ...payload };
      const categories = Array.from(
        new Set([...state.categories, ...Object.keys(payload)])
      );

      const newCurrent = state.currentCategory ?? categories[0] ?? null;

      return {
        loading: false,
        data: mergedData,
        categories,
        currentCategory: newCurrent,
      };
    }),

  // ⭐ Clears everything when switching route
  clearDetections: () =>
    set({
      currentCategory: null,
      categories: [],
      data: {},
      loading: false,
    }),
}));
