import { create } from "zustand";

interface AnalysisState {
  uploadedImage: string | null;
  setUploadedImage: (url: string) => void;
  clearImage: () => void;
}

export const useAnalysisStore = create<AnalysisState>((set) => ({
  uploadedImage: null,

  // Save image URL
  setUploadedImage: (url) => set({ uploadedImage: url }),

  // Remove image (optional)
  clearImage: () => set({ uploadedImage: null }),
}));
