// src/stores/imageMetadataStore.ts
"use client";

import { create } from "zustand";

interface ImageMetadataState {
  imageClass: string | null;
  setImageClass: (cls: string | null) => void;
  clearImageClass: () => void;
}

export const useImageMetadataStore = create<ImageMetadataState>((set) => ({
  imageClass: null,

  setImageClass: (cls) => set({ imageClass: cls }),

  clearImageClass: () => set({ imageClass: null }),
}));
