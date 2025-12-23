"use client";

import { create } from "zustand";

export interface tgComparisonStore {
  id: string;
  image1_url: string | null;
  image2_url: string | null;
  image1_gsd: number | null;
  image2_gsd: number | null;
  created_at: string;
}

interface tgComparisonStoreState {
  tgComparisonStore: tgComparisonStore | null;
  setTgComparisonStore: (p: tgComparisonStore | null) => void;
  clearTgComparisonStore: () => void;
}

export const useTgComparisonStore = create<tgComparisonStoreState>((set) => ({
  tgComparisonStore: null,

  setTgComparisonStore: (p) => set({ tgComparisonStore: p }),

  clearTgComparisonStore: () => set({ tgComparisonStore: null }),
}));
