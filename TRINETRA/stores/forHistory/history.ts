"use client";

import { create } from "zustand";
import type { ChatMessageForModel } from "@/types/grounding/conversation";

interface HistoryState {
  history: ChatMessageForModel[];

  addToHistory: (msg: ChatMessageForModel) => void;
  setHistory: (all: ChatMessageForModel[]) => void;
  clearHistory: () => void;
}

export const useHistoryStore = create<HistoryState>((set) => ({
  history: [],

  addToHistory: (msg) =>
    set((state) => ({
      history: [...state.history, msg],
    })),

  setHistory: (all) => set({ history: all }),

  clearHistory: () => set({ history: [] }),
}));

// export type ChatContentItem =
//   | { type: "text"; text: string }
//   | { type: "image"; image: string }
//   | { type: "dimensions"; height: number; width: number };

// export type ChatMessageForModel = {
//   role: "user" | "assistant";
//   content: ChatContentItem[];
// };
