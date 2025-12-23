"use client";

import { create } from "zustand";
import type { GraphData } from "@/types/grounding/graphAnalysis";

export type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
  graph?: GraphData | null;
};

interface ChatState {
  messages: Message[];
  loading: boolean;

  addMessage: (msg: Message) => void;
  updateMessage: (id: string, update: Partial<Message>) => void;
  setLoading: (v: boolean) => void;
  setMessages: (msgs: Message[]) => void;
  reset: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  loading: false,

  addMessage: (m) => set((state) => ({ messages: [...state.messages, m] })),

  updateMessage: (id, data) =>
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === id ? { ...msg, ...data } : msg
      ),
    })),

  setMessages: (msgs) => set({ messages: msgs }), // ⭐ NEW

  setLoading: (v) => set({ loading: v }),

  reset: () => set({ messages: [], loading: false }),
}));
