"use client";

import { create } from "zustand";
import type { ChatMessageForModel } from "@/types/grounding/conversation";

interface ConversationState {
  messages: ChatMessageForModel[];
  addMessage: (msg: ChatMessageForModel) => void;
  setMessages: (all: ChatMessageForModel[]) => void;
  reset: () => void;
}

export const useConversationStore = create<ConversationState>((set) => ({
  messages: [],

  addMessage: (msg) =>
    set((state) => ({
      messages: [...state.messages, msg],
    })),

  setMessages: (all) => set({ messages: all }),

  reset: () => set({ messages: [] }),
}));
