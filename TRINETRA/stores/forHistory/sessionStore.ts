"use client";

import { create } from "zustand";

export interface SessionItem {
  id: string;
  title: string;
  created_at: string; // coming from Supabase
  image_url: string | null;
  width_km: number | null;
  height_km: number | null;
}

interface SessionState {
  sessions: SessionItem[];
  setSessions: (all: SessionItem[]) => void;
  addSession: (s: SessionItem) => void;
  clearSessions: () => void;
}

export const useSessionStore = create<SessionState>((set) => ({
  sessions: [],
  setSessions: (all) => set({ sessions: all }),
  addSession: (s) =>
    set((state) => ({
      sessions: [...state.sessions, s],
    })),
  clearSessions: () => set({ sessions: [] }),
}));
