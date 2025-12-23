"use client";

import { supabase } from "@/lib/supabaseClient";
import { useImageStore } from "@/stores/imageStore";
import { useChatStore } from "@/stores/chatStore";
import { useConversationStore } from "@/stores/conversationStore";
import { useHistoryStore } from "@/stores/forHistory/history";
import { useDetectionStore } from "@/stores/useDetectionStore";

export async function loadSession(sessionId: string) {
  // ⭐ CLEAR OLD DETECTIONS
  const clearDetections = useDetectionStore.getState().clearDetections;
  clearDetections();

  const setImage = useImageStore.getState().setImage;
  const setChat = useChatStore.getState().setMessages;
  const setConversation = useConversationStore.getState().setMessages;
  const setHistory = useHistoryStore.getState().setHistory;

  // 1. Load session metadata
  const { data: session } = await supabase
    .from("chat_sessions")
    .select("*")
    .eq("id", sessionId)
    .single();

  if (!session) return;

  // 2. Load history (safe default)
  const { data: historyRaw } = await supabase
    .from("chat_history")
    .select("*")
    .eq("session_id", sessionId)
    .order("created_at", { ascending: true });

  const history = historyRaw ?? [];

  // 3. Restore image
  setImage({
    parentId: session.id,
    imageUrl: session.image_url,
    widthKm: session.width_km,
    heightKm: session.height_km,
    filename: session.title ?? "Image",
  });

  // 4. Rebuild UI chat messages
  const chatMessages = history.map((h: any) => ({
    id: crypto.randomUUID(),
    role: h.role,
    text: h.content?.text || "",
    graph: h.content?.graph || null,
  }));

  setChat(chatMessages);

  // 5. Rebuild model conversation
  const modelMessages = history.map((h: any) => h.content);
  setConversation(modelMessages);

  // 6. Restore history store
  setHistory(modelMessages);
}
