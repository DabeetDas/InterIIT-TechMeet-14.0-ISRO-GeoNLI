"use client";

import ReactMarkdown from "react-markdown";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

import { supabase } from "@/lib/supabaseClient";
import { getSessionId } from "@/lib/sessionId";

import { prettyLog } from "@/lib/devLogger";
import { runAnalysis as runAnalysisAgri } from "@/lib/apiRunAnalysisAgriChat";

import { useChatStore } from "@/stores/chatStore";
import { useConversationStore } from "@/stores/conversationStore";
import { useHistoryStore } from "@/stores/forHistory/history";
import { useImageStore } from "@/stores/imageStore";
import { useAnalysisStore } from "@/stores/apiCallAnalysisStore";
import { useDetectionStore } from "@/stores/useDetectionStore";

import TypingLoader from "./TypingLoader";
import GraphMessage from "./GraphMessage";
import { MessagesSquare } from "lucide-react";

import type { Message } from "@/stores/chatStore";
import type { ChatMessageForModel } from "@/types/grounding/conversation";

export default function ChatSectionAgriChat() {
  const { messages, addMessage, updateMessage, setLoading, reset } =
    useChatStore();

  const addConv = useConversationStore((s) => s.addMessage);
  const addHistory = useHistoryStore((s) => s.addToHistory);

  const clearDetections = useDetectionStore((s) => s.clearDetections);
  const mergeDetections = useDetectionStore((s) => s.mergeDetections);

  const [input, setInput] = useState("");

  useEffect(() => {
    return () => reset();
  }, [reset]);

  async function sendMessage() {
    if (!input.trim()) return;

    const img = useImageStore.getState();
    if (!img.imageUrl) return;

    const sessionId = getSessionId();

    clearDetections();

    // 1. UI
    const uiUser: Message = {
      id: crypto.randomUUID(),
      role: "user",
      text: input,
    };
    addMessage(uiUser);

    // 2. MODEL
    const modelUser: ChatMessageForModel = {
      role: "user",
      content: [
        { type: "image", image: img.imageUrl! },
        { type: "dimensions", height: img.heightKm!, width: img.widthKm! },
        { type: "text", text: input },
      ],
    };
    addConv(modelUser);
    addHistory(modelUser);

    // 3. DB Store
    await supabase.from("chat_history").insert({
      session_id: sessionId,
      role: "user",
      content: { text: input },
    });

    // 4. Auto-title first message
    const userMsgs = useHistoryStore
      .getState()
      .history.filter((m) => m.role === "user");

    if (userMsgs.length === 1) {
      const title =
        input.length > 60 ? input.slice(0, 57).trimEnd() + "..." : input;

      await supabase
        .from("chat_sessions")
        .update({ title })
        .eq("id", sessionId);
    }

    // 5. Start Loading
    setInput("");
    setLoading(true);

    const assistantId = crypto.randomUUID();
    addMessage({
      id: assistantId,
      role: "assistant",
      text: "loading",
      graph: null,
    });

    // 6. Run Agri Model
    const payload = {
      conversation: useHistoryStore.getState().history,
    };

    prettyLog("🌾 AGRI PAYLOAD", payload);

    await runAnalysisAgri(payload);
    const result = useAnalysisStore.getState().result;

    // 7. Update UI
    updateMessage(assistantId, {
      text: result?.answer ?? "No answer received.",
      graph: result?.graph ?? null,
    });

    // 8. Add Detections
    if (result?.groundingData) {
      mergeDetections(result.groundingData);
    }

    // 9. Save assistant msg
    if (result?.answer) {
      await supabase.from("chat_history").insert({
        session_id: sessionId,
        role: "assistant",
        content: { text: result.answer },
      });

      addHistory({
        role: "assistant",
        content: [{ type: "text", text: result.answer }],
      });
    }

    setLoading(false);
  }

  return (
    <div
      className="
      flex-1 bg-[#111417] rounded-2xl p-6 flex flex-col gap-4 
      overflow-hidden border border-[#1f1f1f]
    "
    >
      {/* HEADER */}
      <div className="flex flex-col mb-1">
        <div className="flex items-center gap-2">
          <MessagesSquare className="w-5 h-5 text-green-500" />
          <span className="text-gray-200 text-lg font-semibold tracking-wide">
            AgriChat{" "}
            <span className="text-green-400 font-normal">
              (agriculture-tuned)
            </span>
          </span>
        </div>

        <p className="text-[12px] text-gray-400 mt-1 leading-relaxed">
          Ask about crop analysis, soil health, and weather-driven impacts —
          responses are tailored for agriculture.
        </p>
      </div>

      {/* CHAT MESSAGES */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-2 mt-2">
        <AnimatePresence>
          {messages.map((msg) => (
            <motion.div
              key={msg.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.15 }}
              className={`
                p-4 rounded-xl border max-w-[75%]
                ${
                  msg.role === "user"
                    ? "bg-[#1f2a3d] border-[#2a3550] text-gray-100 ml-auto"
                    : "bg-[#171b22] border-[#2a2e33] text-gray-200 mr-auto"
                }
              `}
            >
              {msg.text === "loading" ? (
                <TypingLoader />
              ) : (
                <div className="prose prose-invert max-w-none">
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                </div>
              )}

              {msg.graph && <GraphMessage graph={msg.graph} />}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* INPUT BAR */}
      <div
        className="
        w-full 
        bg-[#1f2937] 
        rounded-2xl 
        flex items-center 
        px-2 py-2
        border border-black/30
      "
      >
        <input
          className="
          flex-1 
          bg-[#1f2937] 
          text-gray-200 
          placeholder-gray-400 
          px-4 py-2
          rounded-xl
          focus:outline-none
        "
          placeholder="Make a query"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />

        {/* KEEP ARROW GREEN */}
        <button
          onClick={sendMessage}
          className="
          bg-green-600
          hover:bg-green-700
          p-3 
          rounded-xl
          transition
          flex items-center justify-center
          mr-1
        "
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-4 w-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="white"
            strokeWidth="2"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M5 12h14M12 5l7 7-7 7"
            />
          </svg>
        </button>
      </div>
    </div>
  );
}
