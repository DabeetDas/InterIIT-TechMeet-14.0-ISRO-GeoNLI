"use client";

import ReactMarkdown from "react-markdown";
import { useState, useEffect } from "react";
import { motion } from "framer-motion";

import { supabase } from "@/lib/supabaseClient";
import { getSessionId } from "@/lib/sessionId";
import { MessagesSquare } from "lucide-react";
import { prettyLog } from "@/lib/devLogger";
import { runAnalysis } from "@/lib/apiRunAnalysis";

import { useChatStore } from "@/stores/chatStore";
import { useConversationStore } from "@/stores/conversationStore";
import { useHistoryStore } from "@/stores/forHistory/history";
import { useImageStore } from "@/stores/imageStore";
import { useAnalysisStore } from "@/stores/apiCallAnalysisStore";

import TypingLoader from "./TypingLoader";
import GraphMessage from "./GraphMessage";

import { toast } from "sonner";
import type { Message } from "@/stores/chatStore";
import type { ChatMessageForModel } from "@/types/grounding/conversation";

let bannerShown = false;

export default function ChatSection() {
  const { messages, addMessage, updateMessage, setLoading, reset } =
    useChatStore();

  const loading = useChatStore((s) => s.loading);

  const addConv = useConversationStore((s) => s.addMessage);
  const addHistory = useHistoryStore((s) => s.addToHistory);

  const [input, setInput] = useState("");
  const [showBanner, setShowBanner] = useState(true);
  const [showSuggestions, setShowSuggestions] = useState(true);

  // 🔥 Toast on first load
  useEffect(() => {
    if (bannerShown) return;
    bannerShown = true;

    requestAnimationFrame(() => {
      toast.custom((t) => (
        <div
          className="
            flex items-center gap-4
            rounded-xl border border-white/10 
            bg-[#111418]/95 backdrop-blur-sm
            px-5 py-4
            shadow-[0_0_25px_-5px_rgba(0,0,0,0.6)]
            text-gray-200 text-sm
            max-w-[360px]
            animate-in fade-in slide-in-from-top-4
          "
        >
          <p className="text-gray-300 leading-relaxed">
            Want crop-focused satellite analysis?{" "}
            <span className="text-blue-400 font-medium">Try AgriChat.</span>
          </p>

          <button
            onClick={() => toast.dismiss(t)}
            className="
              shrink-0
              px-3 py-1.5 
              rounded-lg 
              bg-[#1e2533] 
              text-gray-200 text-xs font-medium
              border border-white/10
              hover:bg-[#273040] hover:border-white/20
              transition-all duration-150
            "
          >
            OK
          </button>
        </div>
      ));
    });
  }, []);

  // Reset store on unmount
  useEffect(() => {
    return () => reset();
  }, [reset]);

  // 👇 Hide suggestions instantly when typing starts
  useEffect(() => {
    if (input.length > 0 && showSuggestions) {
      setShowSuggestions(false);
    }
  }, [input, showSuggestions]);

  async function sendMessage() {
    if (loading) return;
    if (!input.trim()) return;

    const img = useImageStore.getState();
    if (!img.imageUrl) return;

    const sessionId = getSessionId();

    // 🚀 Make UI instantly responsive
    setLoading(true);
    setShowBanner(false);
    setShowSuggestions(false);

    // USER message → UI
    const uiUser: Message = {
      id: crypto.randomUUID(),
      role: "user",
      text: input,
    };
    addMessage(uiUser);

    const textToSend = input;
    setInput(""); // instant clear

    // MODEL PAYLOAD
    const modelUser: ChatMessageForModel = {
      role: "user",
      content: [
        { type: "image", image: img.imageUrl! },
        { type: "dimensions", height: img.heightKm!, width: img.widthKm! },
        { type: "text", text: textToSend },
      ],
    };

    addConv(modelUser);
    addHistory(modelUser);

    // Save user message (async, non-blocking)
    supabase.from("chat_history").insert({
      session_id: sessionId,
      role: "user",
      content: { text: textToSend },
    });

    // Auto-title first message
    const userMsgs = useHistoryStore
      .getState()
      .history.filter((m) => m.role === "user");
    if (userMsgs.length === 0) {
      const title =
        textToSend.length > 60
          ? textToSend.slice(0, 57).trimEnd() + "..."
          : textToSend;

      supabase.from("chat_sessions").update({ title }).eq("id", sessionId);
    }

    // TEMP assistant bubble
    const assistantId = crypto.randomUUID();
    addMessage({
      id: assistantId,
      role: "assistant",
      text: "loading",
      graph: null,
    });

    // 🔥 Run model
    const payload = { conversation: useHistoryStore.getState().history };
    prettyLog("🔥 PAYLOAD", payload);

    await runAnalysis(payload);
    const result = useAnalysisStore.getState().result;

    // Update assistant message
    updateMessage(assistantId, {
      text: result?.answer ?? "No answer received.",
      graph: result?.graph ?? null,
    });

    // Save assistant output
    if (result?.answer) {
      supabase.from("chat_history").insert({
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
    <div className="flex-1 bg-[#111417] rounded-2xl p-6 flex flex-col gap-4 overflow-hidden">
      {/* HEADER */}
      <div className="text-gray-200 text-lg font-medium flex items-center gap-2">
        <MessagesSquare className="w-5 h-5" />
        Chat
      </div>

      {/* Suggested Prompts */}
      {showSuggestions && (
        <div className="flex flex-wrap gap-2 mt-1 mb-1">
          {[
            "Describe the image in detail?",
            "What kind of location is this?",
            "What is major attraction in the image?",
          ].map((prompt) => (
            <button
              key={prompt}
              onClick={async () => {
                if (loading) return;
                setShowSuggestions(false);
                setInput(prompt);
                await new Promise((r) => setTimeout(r, 0));
                sendMessage();
              }}
              className="
                text-xs text-gray-300 
                bg-[#1a1f27] 
                border border-white/10
                px-3 py-1.5 rounded-lg 
                hover:bg-[#232a36] hover:border-white/20
                transition
              "
            >
              {prompt}
            </button>
          ))}
        </div>
      )}

      {/* CHAT MESSAGES */}
      <div className="flex-1 overflow-y-auto pr-1 space-y-4 mt-2">
        {messages.map((msg) => (
          <motion.div
            key={msg.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`
              p-4 rounded-xl max-w-[75%] border transition-all duration-150
              ${
                msg.role === "user"
                  ? "bg-[#1f2a3d] border-[#2a3550] ml-auto text-gray-100"
                  : "bg-[#171b22] border-[#2a2e33] mr-auto text-gray-200"
              }
            `}
          >
            {msg.text === "loading" ? (
              <TypingLoader />
            ) : (
              <div className="prose prose-invert">
                <ReactMarkdown>{msg.text}</ReactMarkdown>
              </div>
            )}

            {msg.graph && <GraphMessage graph={msg.graph} />}
          </motion.div>
        ))}
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
          placeholder="How can I help you, today?"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (loading) return;
            if (e.key === "Enter") sendMessage();
          }}
        />

        <button
          onClick={sendMessage}
          disabled={loading}
          className={`
            p-3 rounded-xl transition flex items-center justify-center mr-1
            border border-black/20
            ${
              loading
                ? "bg-[#4F7CFF]/40 cursor-not-allowed"
                : "bg-[#4F7CFF] hover:bg-[#3d6df0]"
            }
          `}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-4 w-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="black"
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
