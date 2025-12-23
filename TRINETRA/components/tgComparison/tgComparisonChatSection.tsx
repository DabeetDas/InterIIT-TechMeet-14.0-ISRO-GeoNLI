"use client";

/**
 * ChatSection handles:
 * - Rendering chat messages
 * - Sending user text input
 * - Building the model-ready payload
 * - Triggering /api/test/analysisApi
 * - Storing chat history in Supabase
 * - Tracking TG Comparison images + GSD
 */

import ReactMarkdown from "react-markdown";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

import { supabase } from "@/lib/supabaseClient";
import { prettyLog } from "@/lib/devLogger";
import TypingLoader from "../TypingLoader";

// Zustand Stores
import { useAnalysisStore } from "@/stores/apiCallAnalysisStore";
import { useTgComparisonStore } from "@/stores/tgComparison/useTgComparisonStore";
import { useConversationStore } from "@/stores/conversationStore";
import { useChatStore } from "@/stores/chatStore";
import { useHistoryStore } from "@/stores/forHistory/history";

// Components
import GraphMessage from "../GraphMessage";

// Helpers
// import { runAnalysis } from "@/lib/apiRunAnalysis";
import { getSessionId } from "@/lib/sessionId";
import { MessagesSquare } from "lucide-react";
import { runAnalysis } from "@/lib/tgCompare/tgCompareApiRunAnalysis";

// Types
import type { Message } from "@/stores/chatStore";
import type { ChatMessageForModel } from "@/types/grounding/conversation";

export default function ChatSection() {
  const { messages, addMessage, updateMessage, setLoading, reset } =
    useChatStore();

  const addConv = useConversationStore((s) => s.addMessage);
  const addHistory = useHistoryStore((s) => s.addToHistory);

  const { tgComparisonStore: tg } = useTgComparisonStore();

  const [input, setInput] = useState("");
  const [showBanner, setShowBanner] = useState(true);

  const disabled = !tg?.image1_url || !tg?.image2_url || tg.image1_url === null;

  useEffect(() => {
    return () => reset();
  }, [reset]);

  async function sendMessage() {
    console.log("🔥 sendMessage CALLED");
    console.log("📥 input =", input);
    console.log("🟦 disabled =", disabled);
    console.log("🗂 TG Store =", tg);

    if (!input.trim()) {
      console.log("❌ BLOCKED: empty input");
      return;
    }

    if (disabled) {
      console.log("❌ BLOCKED: Chat is disabled (missing images)");
      return;
    }

    console.log("✅ Sending message...");

    const sessionId = getSessionId();
    setShowBanner(false);

    /**
     * 1. UI — Add user message
     */
    const uiUser: Message = {
      id: crypto.randomUUID(),
      role: "user",
      text: input,
    };
    addMessage(uiUser);

    /**
     * 2. MODEL — Build message for AI pipeline
     */
    const modelUser: ChatMessageForModel = {
      role: "user",
      content: [
        { type: "image1", image: tg.image1_url ?? "" },
        { type: "image2", image: tg.image2_url ?? "" },
        {
          type: "dimensions",
          height: tg.image1_gsd ?? 0,
          width: tg.image2_gsd ?? 0,
        },
        { type: "text", text: input },
      ],
    };

    prettyLog("📝 MODEL USER MSG", modelUser);

    addConv(modelUser);
    addHistory(modelUser);

    /**
     * 3. DB — Store user message
     */
    await supabase.from("chat_history").insert({
      session_id: sessionId,
      role: "user",
      content: { text: input },
    });

    /**
     * 4. Auto-title first message
     */
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

    /**
     * 5. UI — Assistant placeholder
     */
    setInput("");
    setLoading(true);

    const assistantId = crypto.randomUUID();
    addMessage({
      id: assistantId,
      role: "assistant",
      text: "loading",
      graph: null,
    });

    /**
     * 6. Build full conversation payload
     */
    const payload = {
      conversation: useHistoryStore.getState().history,
    };

    // prettyLog("📦 PAYLOAD", payload);

    /**
     * 7. AI Request
     */
    await runAnalysis(payload);
    const result = useAnalysisStore.getState().result;

    /**
     * 8. Update assistant's placeholder
     */
    updateMessage(assistantId, {
      text: result?.answer ?? "No answer received.",
      graph: result?.graph ?? null,
    });

    /**
     * 9. DB — Store assistant message
     */
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

  /**
   * -----------------------------------------
   * Component Render
   * -----------------------------------------
   */
  return (
    <div className="flex-1 bg-[#0f1624] rounded-2xl p-6 flex flex-col gap-4 overflow-hidden">
      {/* Section Header */}
      <div className="text-gray-200 text-lg font-medium flex items-center gap-2">
        <MessagesSquare className="w-5 h-5" />
        Chat
      </div>

      {/* Intro Banner */}
      <AnimatePresence>
        {showBanner && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.25 }}
            className="
             rounded-xl border border-white/5 
             bg-[#15181e]/80 backdrop-blur-sm
             px-5 py-4 
             flex items-center justify-between
             shadow-[0_0_20px_-5px_rgba(0,0,0,0.6)]
           "
          >
            <p className="text-gray-200 text-sm tracking-wide">
              Want crop-focused satellite analysis?{" "}
              <span className="text-blue-400 font-medium">Try AgriChat.</span>
            </p>

            <button
              onClick={() => setShowBanner(false)}
              className="
               px-4 py-1.5 rounded-lg 
               bg-[#1e2533] text-gray-200 text-sm
               border border-white/10
               hover:bg-[#273040] hover:border-white/20
               transition-all duration-150
             "
            >
              Ok
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Messages List */}
      <div className="flex-1 overflow-y-auto pr-1 space-y-4 mt-2">
        {messages.map((msg) => (
          <motion.div
            key={msg.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`p-4 rounded-xl ${
              msg.role === "user"
                ? "bg-[#1f2a3d] ml-auto max-w-[75%]"
                : "bg-[#1a2333] mr-auto max-w-[75%]"
            }`}
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

      {/* Input Field */}
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
          placeholder={
            disabled ? "Upload both images to start..." : "Make a query"
          }
          disabled={disabled}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (disabled) return;
            if (e.key === "Enter") sendMessage();
          }}
        />

        <button
          onClick={sendMessage}
          disabled={disabled}
          className={`
      p-3 rounded-xl transition flex items-center justify-center mr-1
      border border-black/20
      ${
        disabled
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
