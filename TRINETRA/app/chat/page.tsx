import ChatSection from "@/components/ChatSection";
import AnalysisSection from "@/components/AnalysisSection";

export default function ChatPage() {
  return (
    <div className="w-full flex justify-center mt-2">
      <div className="w-full h-[calc(100vh-75px)] flex gap-6">
        <ChatSection />
        <AnalysisSection />
      </div>
    </div>
  );
}
