import ChatSection from "@/components/ChatSectionAgriChat";
import AnalysisSection from "@/components/AnalysisSectionAgriChat";

export default function AgriChatPage() {
  return (
    <div className="w-full flex justify-center mt-2">
      <div className="w-full  h-[calc(100vh-75px)] flex gap-6">
        <ChatSection />
        <AnalysisSection />
      </div>
    </div>
  );
}
