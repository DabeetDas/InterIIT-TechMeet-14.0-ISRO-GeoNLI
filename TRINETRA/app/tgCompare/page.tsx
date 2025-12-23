import ChatSection from "@/components/tgComparison/tgComparisonChatSection";
import TgComparisonAnalysisSection from "@/components/tgComparison/tgComparisonAnalysisSection";

export default function TGComparisonChat() {
  return (
    <div className="w-full flex justify-center mt-2">
      <div className="w-full h-[calc(100vh-75px)] flex gap-6">
        <ChatSection />
        <TgComparisonAnalysisSection />
      </div>
    </div>
  );
}
