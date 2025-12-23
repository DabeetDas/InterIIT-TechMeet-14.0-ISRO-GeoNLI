export type ChatContentItem =
  | { type: "text"; text: string }
  | { type: "image"; image: string }
  | { type: "dimensions"; height: number; width: number }
  | { type: "image1"; image: string }
  | { type: "image2"; image: string };

export type ChatMessageForModel = {
  role: "user" | "assistant";
  content: ChatContentItem[];
};
