import { GoogleGenerativeAI } from "@google/generative-ai";

export async function POST(req: Request) {
  try {
    const { prompt } = await req.json();

    const ai = new GoogleGenerativeAI(process.env.GENERATIVE_AI_API_KEY!);

    const model = ai.getGenerativeModel({
      model: "gemini-2.0-flash",
    });

    // Start the streaming request
    const result = await model.generateContentStream({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
    });

    console.log("result of the streaming resoponse:", result);
    // Create a stream for Next.js response
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        for await (const item of result.stream) {
          const text = item.text();
          controller.enqueue(encoder.encode(text));
        }
        controller.close();
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
      },
    });
  } catch (error) {
    console.error("/api/generate-stream error:", error);
    return new Response("Internal Server Error", { status: 500 });
  }
}
