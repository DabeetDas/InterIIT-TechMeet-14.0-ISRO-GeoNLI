import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const { conversation } = await req.json();

    const result = await fetch(
      process.env.NEXT_PUBLIC_BACKEND_URL + "/analyze",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation }),
      }
    );

    const data = await result.json();

    return NextResponse.json(data);
  } catch (e) {
    return NextResponse.json(
      { error: "Failed to process request" },
      { status: 500 }
    );
  }
}
