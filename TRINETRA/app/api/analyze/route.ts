// src/app/api/analyze/route.ts
import { NextResponse } from "next/server";

// 👉 Mock detection data (placeholder for real model)
const mockDetections = {
  airplane: [
    {
      id: "plane_01",
      category: "airplane",
      center: { x: 209, y: 300 },
      size: { width: 100, height: 80 },
      angle: 15,
      score: 0.91,
    },
    {
      id: "plane_02",
      category: "airplane",
      center: { x: 450, y: 350 },
      size: { width: 10, height: 70 },
      angle: -8,
      score: 0.87,
    },
  ],

  cars: [
    {
      id: "car_01",
      category: "cars",
      center: { x: 320, y: 380 },
      size: { width: 120, height: 60 },
      angle: 8,
      score: 0.94,
    },
    {
      id: "car_02",
      category: "cars",
      center: { x: 520, y: 290 },
      size: { width: 110, height: 55 },
      angle: -12,
      score: 0.88,
    },
    {
      id: "car_03",
      category: "cars",
      center: { x: 680, y: 450 },
      size: { width: 130, height: 70 },
      angle: 5,
      score: 0.91,
    },
  ],
};

export async function POST(req: Request) {
  try {
    const { imageId } = await req.json();

    // 👉 Simulate model inference delay
    await new Promise((r) => setTimeout(r, 1200));

    // 👉 Respond with mock detections
    return NextResponse.json({
      success: true,
      imageId,
      detections: mockDetections,
    });
  } catch (err) {
    console.error("API ERROR:", err);
    return NextResponse.json(
      { success: false, error: "Something went wrong" },
      { status: 500 }
    );
  }
}
