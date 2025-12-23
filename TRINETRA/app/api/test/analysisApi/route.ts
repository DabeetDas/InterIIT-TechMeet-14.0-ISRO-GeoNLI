import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const prompt = body.prompt ?? "";

    await new Promise((resolve) => setTimeout(resolve, 1500));

    const mockGroundingData = [
      {
        fdsat1: [
          {
            id: "rock_01",
            category: "rock",
            center: { x: 700, y: 100 },
            size: { width: 60, height: 60 },
            angle: 10,
          },
        ],
      },
    ];
    const graphSample = {
      title: "Land Composition Breakdown",
      segments: [
        { label: "Crops", value: 45 },
        { label: "Grassland", value: 30 },
        { label: "Dry Soil", value: 15 },
        { label: "Eroded", value: 10 },
      ],
    };

    const sampleMockResponse = {
      answer:
        "Vegetation density is high in the north-west region, with moderate barren land patches near the eastern side. Consider soil moisture mapping for better irrigation planning.",

      graph: graphSample,

      groundingData: mockGroundingData,
    };

    const mockResponse = {
      answer:
        "Soil analysis indicates mixed vegetation with moderate erosion risk.",

      graph: {
        title: "Land Composition Breakdown",
        segments: [
          { label: "Crops", value: 45, color: "#6ddf6d" },
          { label: "Grassland", value: 30, color: "#9be87d" },
          { label: "Dry Soil", value: 15, color: "" },
          { label: "Eroded", value: 10, color: "" },
        ],
      },

      groundingData: null, // test null behaviour
    };

    return NextResponse.json(sampleMockResponse);
  } catch (err) {
    console.error(err);
    return NextResponse.json({ error: "Invalid request" }, { status: 500 });
  }
}
