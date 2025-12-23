// src/lib/getImageMetadata.ts
export async function getImageMetadata(imageUrl: string) {
  try {
    const endpoint = process.env.NEXT_PUBLIC_BACKEND_URL + "/imageclass";
    //   "https://isro-backend-api-849482799622.us-central1.run.app/api/metaDataImage";
    console.log(JSON.stringify({ image_url: imageUrl }));

    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_url: imageUrl }),
    });

    if (!res.ok) throw new Error("Failed to fetch image metadata");

    return await res.json(); // { image_class: "Optical" }
  } catch (err) {
    console.error("❌ Metadata error:", err);
    return null;
  }
}
