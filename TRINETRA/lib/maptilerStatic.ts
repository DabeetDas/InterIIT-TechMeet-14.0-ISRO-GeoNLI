export async function fetchStaticMap(lon: number, lat: number, zoom: number) {
  const key = process.env.NEXT_PUBLIC_MAPTILER_KEY;

  const url = `https://api.maptiler.com/maps/streets-v2/static/${lon},${lat},${zoom}/400x300.png?key=${key}`;

  const res = await fetch(url);

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Static map fetch failed: ${res.status} ${text}`);
  }

  const blob = await res.blob();
  return new File([blob], "map.png", { type: "image/png" });
}
