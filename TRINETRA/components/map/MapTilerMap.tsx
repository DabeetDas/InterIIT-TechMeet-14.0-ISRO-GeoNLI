"use client";

import { useEffect, useRef, forwardRef, useImperativeHandle } from "react";
import { Map as MTMap, config as maptilerConfig } from "@maptiler/sdk";
import { GeocodingControl } from "@maptiler/geocoding-control/maptilersdk";
import "@maptiler/sdk/dist/maptiler-sdk.css";
import "@maptiler/geocoding-control/style.css";

export interface MapRefHandle {
  getState: () => { center: [number, number]; zoom: number };
}

const MapContainer = forwardRef<MapRefHandle>((props, ref) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<any>(null);

  useImperativeHandle(ref, () => ({
    getState() {
      if (!mapRef.current) return { center: [0, 0], zoom: 0 };
      const center = mapRef.current.getCenter();
      return {
        center: [center.lng, center.lat],
        zoom: mapRef.current.getZoom(),
      };
    },
  }));

  useEffect(() => {
    if (!containerRef.current) return;

    maptilerConfig.apiKey = process.env.NEXT_PUBLIC_MAPTILER_KEY!;

    const map = new MTMap({
      container: containerRef.current,
      style: "satellite",
      center: [80.23, 13.72], // [lng, lat] → Sriharikota / SDSC
      zoom: 15,
    });

    mapRef.current = map;

    // ⭐ Correct Search Control (NO apiKey here)
    const control = new GeocodingControl({
      placeholder: "Search Delhi, Mumbai…",
      flyTo: true,
    });

    map.addControl(control, "top-left");

    return () => map.remove();
  }, []);

  return <div ref={containerRef} className="w-full h-full" />;
});

export default MapContainer;
