// src/components/GroundedImage.tsx
"use client";

import { useEffect, useState } from "react";
import { Stage, Layer, Rect, Image as KonvaImage } from "react-konva";
import useImage from "use-image";
import { useDetectionStore } from "@/stores/useDetectionStore";

type GroundedImageProps = {
  imageUrl: string;
  containerWidth: number;
  containerHeight: number;
};

export default function GroundedImage({
  imageUrl,
  containerWidth,
  containerHeight,
}: GroundedImageProps) {
  const { currentCategory, data } = useDetectionStore();
  const detections = currentCategory ? data[currentCategory] ?? [] : [];

  const [img] = useImage(imageUrl);
  const [scale, setScale] = useState(1);

  useEffect(() => {
    if (!img) return;

    const scaleX = containerWidth / img.width;
    const scaleY = containerHeight / img.height;
    setScale(Math.min(scaleX, scaleY));
  }, [img, containerWidth, containerHeight]);

  if (!img) return <div className="text-gray-400">Loading image…</div>;

  return (
    <Stage
      width={img.width * scale}
      height={img.height * scale}
      scale={{ x: scale, y: scale }}
      className="rounded-2xl"
      style={{
        width: img.width * scale,
        height: img.height * scale,
      }}
    >
      <Layer>
        <KonvaImage image={img} />
        {detections.map((d) => (
          <Rect
            key={d.id}
            x={d.center.x}
            y={d.center.y}
            width={d.size.width}
            height={d.size.height}
            offsetX={d.size.width / 2}
            offsetY={d.size.height / 2}
            rotation={d.angle}
            stroke="lime"
            strokeWidth={3 / scale}
            listening={false}
          />
        ))}
      </Layer>
    </Stage>
  );
}
