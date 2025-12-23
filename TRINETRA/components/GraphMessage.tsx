"use client";

import { PieChart } from "@mui/x-charts/PieChart";
import type { GraphData } from "@/types/grounding/ApiDataanalysis";

export default function GraphMessage({ graph }: { graph: GraphData }) {
  if (!graph) return null;

  const fallbackColors = [
    "#4caf50",
    "#f44336",
    "#2196f3",
    "#ff9800",
    "#9c27b0",
    "#009688",
  ];

  const data = graph.segments.map((s, i) => ({
    id: i,
    label: s.label,
    value: s.value,
    color:
      s.color && s.color.trim() !== ""
        ? s.color
        : fallbackColors[i % fallbackColors.length],
  }));

  return (
    <div className="mt-3 p-5 rounded-2xl bg-[#0e1521] border border-white/10 shadow-[0_4px_20px_rgba(0,0,0,0.25)]">
      {/* Title */}
      <h3 className="text-white text-lg font-semibold mb-1">{graph.title}</h3>
      <p className="text-gray-400 text-xs mb-4">Ratio of generated leads</p>

      <div className="flex justify-center">
        <PieChart
          height={220}
          margin={{ top: 10, bottom: 10, left: 10, right: 10 }}
          series={[
            {
              data,
              innerRadius: 65,
              outerRadius: 100,
              paddingAngle: 2,
              cornerRadius: 4,
              highlightScope: { fade: "global", highlight: "item" },
            },
          ]}
          slotProps={{
            pieArc: {
              style: {
                stroke: "#0e1521",
                strokeWidth: 2,
                filter: "drop-shadow(0px 2px 6px rgba(0,0,0,0.4))",
              },
            },
          }}
          slots={{ legend: () => null }}
        />
      </div>

      {/* Labels */}
      <div className="mt-4 grid grid-cols-2 gap-4">
        {data.map((segment) => (
          <div
            key={segment.id}
            className="flex flex-col items-center bg-[#111b2a] p-3 rounded-xl border border-white/5"
          >
            <span
              className="text-3xl font-bold"
              style={{ color: segment.color }}
            >
              {segment.value}%
            </span>

            <span className="text-gray-300 text-sm mt-1">{segment.label}</span>

            <div className="w-full mt-2 h-1.5 rounded-full bg-white/10">
              <div
                className="h-full rounded-full"
                style={{
                  width: `${segment.value}%`,
                  backgroundColor: segment.color,
                }}
              ></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
