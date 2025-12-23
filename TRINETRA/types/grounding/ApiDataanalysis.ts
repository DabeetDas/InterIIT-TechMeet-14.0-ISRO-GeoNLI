export type GraphSegment = {
  label: string;
  value: number;
  color?: string;
};

export type GraphData = {
  title: string;
  segments: GraphSegment[];
};

export type Detection = {
  id: string;
  category: string;
  center: { x: number; y: number };
  size: { width: number; height: number };
  angle: number;
  score: 0.1;
};

export type AnalysisResult = {
  answer: string | null;
  graph: GraphData | null;
  groundingData: Record<string, Detection[]> | null;
};
