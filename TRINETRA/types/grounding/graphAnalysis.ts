export type GraphSegment = {
  label: string;
  value: number;
  color?: string;
};

export type GraphData = {
  title: string;
  subtitle?: string;
  segments: GraphSegment[];
};
