export const sampleApiData = {
  answer: "Improve irrigation strategies.",
  graph: {
    title: "Percentage of vegetated(red)/barren(cyan) land",
    data: [85.23851851851852, 14.761481481481482],
  },
  groundingData: [
    {
      planes: [
        {
          id: "plane_01",
          category: "airplane",
          center: { x: 120, y: 90 },
          size: { width: 80, height: 35 },
          angle: 8,
          score: 0.94,
        },
        {
          id: "plane_02",
          category: "airplane",
          center: { x: 260, y: 110 },
          size: { width: 70, height: 30 },
          angle: -6,
          score: 0.91,
        },
        {
          id: "plane_03",
          category: "airplane",
          center: { x: 380, y: 150 },
          size: { width: 75, height: 32 },
          angle: 14,
          score: 0.89,
        },
      ],
      cars: [
        {
          id: "car_01",
          category: "car",
          center: { x: 110, y: 140 },
          size: { width: 40, height: 20 },
          angle: -5,
          score: 0.92,
        },
        {
          id: "car_02",
          category: "car",
          center: { x: 180, y: 160 },
          size: { width: 42, height: 22 },
          angle: 10,
          score: 0.9,
        },
        {
          id: "car_03",
          category: "car",
          center: { x: 260, y: 180 },
          size: { width: 45, height: 22 },
          angle: -7,
          score: 0.88,
        },
      ],
      buses: [
        {
          id: "bus_01",
          category: "bus",
          center: { x: 140, y: 200 },
          size: { width: 90, height: 30 },
          angle: 5,
          score: 0.93,
        },
        {
          id: "bus_02",
          category: "bus",
          center: { x: 300, y: 230 },
          size: { width: 95, height: 32 },
          angle: -4,
          score: 0.9,
        },
      ],
    },
  ],
};
