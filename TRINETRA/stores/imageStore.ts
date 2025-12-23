import { create } from "zustand";

interface ImageState {
  imageUrl: string | null;
  widthKm: number | null;
  heightKm: number | null;
  filename: string | null;
  parentId: string | null;

  setImage: (data: Partial<ImageState>) => void;
  clearImage: () => void;
}

export const useImageStore = create<ImageState>((set) => ({
  imageUrl: null,
  widthKm: null,
  heightKm: null,
  filename: null,
  parentId: null,

  setImage: (data) => set((state) => ({ ...state, ...data })),

  clearImage: () =>
    set({
      imageUrl: null,
      widthKm: null,
      heightKm: null,
      filename: null,
      parentId: null,
    }),
}));
