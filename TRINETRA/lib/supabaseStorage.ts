import { supabase } from "./supabaseClient";

const DEFAULT_BUCKET =
  process.env.NEXT_PUBLIC_SUPABASE_BUCKET ||
  "chatbot-uploads_tg_comparison_module";

export async function uploadFile(
  file: File,
  bucket: string = DEFAULT_BUCKET,
  destPath?: string
) {
  const path = destPath ?? `chatbotUploads/${Date.now()}-${file.name}`;

  const { data, error } = await supabase.storage
    .from(bucket)
    .upload(path, file, {
      cacheControl: "3600",
      upsert: false,
      contentType: file.type || "application/octet-stream",
    });

  if (error) throw error;

  return {
    path: data?.path ?? path,
  };
}

export function getPublicUrl(path: string, bucket: string = DEFAULT_BUCKET) {
  const res = supabase.storage.from(bucket).getPublicUrl(path);
  return res?.data?.publicUrl ?? null;
}

export async function createSignedUrl(
  path: string,
  expires = 60,
  bucket: string = DEFAULT_BUCKET
) {
  const { data, error } = await supabase.storage
    .from(bucket)
    .createSignedUrl(path, expires);

  if (error) throw error;

  return data?.signedUrl ?? null; // 👉 FIX HERE
}

export async function listFiles(prefix = "", bucket: string = DEFAULT_BUCKET) {
  const { data, error } = await supabase.storage
    .from(bucket)
    .list(prefix, { limit: 100 });
  if (error) throw error;
  return data || [];
}
