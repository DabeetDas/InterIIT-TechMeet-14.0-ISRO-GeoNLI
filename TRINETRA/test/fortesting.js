import dotenv from "dotenv";
dotenv.config();

import { supabase } from "../lib/supabaseClient";

async function main() {
  console.log("🔗 Using existing Supabase client...");

  // Insert dummy row
  const fake = {
    image1_url: "https://example.com/fake1.jpg",
    image2_url: "https://example.com/fake2.jpg",
    image1_gsd: 0.5,
    image2_gsd: 0.7,
  };

  const { data: inserted, error: insertError } = await supabase
    .from("tgComparisonModule")
    .insert(fake)
    .select()
    .single();

  if (insertError) {
    console.error("❌ Insert error:", insertError);
    return;
  }

  console.log("✅ Inserted row:", inserted);

  // Fetch latest row
  const { data: fetched, error: fetchError } = await supabase
    .from("tgComparisonModule")
    .select("*")
    .order("created_at", { ascending: false })
    .limit(1)
    .single();

  if (fetchError) {
    console.error("❌ Fetch error:", fetchError);
    return;
  }

  console.log("✅ Latest row:", fetched);

  console.log("🎉 Test complete.");
}

main();
