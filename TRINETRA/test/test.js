// test.js
import { createClient } from "@supabase/supabase-js";

// DIRECTLY PASTE YOUR KEYS HERE
const SUPABASE_URL = "https://zxkdauahrztftfowhkee.supabase.co/";
const SUPABASE_ANON_KEY =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp4a2RhdWFocnp0ZnRmb3doa2VlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzNTUyMzgsImV4cCI6MjA3OTkzMTIzOH0.BJWY6RTVdMQ-7f43IHAQdRFvvlgvLev6gNbSnRi7Sso";

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function main() {
  console.log("🔗 Connected using manual keys");

  const fake = {
    image1_url: "https://example.com/fake1.jpg",
    image2_url: "https://example.com/fake2.jpg",
    image1_gsd: 0.4,
    image2_gsd: 0.7,
  };

  // INSERT
  const { data: inserted, error: insertError } = await supabase
    .from("tgComparisonModule")
    .insert(fake)
    .select()
    .single();

  if (insertError) {
    console.error("❌ INSERT ERROR:", insertError);
    return;
  }

  console.log("✅ Inserted row:", inserted);

  // FETCH
  const { data: fetched, error: fetchError } = await supabase
    .from("tgComparisonModule")
    .select("*")
    .order("created_at", { ascending: false })
    .limit(1)
    .single();

  if (fetchError) {
    console.error("❌ FETCH ERROR:", fetchError);
    return;
  }

  console.log("✅ Latest row:", fetched);

  console.log("🎉 Test complete.");
}

main();
