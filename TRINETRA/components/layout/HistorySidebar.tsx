"use client";

import { useEffect } from "react";
import Drawer from "@mui/material/Drawer";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import CloseIcon from "@mui/icons-material/Close";
import { useState } from "react";
import { useUIStore } from "@/stores/uiStore";
import { useSessionStore } from "@/stores/forHistory/sessionStore";
import { supabase } from "@/lib/supabaseClient";
import { loadSession } from "@/lib/history/loadSession";
import { useRouter } from "next/navigation";

export default function HistorySidebar() {
  const sidebarOpen = useUIStore((s) => s.sidebarOpen);
  const closeSidebar = useUIStore((s) => s.closeSidebar);

  const sessions = useSessionStore((s) => s.sessions);
  const setSessions = useSessionStore((s) => s.setSessions);

  const router = useRouter();

  // ⭐ STEP 7: load sessions on mount
  useEffect(() => {
    async function fetchSessions() {
      const { data, error } = await supabase
        .from("chat_sessions")
        .select("*")
        .order("created_at", { ascending: false });

      if (!error && data) {
        setSessions(data);
      }
    }

    fetchSessions();
  }, [setSessions]);

  async function handleOpenSession(id: string) {
    closeSidebar();
    await loadSession(id);
    router.push("/chat");
  }

  return (
    <Drawer
      anchor="left"
      open={sidebarOpen}
      onClose={closeSidebar}
      PaperProps={{
        sx: {
          width: 320,
          background: "#0f1624",
          color: "#d1d5db",
          p: 2.5,
          borderRight: "1px solid #1f2937",
        },
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          mb: 2,
        }}
      >
        <Typography sx={{ fontWeight: 600 }}>History</Typography>

        <IconButton
          size="small"
          onClick={closeSidebar}
          sx={{ color: "#9ca3af" }}
        >
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>

      {/* Sessions List */}
      <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
        {sessions.map((s) => (
          <Box
            key={s.id}
            onClick={() => handleOpenSession(s.id)}
            sx={{
              background: "#1a2333",
              p: 1.5,
              borderRadius: "10px",
              cursor: "pointer",
              border: "1px solid #1f2937",
              transition: "0.2s",
              "&:hover": { background: "#1f2b3f", borderColor: "#334155" },
            }}
          >
            <Typography sx={{ fontSize: "0.85rem", color: "#e5e7eb" }}>
              {s.title}
            </Typography>

            <Typography
              sx={{
                fontSize: "0.75rem",
                color: "#64748b",
                textAlign: "right",
                mt: 0.5,
              }}
            >
              {new Date(s.created_at).toLocaleDateString()}
            </Typography>
          </Box>
        ))}
      </Box>
    </Drawer>
  );
}
