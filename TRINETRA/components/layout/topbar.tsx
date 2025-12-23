"use client";

import * as React from "react";
import Box from "@mui/material/Box";
import IconButton from "@mui/material/IconButton";
import {
  Menu,
  Sprout,
  MessageSquare,
  ImageOff,
  Earth,
  Images,
} from "lucide-react";
import Button from "@mui/material/Button";
import Avatar from "@mui/material/Avatar";
import { useUIStore } from "@/stores/uiStore";
import Link from "next/link";
import { usePathname } from "next/navigation";

export default function MenuAppBar() {
  const toggleSidebar = useUIStore((s) => s.toggleSidebar);
  const pathname = usePathname();

  const isAgriChat = pathname.startsWith("/agrichat");
  const isTgCompare = pathname.startsWith("/tgCompare");

  return (
    <Box
      sx={{
        width: "100%",
        height: 54,
        px: 2,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        background: "transparent",
        position: "relative",
      }}
    >
      {/* LEFT — Hamburger */}
      <IconButton
        onClick={toggleSidebar}
        sx={{
          width: 36,
          height: 36,
          background: "#162133",
          borderRadius: "10px",
          border: "1px solid #1e293b",
          "&:hover": { background: "#1e293b" },
        }}
      >
        <Menu size={18} color="#82aaff" strokeWidth={2.2} />
      </IconButton>

      {/* CENTER TITLE */}
      <Box
        sx={{
          position: "absolute",
          left: "50%",
          transform: "translateX(-50%)",
          top: 16,
          pointerEvents: "none",
        }}
      ></Box>

      {/* RIGHT BUTTONS */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
        {/* Go to Map */}
        <Link href="/map" style={{ textDecoration: "none" }}>
          <Button
            sx={{
              textTransform: "none",
              color: "#82aaff",
              fontSize: "0.8rem",
              borderRadius: "10px",
              px: 1.8,
              py: 0.6,
              border: "1px solid #1e293b",
              height: 32,
              "&:hover": { background: "#1e293b" },
              gap: 0.6, // 👈 adds space between icon and text
            }}
          >
            <Earth size={16} /> Go to Map
          </Button>
        </Link>

        {/* NORMAL CHAT / AGRICHAT Toggle */}
        <Link
          href={isAgriChat ? "/chat" : "/agrichat"}
          style={{ textDecoration: "none" }}
        >
          <Button
            sx={{
              textTransform: "none",
              color: "#4ade80", // green-400
              fontSize: "0.8rem",
              borderRadius: "10px",
              px: 1.8,
              py: 0.6,
              border: "1px solid #1f3d2b", // deep green border
              height: 32,
              display: "flex",
              alignItems: "center",
              gap: 0.6,
              transition: "all 0.2s ease",
              "&:hover": {
                background: "rgba(34,197,94,0.15)", // soft green hover
                borderColor: "#22c55e", // green-500
                color: "#22c55e",
              },
            }}
          >
            {isAgriChat ? (
              <>
                <MessageSquare size={16} /> Normal Chat
              </>
            ) : (
              <>
                <Sprout size={16} /> AgriChat
              </>
            )}
          </Button>
        </Link>

        {/* TGCOMPARE Toggle */}
        <Link
          href={isTgCompare ? "/chat" : "/tgCompare"}
          style={{ textDecoration: "none" }}
        >
          <Button
            sx={{
              textTransform: "none",
              color: "#82aaff",
              fontSize: "0.8rem",
              borderRadius: "10px",
              px: 1.8,
              py: 0.6,
              border: "1px solid #1e293b",
              height: 32,
              display: "flex",
              alignItems: "center",
              gap: 0.6,
              "&:hover": { background: "#1e293b" },
            }}
          >
            <Images size={16} />
            {isTgCompare ? "Normal Chat" : "Geo Compare"}
          </Button>
        </Link>

        {/* Avatar */}
        <Avatar
          src="https://api.dicebear.com/7.x/thumbs/svg?seed=Woolf"
          sx={{
            width: 34,
            height: 34,
            border: "2px solid #1e293b",
          }}
        />
      </Box>
    </Box>
  );
}
