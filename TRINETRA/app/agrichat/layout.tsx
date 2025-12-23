"use client";

import MenuAppBar from "@/components/layout/topbar";
import HistorySidebar from "@/components/layout/HistorySidebar";
import Box from "@mui/material/Box";

export default function AgriChatLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <Box
      sx={{
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        background: "#000000", // normal dark
        color: "white",
      }}
    >
      {/* Top Navbar */}
      <MenuAppBar />

      {/* Sidebar */}
      <HistorySidebar />

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          overflowY: "auto",
          px: 1.5,
        }}
      >
        {children}
      </Box>
    </Box>
  );
}
