"use client";

import MenuAppBar from "@/components/layout/topbar";
import HistorySidebar from "@/components/layout/HistorySidebar";
import Box from "@mui/material/Box";
import { Toaster } from "sonner";

export default function MainLayout({
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
        background: "#000000",
        color: "white",
      }}
    >
      {/* Top Navbar */}
      <MenuAppBar />

      {/* Sidebar Drawer */}
      <HistorySidebar />

      {/* Sonner Toasts — MUST be inside layout */}
      <Toaster position="top-center" richColors closeButton />

      {/* Main Content Area */}
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
