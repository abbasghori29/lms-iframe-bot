"use client";

import { useState, useCallback, useEffect } from "react";
import { useChat } from "@/hooks/useChat";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import ChatArea from "@/components/ChatArea";
import ChatInput from "@/components/ChatInput";
import type { ImageAttachment } from "@/types";

const SIDEBAR_COLLAPSED_KEY = "cafs_sidebar_collapsed";

export default function Home() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Restore collapsed state from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem(SIDEBAR_COLLAPSED_KEY);
      if (saved === "true") setSidebarCollapsed(true);
    } catch {
      // ignore
    }
  }, []);

  const toggleCollapse = useCallback(() => {
    setSidebarCollapsed((prev) => {
      const next = !prev;
      try {
        localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(next));
      } catch {
        // ignore
      }
      return next;
    });
  }, []);

  const handleOpenSidebar = useCallback(() => {
    // On mobile: open overlay; On desktop: expand if collapsed
    setSidebarOpen(true);
    if (sidebarCollapsed) {
      setSidebarCollapsed(false);
      try {
        localStorage.setItem(SIDEBAR_COLLAPSED_KEY, "false");
      } catch {
        // ignore
      }
    }
  }, [sidebarCollapsed]);

  const {
    sessions,
    activeSessionId,
    messages,
    isLoading,
    send,
    createNewSession,
    switchSession,
    deleteSession,
    renameSession,
  } = useChat();

  const handleSend = useCallback(
    (text: string, image?: ImageAttachment | null) => {
      send(text, image);
    },
    [send]
  );

  return (
    <div className="flex h-dvh overflow-hidden bg-cream-50">
      {/* Sidebar */}
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNewChat={createNewSession}
        onSelectSession={switchSession}
        onDeleteSession={deleteSession}
        onRenameSession={renameSession}
        collapsed={sidebarCollapsed}
        onToggleCollapse={toggleCollapse}
      />

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col min-w-0 relative">
        <Header
          onMenuClick={handleOpenSidebar}
          sidebarCollapsed={sidebarCollapsed}
        />

        <div className="flex-1 flex flex-col min-h-0 bg-pattern">
          <ChatArea
            messages={messages}
            isLoading={isLoading}
            onSendSuggestion={handleSend}
          />

          <ChatInput onSend={handleSend} isLoading={isLoading} />
        </div>
      </main>
    </div>
  );
}
