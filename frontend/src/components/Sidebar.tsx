"use client";

import { useState, useMemo } from "react";
import {
  Plus,
  Search,
  MessageSquare,
  Trash2,
  Pencil,
  Check,
  X,
  PanelLeftClose,
} from "lucide-react";
import type { ChatSession } from "@/types";
import { truncateText, formatTime } from "@/lib/utils";

interface SidebarProps {
  sessions: ChatSession[];
  activeSessionId: string | null;
  isOpen: boolean;
  onClose: () => void;
  onNewChat: () => void;
  onSelectSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, title: string) => void;
  /** Desktop collapse state */
  collapsed: boolean;
  onToggleCollapse: () => void;
}

export default function Sidebar({
  sessions,
  activeSessionId,
  isOpen,
  onClose,
  onNewChat,
  onSelectSession,
  onDeleteSession,
  onRenameSession,
  collapsed,
  onToggleCollapse,
}: SidebarProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");

  const filteredSessions = useMemo(() => {
    if (!searchQuery.trim()) return sessions;
    const q = searchQuery.toLowerCase();
    return sessions.filter(
      (s) =>
        s.title.toLowerCase().includes(q) ||
        s.messages.some((m) => m.content.toLowerCase().includes(q))
    );
  }, [sessions, searchQuery]);

  const handleRename = (id: string) => {
    if (editTitle.trim()) {
      onRenameSession(id, editTitle.trim());
    }
    setEditingId(null);
  };

  const startEditing = (session: ChatSession) => {
    setEditingId(session.id);
    setEditTitle(session.title);
  };

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-brand-900/30 backdrop-blur-sm z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed lg:relative inset-y-0 left-0 z-50
          w-[300px] flex flex-col
          bg-gradient-to-b from-cream-100 to-cream-200
          border-r border-cream-300/60
          shadow-warm-lg lg:shadow-none
          ${isOpen ? "translate-x-0" : "-translate-x-full"}
          ${collapsed ? "lg:-translate-x-full" : "lg:translate-x-0"}
        `}
        style={{
          transition: "transform 200ms ease-out, margin-right 200ms ease-out",
          marginRight: collapsed ? "-300px" : undefined,
        }}
      >
        {/* Branding + collapse */}
        <div className="flex items-center justify-between px-5 pt-5 pb-3">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-brand-400 to-brand-600 flex items-center justify-center shadow-warm">
              <span className="text-white text-xs font-bold">C</span>
            </div>
            <div>
              <h1 className="text-sm font-bold text-brand-800 tracking-wide uppercase">
                ACSF / CAFS
              </h1>
            </div>
          </div>
          {/* Mobile: close | Desktop: collapse */}
          <button
            onClick={() => {
              onClose();           // closes mobile overlay
              onToggleCollapse();  // collapses on desktop
            }}
            className="p-1.5 rounded-lg text-brand-600 hover:bg-cream-300/60 transition-colors"
            title="Collapse sidebar"
          >
            <PanelLeftClose size={20} />
          </button>
        </div>

        {/* New Chat */}
        <div className="px-4 pb-3">
          <button
            onClick={() => {
              onNewChat();
              onClose();
            }}
            className="
              w-full flex items-center justify-center gap-2 
              py-2.5 px-4 rounded-xl
              bg-gradient-to-r from-brand-500 to-brand-600
              text-white font-semibold text-sm
              shadow-warm hover:shadow-warm-lg
              hover:from-brand-600 hover:to-brand-700
              active:scale-[0.98]
              transition-all duration-200
            "
          >
            <Plus size={18} strokeWidth={2.5} />
            New/nouveau chat
          </button>
        </div>

        {/* Search */}
        <div className="px-4 pb-3">
          <div className="relative">
            <Search
              size={15}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-brand-400"
            />
            <input
              type="text"
              placeholder="Search chats"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="
                w-full pl-9 pr-3 py-2 rounded-xl
                bg-white/70 border border-cream-300/80
                text-sm text-brand-800 placeholder:text-brand-400
                focus:outline-none focus:ring-2 focus:ring-brand-400/30 focus:border-brand-400/50
                transition-all duration-200
              "
            />
          </div>
        </div>

        {/* Session List */}
        <div className="flex-1 overflow-y-auto px-3 pb-4 scrollbar-thin">
          <p className="px-2 pt-1 pb-2 text-[11px] font-semibold uppercase tracking-wider text-brand-500/70">
            Your chats : Historique
          </p>

          {filteredSessions.length === 0 && (
            <p className="text-center text-sm text-brand-400 py-8">
              {searchQuery ? "No matches found" : "No conversations yet"}
            </p>
          )}

          <div className="space-y-1">
            {filteredSessions.map((session) => {
              const isActive = session.id === activeSessionId;
              const isEditing = editingId === session.id;

              return (
                <div
                  key={session.id}
                  className={`
                    group relative flex items-center gap-2 px-3 py-2.5 rounded-xl cursor-pointer
                    transition-all duration-200
                    ${isActive
                      ? "bg-white/80 shadow-warm border border-cream-300/50"
                      : "hover:bg-white/50 border border-transparent"
                    }
                  `}
                  onClick={() => {
                    if (!isEditing) {
                      onSelectSession(session.id);
                      onClose();
                    }
                  }}
                >
                  <MessageSquare
                    size={16}
                    className={`flex-shrink-0 ${isActive ? "text-brand-500" : "text-brand-400"
                      }`}
                  />

                  <div className="flex-1 min-w-0">
                    {isEditing ? (
                      <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                        <input
                          type="text"
                          value={editTitle}
                          onChange={(e) => setEditTitle(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") handleRename(session.id);
                            if (e.key === "Escape") setEditingId(null);
                          }}
                          autoFocus
                          className="
                            flex-1 px-1.5 py-0.5 text-sm rounded-md
                            bg-white border border-brand-300
                            focus:outline-none focus:ring-1 focus:ring-brand-400
                          "
                        />
                        <button
                          onClick={() => handleRename(session.id)}
                          className="p-0.5 text-green-600 hover:text-green-700"
                        >
                          <Check size={14} />
                        </button>
                        <button
                          onClick={() => setEditingId(null)}
                          className="p-0.5 text-red-400 hover:text-red-500"
                        >
                          <X size={14} />
                        </button>
                      </div>
                    ) : (
                      <>
                        <p
                          className={`text-sm truncate ${isActive
                              ? "font-semibold text-brand-800"
                              : "font-medium text-brand-700"
                            }`}
                        >
                          {truncateText(session.title, 30)}
                        </p>
                        <p className="text-[11px] text-brand-400 mt-0.5">
                          {session.messages.length > 0
                            ? formatTime(session.updatedAt)
                            : "No messages"}
                        </p>
                      </>
                    )}
                  </div>

                  {/* Actions */}
                  {!isEditing && (
                    <div
                      className={`
                        flex items-center gap-0.5
                        ${isActive ? "opacity-100" : "opacity-0 group-hover:opacity-100"}
                        transition-opacity duration-200
                      `}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <button
                        onClick={() => startEditing(session)}
                        className="p-1 rounded-md text-brand-400 hover:text-brand-600 hover:bg-cream-200/80 transition-colors"
                        title="Rename"
                      >
                        <Pencil size={13} />
                      </button>
                      <button
                        onClick={() => {
                          if (confirm("Delete this chat?")) {
                            onDeleteSession(session.id);
                          }
                        }}
                        className="p-1 rounded-md text-brand-400 hover:text-red-500 hover:bg-red-50 transition-colors"
                        title="Delete"
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-cream-300/60">
          <p className="text-[10px] text-brand-400 text-center">
            Chatbot ACSF / CAFS v1.0
          </p>
        </div>
      </aside>
    </>
  );
}
