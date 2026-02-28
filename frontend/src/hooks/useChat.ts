"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ChatMessage, ChatSession, ImageAttachment } from "@/types";
import { sendMessageStream } from "@/lib/api";
import {
  getUserId,
  generateSessionId,
  loadSessions,
  saveSessions,
} from "@/lib/utils";

export function useChat() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const userId = useRef("");

  // Load from localStorage on mount
  useEffect(() => {
    userId.current = getUserId();
    const stored = loadSessions();
    if (stored.length > 0) {
      setSessions(stored);
      setActiveSessionId(stored[0].id);
    } else {
      const newSession = createNewSessionObject();
      setSessions([newSession]);
      setActiveSessionId(newSession.id);
    }
  }, []);

  // Persist sessions whenever they change
  useEffect(() => {
    if (sessions.length > 0) {
      saveSessions(sessions);
    }
  }, [sessions]);

  const activeSession = sessions.find((s) => s.id === activeSessionId) ?? null;
  const messages = activeSession?.messages ?? [];

  function createNewSessionObject(): ChatSession {
    return {
      id: generateSessionId(),
      title: "New Chat",
      messages: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
  }

  const createNewSession = useCallback(() => {
    const newSession = createNewSessionObject();
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newSession.id);
    setError(null);
    return newSession.id;
  }, []);

  const switchSession = useCallback((sessionId: string) => {
    setActiveSessionId(sessionId);
    setError(null);
  }, []);

  const deleteSession = useCallback(
    (sessionId: string) => {
      setSessions((prev) => {
        const filtered = prev.filter((s) => s.id !== sessionId);
        if (filtered.length === 0) {
          const newSession = createNewSessionObject();
          setActiveSessionId(newSession.id);
          return [newSession];
        }
        if (activeSessionId === sessionId) {
          setActiveSessionId(filtered[0].id);
        }
        return filtered;
      });
    },
    [activeSessionId]
  );

  const renameSession = useCallback((sessionId: string, newTitle: string) => {
    setSessions((prev) =>
      prev.map((s) => (s.id === sessionId ? { ...s, title: newTitle } : s))
    );
  }, []);

  const send = useCallback(
    async (text: string, image?: ImageAttachment | null) => {
      if ((!text.trim() && !image) || isLoading) return;

      const currentSessionId = activeSessionId;
      if (!currentSessionId) return;

      const userMessage: ChatMessage = {
        role: "human",
        content: text.trim(),
        timestamp: new Date().toISOString(),
        imageData: image?.data,
        imageMimeType: image?.mimeType,
      };

      // Optimistically add user message
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== currentSessionId) return s;
          const updated = {
            ...s,
            messages: [...s.messages, userMessage],
            updatedAt: new Date().toISOString(),
          };
          // Auto-title from first message
          if (s.messages.length === 0) {
            const titleText = text.trim() || "Image";
            updated.title =
              titleText.length > 40
                ? titleText.slice(0, 40) + "…"
                : titleText;
          }
          return updated;
        })
      );

      setIsLoading(true);
      setError(null);

      // Track whether we've added the AI message bubble yet.
      // We deliberately wait until the FIRST token arrives so the
      // 3-dot typing animation keeps showing during the "thinking" phase.
      let streamingMessageAdded = false;

      try {
        // Build chat history from current session (text only)
        const session = sessions.find((s) => s.id === currentSessionId);
        const history = session
          ? session.messages.map((m) => ({ role: m.role, content: m.content }))
          : [];

        // Helper: add the streaming AI message to the session
        const addStreamingMessage = (firstToken: string) => {
          setSessions((prev) =>
            prev.map((s) => {
              if (s.id !== currentSessionId) return s;
              return {
                ...s,
                messages: [
                  ...s.messages,
                  {
                    role: "ai" as const,
                    content: firstToken,
                    timestamp: new Date().toISOString(),
                    isStreaming: true,
                  },
                ],
                updatedAt: new Date().toISOString(),
              };
            })
          );
          streamingMessageAdded = true;
        };

        // Helper: update the last (streaming) AI message
        const updateLastAiMessage = (patch: Partial<ChatMessage>) => {
          setSessions((prev) =>
            prev.map((s) => {
              if (s.id !== currentSessionId) return s;
              const msgs = [...s.messages];
              const last = msgs[msgs.length - 1];
              if (last?.role === "ai" && last?.isStreaming) {
                msgs[msgs.length - 1] = { ...last, ...patch };
              }
              return { ...s, messages: msgs };
            })
          );
        };

        for await (const event of sendMessageStream({
          question: text.trim(),
          user_id: userId.current,
          session_id: currentSessionId,
          chat_history: history,
          k: 5,
          use_memory: true,
          store_in_memory: true,
          image_data: image?.data,
          image_mime_type: image?.mimeType,
        })) {
          if (event.type === "token") {
            if (!streamingMessageAdded) {
              // First token — create the AI bubble (replaces 3-dot animation)
              addStreamingMessage(event.content);
            } else {
              // Subsequent tokens — append to existing bubble
              setSessions((prev) =>
                prev.map((s) => {
                  if (s.id !== currentSessionId) return s;
                  const msgs = [...s.messages];
                  const last = msgs[msgs.length - 1];
                  if (last?.role === "ai" && last?.isStreaming) {
                    msgs[msgs.length - 1] = {
                      ...last,
                      content: last.content + event.content,
                    };
                  }
                  return { ...s, messages: msgs };
                })
              );
            }
          } else if (event.type === "done") {
            if (streamingMessageAdded) {
              updateLastAiMessage({ isStreaming: false });
            }
          } else if (event.type === "error") {
            if (!streamingMessageAdded) {
              addStreamingMessage("");
            }
            updateLastAiMessage({
              content: `<p>Sorry, I encountered an error: ${event.content}. Please try again.</p>`,
              isStreaming: false,
            });
            setError(event.content);
          }
        }
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Something went wrong";
        setError(message);

        if (streamingMessageAdded) {
          // Replace the streaming placeholder with the error
          setSessions((prev) =>
            prev.map((s) => {
              if (s.id !== currentSessionId) return s;
              const msgs = [...s.messages];
              const last = msgs[msgs.length - 1];
              if (last?.role === "ai" && last?.isStreaming) {
                msgs[msgs.length - 1] = {
                  ...last,
                  content: `<p>Sorry, I encountered an error: ${message}. Please try again.</p>`,
                  isStreaming: false,
                };
              }
              return { ...s, messages: msgs };
            })
          );
        } else {
          // No streaming message was added yet — add the error as a new message
          setSessions((prev) =>
            prev.map((s) =>
              s.id !== currentSessionId
                ? s
                : {
                    ...s,
                    messages: [
                      ...s.messages,
                      {
                        role: "ai" as const,
                        content: `<p>Sorry, I encountered an error: ${message}. Please try again.</p>`,
                        timestamp: new Date().toISOString(),
                      },
                    ],
                    updatedAt: new Date().toISOString(),
                  }
            )
          );
        }
      } finally {
        setIsLoading(false);
      }
    },
    [activeSessionId, isLoading, sessions]
  );

  return {
    sessions,
    activeSessionId,
    activeSession,
    messages,
    isLoading,
    error,
    send,
    createNewSession,
    switchSession,
    deleteSession,
    renameSession,
  };
}
