"use client";

import { useEffect, useRef } from "react";
import { MessageSquare, GraduationCap } from "lucide-react";
import type { ChatMessage as ChatMessageType } from "@/types";
import ChatMessage, { TypingIndicator } from "./ChatMessage";

interface ChatAreaProps {
  messages: ChatMessageType[];
  isLoading: boolean;
  onSendSuggestion: (text: string) => void;
}

export default function ChatArea({
  messages,
  isLoading,
  onSendSuggestion,
}: ChatAreaProps) {
  const endRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-y-auto scrollbar-thin"
    >
      {messages.length === 0 ? (
        <EmptyState />
      ) : (
        <div className="py-6 px-4 space-y-5">
          {messages.map((msg, idx) => (
            <ChatMessage
              key={`${msg.role}-${idx}`}
              message={msg}
              isLatest={idx === messages.length - 1}
            />
          ))}
          {/* 3-dot animation shows while loading and disappears once
              the first streamed token creates the AI message bubble */}
          {isLoading && !messages.some((m) => m.isStreaming) && (
            <TypingIndicator />
          )}
          <div ref={endRef} />
        </div>
      )}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full px-6 py-12 animate-fade-in">
      {/* Logo / Icon */}
      <div className="relative mb-8">
        <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-brand-400 to-brand-600 flex items-center justify-center shadow-warm-lg rotate-3">
          <GraduationCap size={36} className="text-white -rotate-3" />
        </div>
        <div className="absolute -bottom-1 -right-1 w-7 h-7 rounded-lg bg-white border border-cream-300 flex items-center justify-center shadow-warm">
          <MessageSquare size={14} className="text-brand-500" />
        </div>
      </div>

      {/* Welcome Text */}
      <h2 className="text-2xl font-bold text-brand-800 mb-4 text-center">
        Bienvenue sur l&apos;assistant virtuel de l&apos;ACSF / Welcome to the CAFS Virtual Assistant
      </h2>
      <p className="text-sm text-brand-500 text-center max-w-md leading-relaxed mb-3">
        Cet assistant virtuel a été conçu pour soutenir les étudiants dans leur étude du cours Fonds d&apos;investissement au Canada. Toutes les réponses sont basées uniquement sur le document PDF officiel du cours. Vous pouvez également déposer des fichiers / capture d&apos;écran.
      </p>
      <p className="text-sm text-brand-500 text-center max-w-md leading-relaxed">
        This virtual assistant was designed to support students in their study of the Investment Funds in Canada course. All responses are based solely on the official course PDF document. You can even drop a screenshot or ask a question.
      </p>
    </div>
  );
}

