"use client";

import { useEffect, useRef } from "react";
import { MessageSquare, BookOpen, HelpCircle, GraduationCap } from "lucide-react";
import type { ChatMessage as ChatMessageType } from "@/types";
import ChatMessage, { TypingIndicator } from "./ChatMessage";

interface ChatAreaProps {
  messages: ChatMessageType[];
  isLoading: boolean;
  onSendSuggestion: (text: string) => void;
}

const SUGGESTIONS = [
  { icon: BookOpen, text: "What is the CSI certification?", color: "from-amber-50 to-orange-50 border-amber-200/60 hover:border-amber-300" },
  { icon: HelpCircle, text: "What are the six rules of KYC?", color: "from-sky-50 to-blue-50 border-sky-200/60 hover:border-sky-300" },
  { icon: GraduationCap, text: "Explain mutual fund licensing", color: "from-emerald-50 to-green-50 border-emerald-200/60 hover:border-emerald-300" },
  { icon: MessageSquare, text: "Tell me about ETFs", color: "from-violet-50 to-purple-50 border-violet-200/60 hover:border-violet-300" },
];

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
        <EmptyState onSuggestionClick={onSendSuggestion} />
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

function EmptyState({
  onSuggestionClick,
}: {
  onSuggestionClick: (text: string) => void;
}) {
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
      <h2 className="text-2xl font-bold text-brand-800 mb-2 text-center">
        Welcome to CAFS OnlineCE
      </h2>
      <p className="text-sm text-brand-500 mb-8 text-center max-w-md leading-relaxed">
        Your AI-powered assistant for financial education. Ask about CSI
        certifications, licensing, KYC rules, mutual funds, and more.
      </p>

      {/* Suggestion Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg">
        {SUGGESTIONS.map((item) => (
          <button
            key={item.text}
            onClick={() => onSuggestionClick(item.text)}
            className={`
              flex items-center gap-3 px-4 py-3.5 rounded-xl
              bg-gradient-to-br ${item.color}
              border text-left
              hover:shadow-warm active:scale-[0.98]
              transition-all duration-200 group
            `}
          >
            <item.icon
              size={18}
              className="flex-shrink-0 text-brand-500 group-hover:text-brand-600 transition-colors"
            />
            <span className="text-sm font-medium text-brand-700 leading-snug">
              {item.text}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

