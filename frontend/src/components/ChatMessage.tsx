"use client";

import { useState, useMemo } from "react";
import { Bot, User, ZoomIn, X } from "lucide-react";
import type { ChatMessage as ChatMessageType } from "@/types";
import { sanitizeHtml } from "@/lib/utils";

/**
 * Inject the blinking caret INSIDE the last HTML element so it always
 * appears inline with the final text — never on its own line after a
 * block-level closing tag like </h3> or </p>.
 *
 * Examples:
 *   "<h3>Title</h3>"         → "<h3>Title{caret}</h3>"
 *   "<p>Some text</p>"       → "<p>Some text{caret}</p>"
 *   "plain text"             → "plain text{caret}"
 *   "<li>x</li></ul>"       → "<li>x{caret}</li></ul>"
 */
function injectStreamingCaret(html: string): string {
  const caret = '<span class="streaming-caret" aria-hidden="true"></span>';

  // Match one or more trailing closing tags (e.g. "</p>", "</li></ul></div>")
  const trailingClose = html.match(/(<\/[a-zA-Z][a-zA-Z0-9]*>\s*)+$/);
  if (trailingClose) {
    const pos = html.lastIndexOf(trailingClose[0]);
    // Insert caret just BEFORE the first trailing close tag
    return html.slice(0, pos) + caret + html.slice(pos);
  }

  // No trailing close tags — just append after the last character
  return html + caret;
}

interface ChatMessageProps {
  message: ChatMessageType;
  isLatest?: boolean;
}

export default function ChatMessage({ message, isLatest }: ChatMessageProps) {
  const isHuman = message.role === "human";
  const hasImage = isHuman && message.imageData;

  return (
    <div
      className={`
        flex gap-3 max-w-3xl mx-auto w-full
        ${isLatest ? "animate-fade-in" : ""}
        ${isHuman ? "flex-row-reverse" : "flex-row"}
      `}
    >
      {/* Avatar */}
      <div
        className={`
          flex-shrink-0 w-9 h-9 rounded-xl flex items-center justify-center shadow-warm
          ${
            isHuman
              ? "bg-gradient-to-br from-brand-400 to-brand-600"
              : "bg-white border border-cream-300/80"
          }
        `}
      >
        {isHuman ? (
          <User size={16} className="text-white" />
        ) : (
          <Bot size={16} className="text-brand-500" />
        )}
      </div>

      {/* Message Bubble */}
      <div
        className={`
          relative max-w-[85%] rounded-2xl
          ${
            isHuman
              ? "bg-gradient-to-br from-brand-500 to-brand-600 text-white rounded-tr-md shadow-warm"
              : "bg-white border border-cream-200 text-brand-800 rounded-tl-md shadow-warm"
          }
        `}
      >
        {/* Image attachment */}
        {hasImage && (
          <ImagePreview
            base64={message.imageData!}
            mimeType={message.imageMimeType || "image/png"}
          />
        )}

        {/* Text content */}
        <div className={`px-4 ${hasImage ? "pt-1.5 pb-3" : "py-3"}`}>
          {isHuman ? (
            message.content ? (
              <p className="text-sm leading-relaxed whitespace-pre-wrap">
                {message.content}
              </p>
            ) : hasImage ? (
              <p className="text-sm leading-relaxed text-white/70 italic">
                Sent an image
              </p>
            ) : null
          ) : (
            <div
              className="
                text-sm leading-relaxed
                chat-html
                [&_h3]:text-base [&_h3]:font-semibold [&_h3]:text-brand-800 [&_h3]:mt-3 [&_h3]:mb-1.5
                [&_p]:mb-2 [&_p]:last:mb-0
                [&_ul]:pl-5 [&_ul]:mb-2 [&_ul]:list-disc [&_ul]:space-y-1
                [&_ol]:pl-5 [&_ol]:mb-2 [&_ol]:list-decimal [&_ol]:space-y-1
                [&_li]:text-sm
                [&_strong]:font-semibold [&_strong]:text-brand-700
                [&_a]:text-brand-500 [&_a]:underline [&_a]:underline-offset-2 hover:[&_a]:text-brand-600
                [&_code]:bg-cream-100 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-xs [&_code]:font-mono
                [&_pre]:bg-cream-100 [&_pre]:p-3 [&_pre]:rounded-lg [&_pre]:overflow-x-auto [&_pre]:text-xs [&_pre]:my-2
                [&_blockquote]:border-l-3 [&_blockquote]:border-brand-300 [&_blockquote]:pl-3 [&_blockquote]:italic [&_blockquote]:text-brand-600
                [&_table]:w-full [&_table]:border-collapse [&_table]:my-2
                [&_th]:bg-cream-100 [&_th]:px-3 [&_th]:py-1.5 [&_th]:text-left [&_th]:text-xs [&_th]:font-semibold [&_th]:border [&_th]:border-cream-300
                [&_td]:px-3 [&_td]:py-1.5 [&_td]:text-sm [&_td]:border [&_td]:border-cream-200
              "
              dangerouslySetInnerHTML={{
                __html: message.isStreaming
                  ? injectStreamingCaret(sanitizeHtml(message.content))
                  : sanitizeHtml(message.content),
              }}
            />
          )}

          {/* Timestamp */}
          {message.timestamp && (
            <p
              className={`
                text-[10px] mt-1.5
                ${isHuman ? "text-white/60" : "text-brand-400"}
              `}
            >
              {new Date(message.timestamp).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

/** Inline image preview with lightbox on click */
function ImagePreview({
  base64,
  mimeType,
}: {
  base64: string;
  mimeType: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const src = `data:${mimeType};base64,${base64}`;

  return (
    <>
      <div
        className="relative cursor-pointer group"
        onClick={() => setExpanded(true)}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt="User image"
          className="rounded-t-2xl rounded-br-none max-h-64 w-auto max-w-full object-contain"
          loading="lazy"
        />
        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors rounded-t-2xl flex items-center justify-center">
          <ZoomIn
            size={24}
            className="text-white opacity-0 group-hover:opacity-80 transition-opacity drop-shadow-lg"
          />
        </div>
      </div>

      {/* Lightbox */}
      {expanded && (
        <div
          className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
          onClick={() => setExpanded(false)}
        >
          <button
            className="absolute top-4 right-4 p-2 rounded-full bg-white/20 hover:bg-white/30 text-white transition-colors"
            onClick={() => setExpanded(false)}
          >
            <X size={20} />
          </button>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={src}
            alt="Expanded view"
            className="max-w-full max-h-full rounded-xl shadow-2xl object-contain"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </>
  );
}

export function TypingIndicator() {
  return (
    <div className="flex gap-3 max-w-3xl mx-auto w-full animate-fade-in">
      <div className="flex-shrink-0 w-9 h-9 rounded-xl flex items-center justify-center bg-white border border-cream-300/80 shadow-warm">
        <Bot size={16} className="text-brand-500" />
      </div>
      <div className="bg-white border border-cream-200 rounded-2xl rounded-tl-md px-5 py-4 shadow-warm">
        <div className="flex items-center gap-1.5">
          <span
            className="w-2 h-2 rounded-full bg-brand-400 animate-bounce-dot"
            style={{ animationDelay: "0s" }}
          />
          <span
            className="w-2 h-2 rounded-full bg-brand-400 animate-bounce-dot"
            style={{ animationDelay: "0.16s" }}
          />
          <span
            className="w-2 h-2 rounded-full bg-brand-400 animate-bounce-dot"
            style={{ animationDelay: "0.32s" }}
          />
        </div>
      </div>
    </div>
  );
}
