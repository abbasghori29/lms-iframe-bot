"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Send, Mic, MicOff, Loader2, Paperclip, X, ImageIcon } from "lucide-react";
import { useVoiceRecorder } from "@/hooks/useVoiceRecorder";
import { processImage, revokePreview } from "@/lib/image";
import type { ImageAttachment } from "@/types";

interface ChatInputProps {
  onSend: (text: string, image?: ImageAttachment | null) => void;
  isLoading: boolean;
}

export default function ChatInput({ onSend, isLoading }: ChatInputProps) {
  const [text, setText] = useState("");
  const [image, setImage] = useState<ImageAttachment | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-send after transcription — no need to press the send button
  const handleTranscription = useCallback(
    (transcribed: string) => {
      const combined = text.trim()
        ? `${text.trim()} ${transcribed}`
        : transcribed;
      setText("");
      if (textareaRef.current) textareaRef.current.style.height = "auto";
      onSend(combined, image ?? undefined);
      setImage(null);
    },
    [text, image, onSend]
  );

  const { isRecording, isTranscribing, audioLevels, toggleRecording } =
    useVoiceRecorder({
      onTranscription: handleTranscription,
    });

  // ─── Image handling ───────────────────────────────
  const attachImage = useCallback(async (file: File | Blob) => {
    setImageError(null);
    try {
      const attachment = await processImage(file);
      // Revoke old preview if replacing
      revokePreview(image);
      setImage(attachment);
    } catch (err) {
      setImageError(err instanceof Error ? err.message : "Failed to process image");
      setTimeout(() => setImageError(null), 4000);
    }
  }, [image]);

  const removeImage = useCallback(() => {
    revokePreview(image);
    setImage(null);
    setImageError(null);
  }, [image]);

  // Clean up preview URL on unmount
  useEffect(() => {
    return () => {
      revokePreview(image);
    };
  }, [image]);

  // ─── File input handler ───────────────────────────
  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) attachImage(file);
      // Reset so the same file can be selected again
      e.target.value = "";
    },
    [attachImage]
  );

  // ─── Paste handler (Ctrl+V image) ────────────────
  const handlePaste = useCallback(
    (e: React.ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          e.preventDefault();
          const file = item.getAsFile();
          if (file) attachImage(file);
          return;
        }
      }
    },
    [attachImage]
  );

  // ─── Drag & Drop ─────────────────────────────────
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file && file.type.startsWith("image/")) {
        attachImage(file);
      }
    },
    [attachImage]
  );

  // ─── Send ─────────────────────────────────────────
  const handleSend = useCallback(() => {
    if ((!text.trim() && !image) || isLoading) return;
    onSend(text.trim(), image);
    setText("");
    setImage(null);
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [text, image, isLoading, onSend]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 150) + "px";
  }, [text]);

  const canSend = (text.trim() || image) && !isLoading;

  return (
    <div
      className="border-t border-cream-200/80 bg-gradient-to-t from-cream-100/90 to-cream-50/60 backdrop-blur-md"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Drag overlay */}
      {isDragging && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-brand-500/10 backdrop-blur-sm border-2 border-dashed border-brand-400 rounded-2xl m-2 pointer-events-none">
          <div className="flex flex-col items-center gap-2 text-brand-600">
            <ImageIcon size={32} />
            <span className="text-sm font-semibold">Drop image here</span>
          </div>
        </div>
      )}

      <div className="max-w-3xl mx-auto px-4 py-3">
        {/* Image preview */}
        {image && (
          <div className="mb-2 flex items-start gap-2 animate-fade-in">
            <div className="relative group">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={image.preview}
                alt="Attached"
                className="h-20 w-auto max-w-[200px] rounded-xl border border-cream-300 object-cover shadow-warm"
              />
              <button
                onClick={removeImage}
                className="
                  absolute -top-2 -right-2
                  w-5 h-5 rounded-full
                  bg-red-500 text-white
                  flex items-center justify-center
                  shadow-md
                  opacity-0 group-hover:opacity-100
                  transition-opacity duration-200
                  hover:bg-red-600
                "
                title="Remove image"
              >
                <X size={12} />
              </button>
            </div>
            <span className="text-xs text-brand-400 mt-1 truncate max-w-[150px]">
              {image.name} ({(image.size / 1024).toFixed(0)} KB)
            </span>
          </div>
        )}

        {/* Image error */}
        {imageError && (
          <p className="text-xs text-red-500 mb-2 animate-fade-in">{imageError}</p>
        )}

        <div
          className={`
            flex items-end gap-2
            bg-white rounded-2xl
            border
            ${isDragging ? "border-brand-400 shadow-warm-lg" : "border-cream-300/70"}
            shadow-warm
            focus-within:shadow-warm-lg focus-within:border-brand-400/40
            transition-all duration-200
            px-3 py-2
          `}
        >
          {/* Image upload button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isLoading}
            className="
              flex-shrink-0 w-9 h-9 rounded-xl
              flex items-center justify-center
              bg-cream-100 text-brand-500
              hover:bg-cream-200 hover:text-brand-600
              active:scale-95
              disabled:opacity-40 disabled:cursor-not-allowed
              transition-all duration-200
            "
            title="Attach image (PNG, JPEG, GIF, WebP)"
          >
            <Paperclip size={16} />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/gif,image/webp"
            onChange={handleFileSelect}
            className="hidden"
          />

          {/* Text Input */}
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            placeholder={image ? "Add a message about this image…" : "Ask a question… or paste/drop an image"}
            rows={1}
            disabled={isLoading}
            className="
              flex-1 resize-none bg-transparent
              text-sm text-brand-800 placeholder:text-brand-400/70
              focus:outline-none
              max-h-[150px] py-1.5
              disabled:opacity-50
            "
          />

          {/* Voice Button + Waveform */}
          {isRecording ? (
            <div className="flex items-center gap-1.5">
              {/* Live waveform bars */}
              <div className="flex items-center gap-[2px] h-9 px-1">
                {audioLevels.map((level, i) => (
                  <div
                    key={i}
                    className="w-[3px] rounded-full bg-red-500"
                    style={{
                      height: `${Math.max(4, level * 28)}px`,
                      transition: "height 80ms ease-out",
                    }}
                  />
                ))}
              </div>
              {/* Stop button */}
              <button
                onClick={toggleRecording}
                className="
                  relative flex-shrink-0 w-9 h-9 rounded-xl
                  flex items-center justify-center
                  bg-red-500 text-white shadow-lg shadow-red-200 scale-110
                  transition-all duration-200
                "
                title="Stop recording"
              >
                <MicOff size={16} />
                <span className="absolute inset-0 rounded-xl border-2 border-red-400 animate-pulse-ring" />
              </button>
            </div>
          ) : (
            <button
              onClick={toggleRecording}
              disabled={isLoading || isTranscribing}
              className={`
                relative flex-shrink-0 w-9 h-9 rounded-xl
                flex items-center justify-center
                transition-all duration-200
                ${
                  isTranscribing
                    ? "bg-brand-100 text-brand-500"
                    : "bg-cream-100 text-brand-500 hover:bg-cream-200 hover:text-brand-600"
                }
                disabled:opacity-40 disabled:cursor-not-allowed
              `}
              title="Voice input"
            >
              {isTranscribing ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <Mic size={16} />
              )}
            </button>
          )}

          {/* Send Button */}
          <button
            onClick={handleSend}
            disabled={!canSend}
            className="
              flex-shrink-0 w-9 h-9 rounded-xl
              flex items-center justify-center
              bg-gradient-to-r from-brand-500 to-brand-600
              text-white shadow-warm
              hover:from-brand-600 hover:to-brand-700 hover:shadow-warm-lg
              active:scale-95
              disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:shadow-warm
              transition-all duration-200
            "
            title="Send message"
          >
            {isLoading ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Send size={15} />
            )}
          </button>
        </div>

        {/* Status Messages */}
        {isRecording && (
          <p className="text-center text-xs text-red-500 mt-2 font-medium flex items-center justify-center gap-1">
            <span className="inline-block w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            Listening… tap stop to send
          </p>
        )}
        {isTranscribing && (
          <p className="text-center text-xs text-brand-500 mt-2 font-medium">
            Transcribing &amp; sending…
          </p>
        )}
      </div>
    </div>
  );
}
