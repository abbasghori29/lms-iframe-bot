import type { ChatRequest, ChatResponse, StreamEvent, TranscribeResponse } from "@/types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "https://api.onlinece.ca";

export async function sendMessage(
  request: ChatRequest
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/api/v1/chat/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to send message");
  }

  return res.json();
}

/**
 * Stream a chat response token-by-token from the /chat/stream SSE endpoint.
 *
 * Usage:
 *   for await (const event of sendMessageStream(request)) {
 *     if (event.type === "token")  appendToMessage(event.content);
 *     if (event.type === "done")   finalize(event.sources);
 *     if (event.type === "error")  showError(event.content);
 *   }
 */
export async function* sendMessageStream(
  request: ChatRequest
): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${API_BASE}/api/v1/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to start stream");
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE lines are separated by "\n\n"; split and process each complete line
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? ""; // keep any incomplete trailing line in buffer

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6).trim();
      if (payload === "[DONE]") return;
      try {
        yield JSON.parse(payload) as StreamEvent;
      } catch {
        // ignore malformed lines
      }
    }
  }
}

export async function transcribeAudio(
  audioBlob: Blob,
  mimeType = "audio/webm"
): Promise<TranscribeResponse> {
  let extension = "webm";
  if (mimeType.includes("mp4")) extension = "mp4";
  else if (mimeType.includes("ogg")) extension = "ogg";
  else if (mimeType.includes("wav")) extension = "wav";

  const formData = new FormData();
  formData.append("audio", audioBlob, `recording.${extension}`);

  const res = await fetch(`${API_BASE}/api/v1/speech/transcribe`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Transcription failed");
  }

  return res.json();
}

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  }
}

