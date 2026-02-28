export interface ChatMessage {
  role: "human" | "ai";
  content: string;
  timestamp?: string;
  /** Base64 image data (stored for display in chat history) */
  imageData?: string;
  /** MIME type of the image */
  imageMimeType?: string;
  /** True while the AI message is still being streamed token-by-token */
  isStreaming?: boolean;
}

/** One event emitted by the /chat/stream SSE endpoint */
export type StreamEvent =
  | { type: "token"; content: string }
  | { type: "done"; sources: SourceInfo[]; memory_used: boolean; contextualized_query: string }
  | { type: "error"; content: string };

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
}

export interface SourceInfo {
  source: string;
  page: string;
  content_preview: string;
}

export interface ChatRequest {
  question: string;
  user_id?: string;
  session_id?: string;
  chat_history?: { role: string; content: string }[];
  k?: number;
  use_memory?: boolean;
  store_in_memory?: boolean;
  /** Base64-encoded image data */
  image_data?: string;
  /** MIME type of the image (image/png, image/jpeg, etc.) */
  image_mime_type?: string;
}

export interface ChatResponse {
  answer: string;
  sources: SourceInfo[];
  context_used: number;
  quick_suggestions: string[];
  memory_used: boolean;
  user_id?: string;
  session_id?: string;
  contextualized_query?: string;
  error?: string;
}

export interface TranscribeResponse {
  success: boolean;
  text: string;
  language?: string;
  error?: string;
}

/** Pending image attachment before sending */
export interface ImageAttachment {
  data: string; // base64
  mimeType: string;
  preview: string; // object URL for display
  name: string;
  size: number; // bytes
}
