import type { ImageAttachment } from "@/types";

const MAX_DIMENSION = 2048;
const MAX_FILE_SIZE = 4 * 1024 * 1024; // 4 MB
const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/gif", "image/webp"];

/**
 * Validates, resizes, and converts a File/Blob to an ImageAttachment.
 * - Enforces max 2048×2048 (GPT-4o recommendation)
 * - Compresses to JPEG if oversized
 * - Returns base64 data ready for the API
 */
export async function processImage(file: File | Blob): Promise<ImageAttachment> {
  const mimeType = file.type || "image/png";

  if (!ACCEPTED_TYPES.includes(mimeType)) {
    throw new Error(`Unsupported image type: ${mimeType}. Use PNG, JPEG, GIF, or WebP.`);
  }

  if (file.size > 20 * 1024 * 1024) {
    throw new Error("Image is too large (max 20 MB before compression).");
  }

  // Load into an Image element to get dimensions
  const bitmap = await createImageBitmap(file);
  const { width, height } = bitmap;

  // Decide if we need to resize
  let targetWidth = width;
  let targetHeight = height;

  if (width > MAX_DIMENSION || height > MAX_DIMENSION) {
    const scale = MAX_DIMENSION / Math.max(width, height);
    targetWidth = Math.round(width * scale);
    targetHeight = Math.round(height * scale);
  }

  // Draw onto a canvas (resizes if needed)
  const canvas = new OffscreenCanvas(targetWidth, targetHeight);
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(bitmap, 0, 0, targetWidth, targetHeight);
  bitmap.close();

  // Convert to blob — use JPEG for large images, keep original for small
  let outputType = mimeType;
  let quality = 0.92;

  // If original is > MAX_FILE_SIZE, switch to JPEG for better compression
  if (file.size > MAX_FILE_SIZE && mimeType !== "image/jpeg") {
    outputType = "image/jpeg";
    quality = 0.85;
  }

  const outputBlob = await canvas.convertToBlob({ type: outputType, quality });

  // If still too large, try harder compression
  let finalBlob = outputBlob;
  if (finalBlob.size > MAX_FILE_SIZE) {
    finalBlob = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.7 });
  }

  // Convert to base64
  const base64 = await blobToBase64(finalBlob);

  // Create a preview URL
  const preview = URL.createObjectURL(finalBlob);

  const name = file instanceof File ? file.name : "pasted-image";

  return {
    data: base64,
    mimeType: finalBlob.type || "image/jpeg",
    preview,
    name,
    size: finalBlob.size,
  };
}

/** Convert a Blob to a pure base64 string (no data: prefix) */
function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      // Strip the data:...;base64, prefix
      const base64 = result.split(",")[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/** Revoke a preview URL to free memory */
export function revokePreview(attachment: ImageAttachment | null) {
  if (attachment?.preview) {
    URL.revokeObjectURL(attachment.preview);
  }
}

