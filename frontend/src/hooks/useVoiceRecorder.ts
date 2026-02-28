"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { transcribeAudio } from "@/lib/api";

interface UseVoiceRecorderOptions {
  /** Called with the transcribed text when recording + transcription finishes. */
  onTranscription: (text: string) => void;
}

/** Number of frequency bars exposed for the waveform visualiser. */
const BAR_COUNT = 24;

export function useVoiceRecorder({ onTranscription }: UseVoiceRecorderOptions) {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);

  /** Normalised frequency values (0-1) for each bar – drives the UI animation. */
  const [audioLevels, setAudioLevels] = useState<number[]>(
    () => new Array(BAR_COUNT).fill(0)
  );

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const mimeTypeRef = useRef("audio/webm");

  // Web Audio API refs (for the live waveform)
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const rafIdRef = useRef<number>(0);

  // ── Waveform animation loop ────────────────────────────────────────
  const startVisualization = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    const tick = () => {
      analyser.getByteFrequencyData(dataArray);

      // Pick evenly-spaced bins and normalise 0-255 → 0-1
      const step = Math.floor(dataArray.length / BAR_COUNT);
      const levels: number[] = [];
      for (let i = 0; i < BAR_COUNT; i++) {
        levels.push(dataArray[i * step] / 255);
      }
      setAudioLevels(levels);
      rafIdRef.current = requestAnimationFrame(tick);
    };

    rafIdRef.current = requestAnimationFrame(tick);
  }, []);

  const stopVisualization = useCallback(() => {
    cancelAnimationFrame(rafIdRef.current);
    setAudioLevels(new Array(BAR_COUNT).fill(0));

    // Tear down Web Audio nodes
    sourceRef.current?.disconnect();
    sourceRef.current = null;
    analyserRef.current?.disconnect();
    analyserRef.current = null;
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current = null;
  }, []);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafIdRef.current);
      sourceRef.current?.disconnect();
      analyserRef.current?.disconnect();
      audioCtxRef.current?.close().catch(() => {});
    };
  }, []);

  // ── Start recording ────────────────────────────────────────────────
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 48000,
          channelCount: 1,
        },
      });

      chunksRef.current = [];

      // ── Web Audio: wire up AnalyserNode for the waveform ─────────
      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.7;
      source.connect(analyser);

      audioCtxRef.current = audioCtx;
      sourceRef.current = source;
      analyserRef.current = analyser;
      startVisualization();

      // ── MediaRecorder ────────────────────────────────────────────
      const preferredTypes = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
        "audio/ogg;codecs=opus",
      ];
      let mimeType = "audio/webm";
      for (const type of preferredTypes) {
        if (MediaRecorder.isTypeSupported(type)) {
          mimeType = type;
          break;
        }
      }
      mimeTypeRef.current = mimeType;

      const recorder = new MediaRecorder(stream, {
        mimeType,
        audioBitsPerSecond: 128000,
      });

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        stopVisualization();

        const blob = new Blob(chunksRef.current, { type: mimeTypeRef.current });

        if (blob.size < 1000) {
          // Too short — discard
          return;
        }

        setIsTranscribing(true);
        try {
          const result = await transcribeAudio(blob, mimeTypeRef.current);
          if (result.text?.trim()) {
            onTranscription(result.text.trim());
          }
        } catch (err) {
          console.error("Transcription failed:", err);
        } finally {
          setIsTranscribing(false);
        }
      };

      mediaRecorderRef.current = recorder;
      recorder.start(100);
      setIsRecording(true);
    } catch (err) {
      console.error("Microphone access denied:", err);
      alert("Could not access microphone. Please check your browser permissions.");
    }
  }, [onTranscription, startVisualization, stopVisualization]);

  // ── Stop recording ─────────────────────────────────────────────────
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  }, []);

  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  return {
    isRecording,
    isTranscribing,
    audioLevels,
    toggleRecording,
    startRecording,
    stopRecording,
  };
}
