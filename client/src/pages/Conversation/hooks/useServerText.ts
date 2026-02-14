import { useCallback, useEffect, useState } from "react";
import { useSocketContext } from "../SocketContext";
import { decodeMessage } from "../../../protocol/encoder";

export type ToolTimelineEvent = {
  tool: string;
  phase: "invoked" | "completed" | "error";
  timestamp: number;
  triggerText?: string;
  success?: boolean;
  returnCode?: number;
  durationMs?: number;
  stdout?: string;
  error?: string;
};

export type ToolStatus = {
  tool: string;
  status: "running" | "completed" | "failed" | "error";
  updatedAt: number;
  triggerText?: string;
  returnCode?: number;
  durationMs?: number;
  stdout?: string;
  error?: string;
};

export type UserTranscriptEvent = {
  text: string;
  timestamp: number;
};

export const useServerText = () => {
  const [text, setText] = useState<string[]>([]);
  const [totalTextMessages, setTotalTextMessages] = useState(0);
  const [toolEvents, setToolEvents] = useState<ToolTimelineEvent[]>([]);
  const [toolStatuses, setToolStatuses] = useState<Record<string, ToolStatus>>({});
  const [userTranscript, setUserTranscript] = useState<UserTranscriptEvent[]>([]);
  const { socket } = useSocketContext();

  const onSocketMessage = useCallback((e: MessageEvent) => {
    const dataArray = new Uint8Array(e.data);
    const message = decodeMessage(dataArray);
    if (message.type === "text") {
      setText(text => [...text, message.data]);
      setTotalTextMessages(count => count + 1);
      return;
    }
    if (message.type === "tool_event") {
      if (message.data.event === "tool_invoked") {
        const ts = message.data.timestamp ? message.data.timestamp * 1000 : Date.now();
        setToolEvents(events => [
          ...events,
          {
            tool: message.data.tool,
            phase: "invoked",
            timestamp: ts,
            triggerText: message.data.trigger_text,
          },
        ]);
        setToolStatuses(statuses => ({
          ...statuses,
          [message.data.tool]: {
            tool: message.data.tool,
            status: "running",
            updatedAt: ts,
            triggerText: message.data.trigger_text,
          },
        }));
        return;
      }
      if (message.data.event === "tool_completed") {
        const ts = Date.now();
        setToolEvents(events => [
          ...events,
          {
            tool: message.data.tool,
            phase: "completed",
            timestamp: ts,
            success: message.data.success,
            returnCode: message.data.return_code,
            durationMs: message.data.duration_ms,
            stdout: message.data.stdout,
          },
        ]);
        setToolStatuses(statuses => ({
          ...statuses,
          [message.data.tool]: {
            tool: message.data.tool,
            status: message.data.success ? "completed" : "failed",
            updatedAt: ts,
            returnCode: message.data.return_code,
            durationMs: message.data.duration_ms,
            stdout: message.data.stdout?.trim().slice(0, 180),
          },
        }));
        return;
      }
      if (message.data.event === "tool_error") {
        const ts = Date.now();
        setToolEvents(events => [
          ...events,
          {
            tool: message.data.tool,
            phase: "error",
            timestamp: ts,
            error: message.data.error,
          },
        ]);
        setToolStatuses(statuses => ({
          ...statuses,
          [message.data.tool]: {
            tool: message.data.tool,
            status: "error",
            updatedAt: ts,
            error: message.data.error,
          },
        }));
      }
      return;
    }
    if (message.type === "metadata" && message.data && typeof message.data === "object") {
      const payload = message.data as { kind?: unknown; text?: unknown; timestamp?: unknown };
      if (payload.kind === "user_transcript" && typeof payload.text === "string") {
        const transcriptText = payload.text;
        const ts = typeof payload.timestamp === "number" ? payload.timestamp * 1000 : Date.now();
        setUserTranscript(events => [
          ...events,
          {
            text: transcriptText,
            timestamp: ts,
          },
        ]);
      }
    }
  }, []);

  useEffect(() => {
    const currentSocket = socket;
    if (!currentSocket) {
      return;
    }
    setText([]);
    setToolEvents([]);
    setToolStatuses({});
    setUserTranscript([]);
    currentSocket.addEventListener("message", onSocketMessage);
    return () => {
      currentSocket.removeEventListener("message", onSocketMessage);
    };
  }, [socket]);

  return { text, totalTextMessages, toolEvents, toolStatuses, userTranscript };
};
