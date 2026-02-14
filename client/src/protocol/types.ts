export type MessageType =
  | "handshake"
  | "audio"
  | "text"
  | "control"
  | "metadata"
  | "tool_event";

export const VERSIONS_MAP = {
  0: 0b00000000,
} as const;

export const MODELS_MAP = {
  0: 0b00000000,
} as const;

export type VERSION = keyof typeof VERSIONS_MAP;

export type MODEL = keyof typeof MODELS_MAP;

export type WSMessage =
  | {
      type: "handshake";
      version: VERSION;
      model: MODEL;
    }
  | {
      type: "audio";
      data: Uint8Array;
    }
  | {
      type: "text";
      data: string;
    }
  | {
      type: "control";
      action: CONTROL_MESSAGE;
    }
  | {
      type: "metadata";
      data: unknown;
    }
  | {
    type: "error";
    data: string;
  }
  | {
    type:"ping";
  }
  | {
    type: "tool_event";
    data: ToolEvent;
  }

export interface ToolEvent {
  event: "tool_invoked" | "tool_completed" | "tool_error";
  tool: string;
  trigger_text?: string;
  timestamp?: number;
  success?: boolean;
  return_code?: number;
  stdout?: string;
  duration_ms?: number;
  error?: string;
}

export type SocketStatus = "connected" | "disconnected" | "connecting";

export const CONTROL_MESSAGES_MAP = {
  start: 0b00000000,
  endTurn: 0b00000001,
  pause: 0b00000010,
  restart: 0b00000011,
} as const;

export type CONTROL_MESSAGE = keyof typeof CONTROL_MESSAGES_MAP;
