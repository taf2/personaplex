import json
import time

from aiohttp import web


TOOL_EVENT_MSG_TYPE = 0x07


def _encode(payload: dict) -> bytes:
    return bytes([TOOL_EVENT_MSG_TYPE]) + json.dumps(payload).encode("utf-8")


def encode_tool_invoked(tool_name: str, trigger_text: str) -> bytes:
    return _encode({
        "event": "tool_invoked",
        "tool": tool_name,
        "trigger_text": trigger_text,
        "timestamp": time.time(),
    })


def encode_tool_completed(
    tool_name: str,
    success: bool,
    return_code: int,
    stdout: str,
    duration_ms: float,
) -> bytes:
    return _encode({
        "event": "tool_completed",
        "tool": tool_name,
        "success": success,
        "return_code": return_code,
        "stdout": stdout,
        "duration_ms": duration_ms,
    })


def encode_tool_error(tool_name: str, error: str) -> bytes:
    return _encode({
        "event": "tool_error",
        "tool": tool_name,
        "error": error,
    })


async def send_tool_event(ws: web.WebSocketResponse, payload: bytes) -> None:
    """Send a tool event message, silently ignoring closed connections."""
    if not ws.closed:
        try:
            await ws.send_bytes(payload)
        except ConnectionResetError:
            pass
