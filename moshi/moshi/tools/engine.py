import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from aiohttp import web

from .config import ToolDefinition
from .executor import ShellExecutor
from .protocol import (
    encode_tool_completed,
    encode_tool_error,
    encode_tool_invoked,
    send_tool_event,
)

logger = logging.getLogger(__name__)

_SENTENCE_ENDINGS = frozenset(".!?")


class TextAccumulator:
    """Per-connection token buffer that detects sentence boundaries."""

    def __init__(self, min_tokens: int = 8):
        self._buffer: str = ""
        self._token_count: int = 0
        self._min_tokens = min_tokens

    def accumulate_token(self, text: str) -> None:
        self._buffer += text
        self._token_count += 1

    def should_check(self) -> bool:
        if self._token_count < self._min_tokens:
            return False
        if not self._buffer:
            return False
        return self._buffer.rstrip()[-1:] in _SENTENCE_ENDINGS

    def flush(self) -> str:
        text = self._buffer
        self._buffer = ""
        self._token_count = 0
        return text


class ToolIntentEngine:
    """Shared engine that classifies text intent using Gemma 3 and executes tools."""

    def __init__(
        self,
        tools: list[ToolDefinition],
        model_id: str = "google/gemma-3-1b-it",
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        min_tokens: int = 8,
    ):
        self.tools = {t.name: t for t in tools if t.enabled}
        self.min_tokens = min_tokens
        self._executor = ShellExecutor()
        self._checking = asyncio.Lock()
        self._thread_pool = ThreadPoolExecutor(max_workers=1)
        self._lm_gen = None
        self._text_tokenizer = None
        self._tool_cooldowns: dict[str, float] = {}  # tool_name → monotonic timestamp
        self._cooldown_seconds: float = 15.0

        # Build prompt template
        tool_lines = []
        for t in tools:
            if t.enabled:
                tool_lines.append(f"- {t.name}: {t.description}")
        self._tool_list_str = "\n".join(tool_lines)
        self._tool_names = set(self.tools.keys())

        # Build the system prompt suffix for Moshi
        numbered = [f"{i}. {t.name} - {t.description}" for i, t in enumerate(self.tools.values(), 1)]
        self.system_prompt_suffix = (
            " You can use tools by announcing your intent to use any of the following tools: "
            + ", ".join(numbered)
        )

        # Load model on a separate CUDA stream
        logger.info("Loading intent model: %s", model_id)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(device)
        self._model.eval()
        self._device = torch.device(device) if isinstance(device, str) else device
        self._cuda_stream = torch.cuda.Stream(device=self._device) if self._device.type == "cuda" else None
        logger.info("Intent model loaded on %s", device)

    def set_model_refs(self, lm_gen, text_tokenizer) -> None:
        """Wire in the LMGen and text tokenizer for result injection."""
        self._lm_gen = lm_gen
        self._text_tokenizer = text_tokenizer

    def _build_prompt(self, text: str) -> str:
        return (
            "You are a tool-calling intent classifier. The following is text spoken by "
            "an AI voice assistant during a live conversation.\n\n"
            "Determine if the assistant is ANNOUNCING that it will perform a specific "
            "tool action RIGHT NOW.\n\n"
            "Rules:\n"
            "- ONLY return a tool name if the assistant is clearly stating it will "
            "perform the action (e.g. 'Let me check the time', 'I'll look that up').\n"
            "- Return \"none\" if the assistant is merely mentioning a capability, "
            "greeting the user, discussing a topic, reporting a result, or asking "
            "a question.\n"
            "- Return \"none\" if the text contains a time value like '12:30 PM' — "
            "that means the tool already ran.\n\n"
            f"Available tools:\n{self._tool_list_str}\n\n"
            f'Assistant speech: "{text}"\n\n'
            'Respond with ONLY the tool name or "none".'
        )

    def _infer(self, text: str) -> str | None:
        """Run Gemma inference (blocking, meant for thread pool)."""
        prompt = self._build_prompt(text)

        messages = [{"role": "user", "content": prompt}]
        inputs = self._tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
        )
        input_ids = inputs["input_ids"].to(self._device)

        ctx = torch.cuda.stream(self._cuda_stream) if self._cuda_stream else nullcontext()
        with torch.no_grad(), ctx:
            output = self._model.generate(
                input_ids,
                max_new_tokens=16,
                do_sample=False,
            )
        new_tokens = output[0][input_ids.shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()

        if response in self._tool_names:
            return response
        return None

    async def check_and_execute(
        self,
        text: str,
        ws: web.WebSocketResponse,
        clog,
    ) -> None:
        """Classify intent and execute tool if matched. Non-blocking debounce."""
        if self._checking.locked():
            return

        async with self._checking:
            loop = asyncio.get_running_loop()
            try:
                tool_name = await loop.run_in_executor(self._thread_pool, self._infer, text)
            except Exception:
                logger.exception("Intent inference failed")
                return

            if tool_name is None:
                return

            # Per-tool cooldown: skip if this tool fired recently.
            now = time.monotonic()
            last_fired = self._tool_cooldowns.get(tool_name, 0.0)
            if now - last_fired < self._cooldown_seconds:
                clog.log("info", f"Tool {tool_name} on cooldown, skipping")
                return

            tool = self.tools[tool_name]
            clog.log("info", f"Tool intent detected: {tool_name}")
            self._tool_cooldowns[tool_name] = now

            await send_tool_event(ws, encode_tool_invoked(tool_name, text))

            try:
                result = await self._executor.execute(tool)
            except Exception as exc:
                clog.log("error", f"Tool execution failed: {exc}")
                await send_tool_event(ws, encode_tool_error(tool_name, str(exc)))
                return

            if result.timed_out:
                clog.log("warning", f"Tool {tool_name} timed out")
                await send_tool_event(ws, encode_tool_error(tool_name, "Command timed out"))
            else:
                success = result.return_code == 0
                clog.log("info", f"Tool {tool_name} completed: rc={result.return_code}")
                await send_tool_event(
                    ws,
                    encode_tool_completed(
                        tool_name,
                        success,
                        result.return_code,
                        result.stdout,
                        result.duration_ms,
                    ),
                )

                # Inject tool result into Moshi's text token queue so it
                # speaks the result in its own voice.
                if success and tool.inject_result and result.stdout.strip() and self._lm_gen is not None:
                    result_text = tool.result_prefix + result.stdout.strip()
                    tokens = self._text_tokenizer.encode(result_text)
                    self._lm_gen.inject_text_tokens(tokens)
                    clog.log("info", f"Injected {len(tokens)} text tokens for tool result")


# Python 3.10 compat — contextlib.nullcontext works but import it properly
from contextlib import nullcontext
