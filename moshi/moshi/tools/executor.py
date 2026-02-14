import asyncio
from dataclasses import dataclass
import os
import time

from .config import ToolDefinition


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool
    duration_ms: float


class ShellExecutor:
    async def execute(self, tool: ToolDefinition) -> ExecutionResult:
        if tool.executor is None:
            return ExecutionResult(
                stdout="",
                stderr="No executor configured",
                return_code=-1,
                timed_out=False,
                duration_ms=0.0,
            )

        env = os.environ.copy()
        env.update(tool.executor.env)
        timeout = tool.executor.timeout_seconds

        start = time.monotonic()
        timed_out = False
        try:
            proc = await asyncio.create_subprocess_shell(
                tool.executor.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            return_code = proc.returncode or 0
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            stdout_bytes = b""
            stderr_bytes = b""
            return_code = -1
            timed_out = True

        duration_ms = (time.monotonic() - start) * 1000
        return ExecutionResult(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            return_code=return_code,
            timed_out=timed_out,
            duration_ms=duration_ms,
        )
