from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ToolExecutor:
    type: str  # "shell" for now
    command: str
    timeout_seconds: int = 10
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class ToolDefinition:
    name: str
    description: str
    enabled: bool = True
    executor: ToolExecutor | None = None
    examples: list[str] = field(default_factory=list)
    result_prefix: str = ""
    inject_result: bool = True
    execution_mode: str = "async"  # "async" or "blocking"
    blocking_triggers: list[str] = field(default_factory=list)


def load_tools(directory: str) -> list[ToolDefinition]:
    """Scan a directory for *.yaml tool definitions and return validated ToolDefinition list."""
    tools_dir = Path(directory)
    if not tools_dir.is_dir():
        raise FileNotFoundError(f"Tools directory not found: {directory}")

    tools: list[ToolDefinition] = []
    seen_names: set[str] = set()

    for yaml_path in sorted(tools_dir.glob("*.yaml")):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict) or "name" not in data:
            raise ValueError(f"Invalid tool config in {yaml_path}: missing 'name'")

        name = data["name"]
        if name in seen_names:
            raise ValueError(f"Duplicate tool name '{name}' in {yaml_path}")
        seen_names.add(name)

        execution_mode = data.get("execution_mode", "async")
        if execution_mode not in ("async", "blocking"):
            raise ValueError(
                f"Invalid execution_mode '{execution_mode}' in {yaml_path}; expected 'async' or 'blocking'"
            )
        blocking_triggers = data.get("blocking_triggers", [])
        if not isinstance(blocking_triggers, list) or not all(isinstance(x, str) for x in blocking_triggers):
            raise ValueError(
                f"Invalid blocking_triggers in {yaml_path}; expected a list of strings"
            )
        blocking_triggers = [x.strip().lower() for x in blocking_triggers if x.strip()]

        executor = None
        if "executor" in data and data["executor"] is not None:
            ex = data["executor"]
            executor = ToolExecutor(
                type=ex.get("type", "shell"),
                command=ex["command"],
                timeout_seconds=ex.get("timeout_seconds", 10),
                env=ex.get("env", {}),
            )

        tool = ToolDefinition(
            name=name,
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
            executor=executor,
            examples=data.get("examples", []),
            result_prefix=data.get("result_prefix", ""),
            inject_result=data.get("inject_result", True),
            execution_mode=execution_mode,
            blocking_triggers=blocking_triggers,
        )
        tools.append(tool)

    return tools
