"""Chat backends with a common tool-calling interface.

  anthropic : Claude API (proof of concept, e.g. claude-haiku-5-5)
  openai    : any OpenAI-compatible endpoint. This is also the path for the
              final offline submission: serve an open-weight model with vLLM
              inside the container and point OPENAI_BASE_URL at it.
  scripted  : no LLM; fixed tool calls, for testing the plumbing offline.
"""
from __future__ import annotations

import json
import re
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0          # actual USD charged, when the API reports it (OpenRouter)
    cached_tokens: int = 0


class AnthropicChat:
    def __init__(self, model: str, system: str, tools: list[dict]):
        import anthropic

        self.client = anthropic.Anthropic(max_retries=5)
        self.model = model
        self.system = system
        self.tools = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                      for t in tools]
        self.effort = os.environ.get("DATAMATE_EFFORT", "medium")
        self.messages: list[dict] = []

    def send_user(self, text: str) -> Reply:
        self.messages.append({"role": "user", "content": text})
        return self._call()

    def send_tool_results(self, results: list[tuple[str, str]], note: str | None = None) -> Reply:
        content: list[dict] = [{"type": "tool_result", "tool_use_id": i, "content": c} for i, c in results]
        if note:
            content.append({"type": "text", "text": note})
        self.messages.append({"role": "user", "content": content})
        return self._call()

    def _call(self) -> Reply:
        r = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=self.system,
            tools=self.tools,
            messages=self.messages,
            output_config={"effort": self.effort},
        )
        # Append the full content (incl. thinking blocks) so the next turn is valid.
        self.messages.append({"role": "assistant", "content": r.content})
        reply = Reply(input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens)
        for b in r.content:
            if b.type == "text":
                reply.text += b.text
            elif b.type == "tool_use":
                reply.tool_calls.append(ToolCall(b.id, b.name, dict(b.input)))
        if r.stop_reason == "refusal":
            reply.text += "\n[refusal]"
        return reply


class OpenAIChat:
    client_kwargs: dict = {}  # honours OPENAI_API_KEY / OPENAI_BASE_URL unless a subclass sets them

    def __init__(self, model: str, system: str, tools: list[dict]):
        from openai import OpenAI

        self.client = OpenAI(max_retries=5, timeout=300, **self.client_kwargs)
        self.model = model
        self.tools = [{"type": "function", "function": t} for t in tools]
        self.messages: list[dict] = [{"role": "system", "content": system}]
        temp = os.environ.get("DATAMATE_TEMPERATURE", "0")
        self.extra: dict = {}
        if temp != "":
            self.extra["temperature"] = float(temp)
        if os.environ.get("DATAMATE_SEED"):
            self.extra["seed"] = int(os.environ["DATAMATE_SEED"])
        # DATAMATE_REASONING: off | on | low | medium | high (OpenRouter-style `reasoning` field)
        reasoning = os.environ.get("DATAMATE_REASONING", "").lower()
        if reasoning:
            self.extra["extra_body"] = {"reasoning": {"enabled": False} if reasoning == "off" else
                                        {"enabled": True} if reasoning == "on" else {"effort": reasoning}}

    def send_user(self, text: str) -> Reply:
        self.messages.append({"role": "user", "content": text})
        return self._call()

    def send_tool_results(self, results: list[tuple[str, str]], note: str | None = None) -> Reply:
        for i, c in results:
            self.messages.append({"role": "tool", "tool_call_id": i, "content": c})
        if note:
            self.messages.append({"role": "user", "content": note})
        return self._call()

    def _call(self) -> Reply:
        from . import thermal
        if thermal.GUARD:  # local GPU runs: wait while the GPU is hot
            thermal.GUARD.wait()
        r = self.client.chat.completions.create(
            model=self.model, messages=self.messages, tools=self.tools,
            max_completion_tokens=8000, **self.extra,
        )
        msg = r.choices[0].message
        entry: dict = {"role": "assistant", "content": msg.content or ""}
        reply = Reply(text=msg.content or "",
                      input_tokens=getattr(r.usage, "prompt_tokens", 0) or 0,
                      output_tokens=getattr(r.usage, "completion_tokens", 0) or 0,
                      cost=float(getattr(r.usage, "cost", 0) or 0),
                      cached_tokens=getattr(getattr(r.usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0)
        if msg.tool_calls:
            entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                reply.tool_calls.append(ToolCall(tc.id, tc.function.name, args))
        self.messages.append(entry)
        return reply


CODEX_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["run_sql", "run_python", "final_answer"]},
        "query": {"type": ["string", "null"], "description": "run_sql: DuckDB SQL"},
        "code": {"type": ["string", "null"], "description": "run_python: Python code"},
        "type": {"type": ["string", "null"], "enum": ["number", "percent", "date", "boolean", "string", "list",
                                                 "table", "unanswerable", "abstain", None]},
        "answer_json": {"type": ["string", "null"],
                        "description": "final_answer: the answer encoded as JSON, e.g. 123 or \"Tên\" or [\"a\",\"b\"] "
                                       "or {\"columns\":[...],\"rows\":[[...]]}"},
        "confidence": {"type": ["number", "null"]},
        "reason": {"type": ["string", "null"]},
    },
    "required": ["action", "query", "code", "type", "answer_json", "confidence", "reason"],
    "additionalProperties": False,
}

CODEX_PROTOCOL = """
# How you act
You are called once per step. Do NOT run shell commands or read files yourself: the host
executes your chosen action and shows you the result in the next step. Reply with exactly
one JSON action matching the output schema (unused fields = null):
- {"action":"run_sql","query":...}
- {"action":"run_python","code":...}
- {"action":"final_answer","type":...,"answer_json":...,"confidence":...,"reason":...}
"""


class CodexChat:
    """Uses the local Codex CLI (`codex exec`, ChatGPT login) as the LLM. Stateless: every
    step sends the whole transcript and asks for one structured JSON action."""

    _counter = 0
    _lock = __import__("threading").Lock()

    def __init__(self, model: str | None, system: str, tools: list[dict]):
        import shutil
        import tempfile

        self.exe = shutil.which("codex") or "codex"
        self.model = model
        self.system = system + CODEX_PROTOCOL
        self.transcript: list[str] = []
        self.dir = Path(tempfile.mkdtemp(prefix="datamate_codex_"))
        (self.dir / "schema.json").write_text(json.dumps(CODEX_ACTION_SCHEMA), encoding="utf-8")
        self.effort = os.environ.get("DATAMATE_EFFORT", "medium")
        self.timeout = int(os.environ.get("DATAMATE_CODEX_TIMEOUT", "600"))

    def send_user(self, text: str) -> Reply:
        self.transcript.append(f"## USER\n{text}")
        return self._call()

    def send_tool_results(self, results: list[tuple[str, str]], note: str | None = None) -> Reply:
        for _, c in results:
            self.transcript.append(f"## RESULT\n{c}")
        if note:
            self.transcript.append(f"## USER\n{note}")
        return self._call()

    def _call(self) -> Reply:
        import subprocess

        with CodexChat._lock:
            CodexChat._counter += 1
            n = CodexChat._counter
        out_file = self.dir / f"out_{n}.json"
        prompt = self.system + "\n\n# Conversation so far\n\n" + "\n\n".join(self.transcript) + \
            "\n\n# Your next action (JSON only)"
        cmd = [self.exe, "exec", "--ephemeral", "--skip-git-repo-check", "-s", "read-only",
               "-C", str(self.dir), "--output-schema", str(self.dir / "schema.json"),
               "-o", str(out_file), "-c", f'model_reasoning_effort="{self.effort}"', "--color", "never"]
        if self.model:
            cmd += ["-m", self.model]
        cmd.append("-")
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=self.timeout)
        raw = out_file.read_text(encoding="utf-8") if out_file.exists() else ""
        if not raw.strip():
            errs = [l for l in proc.stderr.splitlines() if l.startswith("ERROR")]
            raise RuntimeError(f"codex exec failed (rc={proc.returncode}): "
                               f"{' | '.join(dict.fromkeys(errs)) or proc.stderr[-1500:]}")
        self.transcript.append(f"## ASSISTANT\n{raw.strip()}")
        return self._parse(raw, n)

    @staticmethod
    def _parse(raw: str, n: int) -> Reply:
        try:
            a = json.loads(raw)
        except json.JSONDecodeError:
            return Reply(text=raw)
        act = a.get("action")
        if act == "run_sql":
            args = {"query": a.get("query") or ""}
        elif act == "run_python":
            args = {"code": a.get("code") or ""}
        elif act == "final_answer":
            ans = a.get("answer_json")
            try:
                ans = json.loads(ans) if isinstance(ans, str) else ans
            except json.JSONDecodeError:
                pass  # keep the raw string
            args = {"type": a.get("type") or "abstain", "answer": ans,
                    "confidence": a.get("confidence") or 0, "reason": a.get("reason") or ""}
        else:
            return Reply(text=raw)
        return Reply(tool_calls=[ToolCall(f"c{n}", act, args)])


# OpenCode free models only answer requests that look like normal OpenCode agent use, so
# instead of disabling every tool we deny everything that can touch the machine or network
# and leave read-only file tools, which only see an empty temp folder.
OPENCODE_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "permission": {k: "deny" for k in (
        "bash", "edit", "write", "patch", "webfetch", "websearch", "task", "external_directory",
        "doom_loop", "execute", "skill", "experimental_image_generate", "todowrite",
        "opencode_worktree_create", "opencode_worktree_list", "opencode_session_move",
        "opencode_session_rename", "opencode_models", "opencode_list_mcp_resources",
        "opencode_read_mcp_resource")},
}

OPENCODE_JSON_HINT = """
Do NOT call any tool (no read, glob, grep, question): just write the JSON object as plain text,
with keys action, query, code, type, answer_json, confidence, reason (unused = null).
"""


class OpenCodeChat(CodexChat):
    """Local OpenCode CLI (`opencode run --standalone`, e.g. opencode/*-free models). Stateless
    like CodexChat: whole transcript per step, one JSON action back (no schema enforcement)."""

    def __init__(self, model: str | None, system: str, tools: list[dict]):
        import shutil
        import tempfile

        self.exe = shutil.which("opencode") or "opencode"
        self.model = model
        self.system = system + CODEX_PROTOCOL + OPENCODE_JSON_HINT
        self.transcript: list[str] = []
        self.dir = Path(tempfile.mkdtemp(prefix="datamate_opencode_"))
        (self.dir / "opencode.json").write_text(json.dumps(OPENCODE_CONFIG), encoding="utf-8")
        self.timeout = int(os.environ.get("DATAMATE_CODEX_TIMEOUT", "600"))

    def _call(self) -> Reply:
        import subprocess

        with CodexChat._lock:
            CodexChat._counter += 1
            n = CodexChat._counter
        prompt = self.system + "\n\n# Conversation so far\n\n" + "\n\n".join(self.transcript) + \
            "\n\n# Your next action (one JSON object only, no tools)"
        cmd = [self.exe, "run", "--standalone", "--format", "json"] + (["-m", self.model] if self.model else [])
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=self.timeout, cwd=self.dir)
        texts, errors, tokens = [], [], [0, 0]
        for line in proc.stdout.splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") == "text":
                texts.append(e.get("part", {}).get("text") or "")
            elif e.get("type") == "error":
                err = e.get("error") or {}
                errors.append(f"{err.get('type')}: {err.get('message')}")
            elif e.get("type") == "step_finish":
                t = (e.get("part") or {}).get("tokens") or {}
                tokens[0] += int(t.get("input") or 0) + int((t.get("cache") or {}).get("read") or 0)
                tokens[1] += int(t.get("output") or 0) + int(t.get("reasoning") or 0)
        raw = texts[-1].strip() if texts else ""
        if not raw:
            raise RuntimeError(f"opencode run failed (rc={proc.returncode}): "
                               f"{' | '.join(errors) or proc.stderr[-800:]}")
        m = re.search(r"\{.*\}", raw, re.S)  # tolerate ```json fences / chatter around the object
        raw = m.group(0) if m else raw
        self.transcript.append(f"## ASSISTANT\n{raw}")
        reply = self._parse(raw, n)
        reply.input_tokens, reply.output_tokens = tokens
        return reply


class ScriptedChat:
    """Offline stand-in: lists tables, then abstains. Exercises every code path."""

    def __init__(self, model: str, system: str, tools: list[dict]):
        self.step = 0

    def send_user(self, text: str) -> Reply:
        return self._next()

    def send_tool_results(self, results, note=None) -> Reply:
        return self._next()

    def _next(self) -> Reply:
        self.step += 1
        if self.step == 1:
            return Reply(tool_calls=[ToolCall("s1", "run_sql", {"query": "SHOW TABLES"})])
        if self.step == 2:
            return Reply(tool_calls=[ToolCall("s2", "run_python", {"code": "print(sorted(tables))"})])
        return Reply(tool_calls=[ToolCall("s3", "final_answer", {
            "type": "abstain", "answer": None, "confidence": 0.0, "reason": "scripted backend"})])


class OpenRouterChat(OpenAIChat):
    """Hosted open-weight models for development (same code path as a local vLLM server)."""

    def __init__(self, model: str, system: str, tools: list[dict]):
        self.client_kwargs = {"base_url": "https://openrouter.ai/api/v1",
                              "api_key": os.environ.get("OPENROUTER_API_KEY")}
        super().__init__(model, system, tools)


class OllamaChat(OpenAIChat):
    """Local Ollama server (free, offline; same code path as vLLM in the final image)."""

    def __init__(self, model: str, system: str, tools: list[dict]):
        self.client_kwargs = {"base_url": os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
                              "api_key": "ollama"}
        super().__init__(model, system, tools)
        if os.environ.get("DATAMATE_REASONING", "").lower() == "off":
            self.extra["extra_body"] = {"reasoning_effort": "none", "think": False}
        # a runaway generation would otherwise run into the request timeout and be retried 5 times
        self.extra["max_tokens"] = int(os.environ.get("DATAMATE_MAX_TOKENS", "4096"))

    def _call(self) -> Reply:
        from openai import InternalServerError

        try:
            return super()._call()
        except InternalServerError as e:
            if "repeat" not in str(e):
                raise
            # greedy decoding sometimes loops until Ollama aborts it; retry once with a little sampling
            saved = dict(self.extra)
            self.extra.update(temperature=0.6, seed=0)
            try:
                return super()._call()
            finally:
                self.extra = saved


BACKENDS = {"anthropic": AnthropicChat, "openai": OpenAIChat, "openrouter": OpenRouterChat, "ollama": OllamaChat,
            "codex": CodexChat, "opencode": OpenCodeChat, "scripted": ScriptedChat}
# codex: None = the model set in ~/.codex/config.toml
DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5", "openai": "gpt-6-luna", "openrouter": "qwen/qwen3.5-9b", "ollama": "qwen3.5-9b-40k", "codex": None, "opencode": "opencode/deepseek-v4.1-flash-free", "scripted": "none"}


def make_chat(backend: str, model: str | None, system: str, tools: list[dict]):
    return BACKENDS[backend](model or DEFAULT_MODELS[backend], system, tools)
