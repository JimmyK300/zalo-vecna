"""Local, single-GPU Transformers inference using native Qwen3.5 tool XML."""
import json
import os
import re
import threading
from functools import lru_cache

MODEL = "VikramPal/Qwen3.5-9B-Agentic-GRPO"
REVISION = "f7644c67ab8b2d4fce4434baf96ab150626a8c3e"
LOCK = threading.Lock()


def parse_calls(text):
    calls = []
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S)
    if text.count("<tool_call>") != len(blocks):
        raise ValueError("Truncated tool call")
    for block in blocks:
        fn = re.fullmatch(r"\s*<function=([a-z_]+)>(.*?)</function>\s*", block, re.S)
        if not fn or fn[1] not in ("run_sql", "run_python", "final_answer"):
            raise ValueError("Unknown/malformed native function")
        args = {}
        for key, value in re.findall(r"<parameter=([a-z_]+)>(.*?)</parameter>", fn[2], re.S):
            if key in args:
                raise ValueError("Duplicate argument")
            value = value.strip()
            if key in ("answer", "confidence"):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    # Native template renders scalar Python booleans/null this way.
                    if value in ("True", "False", "None"):
                        value = {"True": True, "False": False, "None": None}[value]
                    elif key == "confidence":
                        raise ValueError("Invalid confidence")
            args[key] = value
        required = {"run_sql": {"query"}, "run_python": {"code"},
                    "final_answer": {"type", "answer", "reason", "confidence"}}[fn[1]]
        if set(args) != required:
            raise ValueError("Missing/extra tool arguments")
        calls.append((fn[1], args))
    return calls


@lru_cache(maxsize=1)
def load_model(model, revision, quantized):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime first")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    options = dict(revision=revision, dtype=dtype, device_map={"": 0},
                   attn_implementation="sdpa", trust_remote_code=False)
    if quantized:
        options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    return AutoModelForCausalLM.from_pretrained(model, **options).eval(), tokenizer


class TransformersChat:
    def __init__(self, model, system, tools):
        self.model = model
        self.tools = [{"type": "function", "function": t} for t in tools]
        self.messages = [{"role": "system", "content": system}]
        self.extra = {"temperature": 0, "seed": 0}

    def send_user(self, text):
        self.messages.append({"role": "user", "content": text})
        return self._call()

    def send_tool_results(self, results, note=None):
        for call_id, content in results:
            self.messages.append({"role": "tool", "content": content, "tool_call_id": call_id})
        if note:
            self.messages.append({"role": "user", "content": note})
        return self._call()

    def _call(self):
        import torch
        from .llm import Reply, ToolCall
        with LOCK:
            model, tokenizer = load_model(self.model, os.environ.get("DATAMATE_MODEL_REVISION", REVISION),
                                          os.environ.get("DATAMATE_4BIT", "1") == "1")
            prompt = tokenizer.apply_chat_template(self.messages, tools=self.tools, tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            n = inputs["input_ids"].shape[1]
            budget = int(os.environ.get("DATAMATE_MAX_TOKENS", "1536"))
            if n + budget > int(os.environ.get("DATAMATE_CONTEXT", "32768")):
                raise ValueError("Context overflow: refusing to truncate rulebook/schema")
            temp = self.extra.get("temperature", 0)
            sampling = {"temperature": temp, "top_p": 0.9} if temp > 0 else {}
            if temp > 0:
                torch.manual_seed(self.extra.get("seed", 0))
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=budget, do_sample=temp > 0,
                    use_cache=True, logits_to_keep=1, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id, **sampling)
            text = tokenizer.decode(generated[0, n:], skip_special_tokens=False)
            text = text.replace(tokenizer.eos_token or "<|im_end|>", "").strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            calls = parse_calls(text)
            content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
            native = [{"type": "function", "function": {"name": name, "arguments": args}} for name, args in calls]
            self.messages.append({"role": "assistant", "content": content, "reasoning_content": "", "tool_calls": native})
            return Reply(text=content, tool_calls=[ToolCall(f"hf_{len(self.messages)}_{i}", name, args)
                         for i, (name, args) in enumerate(calls)], input_tokens=n,
                         output_tokens=generated.shape[1] - n)
