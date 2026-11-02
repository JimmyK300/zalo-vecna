import json
import time
import urllib.error
import urllib.parse
import urllib.request


class LocalModel:
    def __init__(self, endpoint="http://127.0.0.1:8080/v1", model=None, max_tokens=1024, seed=42):
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Only a loopback HTTP model endpoint is allowed; data stays on this computer.")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Use a plain loopback endpoint without credentials/query/fragment")
        self.endpoint, self.model = endpoint.rstrip("/"), model
        self.max_tokens, self.seed = max_tokens, seed
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.records = []

    def request(self, route, body=None, timeout=60):
        data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        request = urllib.request.Request(self.endpoint + route, data=data,
                                         headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=max(.1, timeout)) as response:
                return json.loads(response.read(4_000_000))
        except urllib.error.HTTPError as error:
            detail = error.read(3000).decode("utf-8", errors="replace")
            raise RuntimeError(f"Model HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Cannot reach local model at {self.endpoint}: {error.reason}") from error

    def connect(self):
        response = self.request("/models", timeout=5)
        ids = [v["id"] for v in response.get("data", [])]
        if not ids:
            raise RuntimeError("No models loaded at local endpoint")
        if self.model and self.model not in ids:
            raise RuntimeError(f"Requested model {self.model!r} is not loaded. Available: {ids}")
        self.model = self.model or ids[0]
        return ids

    def chat(self, messages, timeout=120, json_mode=True, require_complete=True):
        body = {"model": self.model, "messages": messages, "temperature": 0,
                "seed": self.seed, "max_tokens": self.max_tokens,
                "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        response = self.request("/chat/completions", body, timeout)
        elapsed = time.perf_counter() - start
        choice = response["choices"][0]
        record = {"request_seconds": elapsed, "usage": response.get("usage", {}),
                  "server_timings": response.get("timings", {}),
                  "finish_reason": choice.get("finish_reason")}
        self.records.append(record)
        content = choice["message"].get("content") or ""
        if not content:
            raise ValueError("Model returned no content; check thinking/token budget and server chat template")
        if require_complete and choice.get("finish_reason") == "length":
            raise ValueError("Model exhausted max_tokens; output may be incomplete")
        return content, record
