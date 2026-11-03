"""Short, guarded GPU load test for the local Ollama model.

Polls the AMD sensors every 0.2 s; if the hotspot or edge reaches LIMIT, force-kills every
ollama.exe process (frees the GPU instantly) and stops. Prints a compact temperature trace.

python scripts/gpu_load_test.py [prompt_chars] [max_tokens] [limit_c]
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time

from openai import OpenAI

sys.path.insert(0, ".")
from datamate.thermal import AmdGpu  # noqa: E402

PROMPT_CHARS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 150
LIMIT = float(sys.argv[3]) if len(sys.argv) > 3 else 95

gpu = AmdGpu()
trace, tripped, done = [], threading.Event(), threading.Event()


def kill_ollama() -> None:
    subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True)


def watch() -> None:
    t0 = time.time()
    while not done.is_set():
        s = gpu.read()
        hot = max(s.get("hotspot_c", 0), s.get("edge_c", 0))
        trace.append((round(time.time() - t0, 1), s.get("edge_c"), s.get("hotspot_c"), s.get("asic_power_w"),
                      s.get("fan_pct"), s.get("gfx_clock_mhz"), s.get("gfx_activity_pct")))
        if hot >= LIMIT and not tripped.is_set():
            tripped.set()
            kill_ollama()
        time.sleep(0.2)


th = threading.Thread(target=watch, daemon=True)
th.start()
time.sleep(1.0)  # baseline
err = None
t = time.time()
try:
    c = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama", timeout=120, max_retries=0)
    r = c.chat.completions.create(
        model="qwen3.5-9b-40k", max_tokens=MAX_TOKENS, temperature=0,
        messages=[{"role": "system", "content": ("Dữ liệu cửa hàng. " * (PROMPT_CHARS // 18))},
                  {"role": "user", "content": "Tóm tắt trong một câu."}],
        extra_body={"reasoning_effort": "none", "think": False})
    res = f"ok in {time.time() - t:.1f}s, in={r.usage.prompt_tokens} out={r.usage.completion_tokens}"
except Exception as e:  # noqa: BLE001
    res = f"request ended: {type(e).__name__}: {str(e)[:120]}"
time.sleep(3.0)  # watch the cool-down
done.set()
th.join()
print(res, "| TRIPPED" if tripped.is_set() else "")
print("t(s) edge hotspot W fan% MHz act%")
for row in trace[::2]:
    print(*row)
print("max edge", max(r[1] or 0 for r in trace), "max hotspot", max(r[2] or 0 for r in trace),
      "max W", max(r[3] or 0 for r in trace), "max MHz", max(r[5] or 0 for r in trace))
