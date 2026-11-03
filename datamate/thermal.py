"""GPU temperature guard for local runs (development only; not used in the competition image).

Reads AMD sensors through the driver's ADL library (atiadlxx.dll, Overdrive8 PMLog), no admin
needed. A background thread polls every second and drives a gate that LLM calls wait on:

  temp >= slow  -> each new model request first sleeps a few seconds (more the hotter it is)
  temp >= soft  -> pause new model requests until temp <= resume
  temp >= hard  -> trip: unload the model (ollama stop) and stop the run

The temperature used is the hottest of hotspot (junction) and edge.
python -m datamate.thermal          # print sensors once a second
"""
from __future__ import annotations

import ctypes
import logging
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger("datamate.thermal")

# ADL_PMLOG_SENSORS indices (adl_defines.h)
SENSORS = {"gfx_clock_mhz": 1, "mem_clock_mhz": 2, "edge_c": 8, "mem_c": 9, "fan_rpm": 14, "fan_pct": 15,
           "gfx_activity_pct": 19, "asic_power_w": 23, "hotspot_c": 27}


class _Sensor(ctypes.Structure):
    _fields_ = [("supported", ctypes.c_int), ("value", ctypes.c_int)]


class _PMLog(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int), ("sensors", _Sensor * 256)]


_MALLOC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int)(lambda n: ctypes.cast(ctypes.create_string_buffer(n), ctypes.c_void_p).value)


class AmdGpu:
    def __init__(self):
        self.adl = ctypes.WinDLL("atiadlxx.dll")
        self.ctx = ctypes.c_void_p()
        if self.adl.ADL2_Main_Control_Create(_MALLOC, 1, ctypes.byref(self.ctx)) != 0:
            raise RuntimeError("ADL init failed")
        n = ctypes.c_int()
        self.adl.ADL2_Adapter_NumberOfAdapters_Get(self.ctx, ctypes.byref(n))
        self.index = next((i for i in range(n.value) if self._query(i) is not None), None)
        if self.index is None:
            raise RuntimeError("no AMD adapter reports PMLog sensors")

    def _query(self, i: int) -> dict | None:
        out = _PMLog()
        if self.adl.ADL2_New_QueryPMLogData_Get(self.ctx, i, ctypes.byref(out)) != 0:
            return None
        vals = {k: out.sensors[v].value for k, v in SENSORS.items() if out.sensors[v].supported}
        return vals if vals.get("edge_c") or vals.get("hotspot_c") else None

    def read(self) -> dict:
        return self._query(self.index) or {}


class ThermalGuard:
    """Polls the GPU; `wait()` blocks callers while hot; `tripped` is set past the hard limit."""

    def __init__(self, soft: float = 90, resume: float = 80, hard: float = 100, slow: float = 85,
                 slow_delay: float = 3.0, csv: Path | None = None, on_trip=None, interval: float = 1.0):
        self.gpu = AmdGpu()
        self.soft, self.resume, self.hard = soft, resume, hard
        self.slow, self.slow_delay = slow, slow_delay
        self.cool = threading.Event()
        self.cool.set()
        self.tripped = threading.Event()
        self.on_trip = on_trip
        self.interval = interval
        self.csv = csv
        self.last: dict = {}
        self.max_seen = 0.0
        self.paused_s = 0.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "ThermalGuard":
        if self.csv:
            self.csv.parent.mkdir(parents=True, exist_ok=True)
            if not self.csv.exists():
                self.csv.write_text("time," + ",".join(SENSORS) + ",state\n", encoding="utf-8")
        self._t.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def temp(self, s: dict) -> float:
        return float(max(s.get("hotspot_c", 0), s.get("edge_c", 0)))

    def _loop(self) -> None:
        pause_start = None
        while not self._stop.is_set():
            try:
                s = self.gpu.read()
            except Exception as e:  # noqa: BLE001 - a sensor glitch must not kill the run; fail safe = pause
                log.warning("sensor read failed: %s", e)
                s = {}
            t = self.temp(s) if s else self.soft  # unknown temperature -> treat as hot
            self.last, self.max_seen = s, max(self.max_seen, t)
            if t >= self.hard and not self.tripped.is_set():
                self.tripped.set()
                self.cool.clear()
                log.error("GPU %.0f°C >= %.0f°C: hard stop", t, self.hard)
                if self.on_trip:
                    self.on_trip()
            elif t >= self.soft and self.cool.is_set():
                self.cool.clear()
                pause_start = time.time()
                log.warning("GPU %.0f°C >= %.0f°C: pausing model requests", t, self.soft)
            elif t <= self.resume and not self.cool.is_set() and not self.tripped.is_set():
                self.cool.set()
                self.paused_s += time.time() - (pause_start or time.time())
                log.info("GPU %.0f°C: resuming", t)
            if self.csv:
                state = "tripped" if self.tripped.is_set() else "paused" if not self.cool.is_set() else "ok"
                with self.csv.open("a", encoding="utf-8") as f:
                    f.write(time.strftime("%H:%M:%S") + "," + ",".join(str(s.get(k, "")) for k in SENSORS)
                            + f",{state}\n")
            self._stop.wait(self.interval)

    def wait(self) -> None:
        """Block until the GPU is cool; raise once tripped. Between `slow` and `soft` each request
        first sleeps, so the GPU gets idle time (duty cycling) before it ever has to pause."""
        while not self.cool.wait(1.0):
            if self.tripped.is_set():
                break
        if self.tripped.is_set():
            raise RuntimeError("thermal guard tripped: GPU reached the hard temperature limit")
        t = self.temp(self.last) if self.last else 0
        if t >= self.slow:
            time.sleep(self.slow_delay * (1 + (t - self.slow) / max(1, self.soft - self.slow)))


def unload_ollama(model: str) -> None:
    """Force-kill Ollama: frees the GPU at once (a polite `ollama stop` took >3 s, too slow when the
    card heats 40 C in 2 s). The tray app restarts the server; the run stops on the trip anyway."""
    subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True, timeout=30)


GUARD: ThermalGuard | None = None  # set by run.py when --thermal-guard is on; LLM calls wait on it


def _unload_all() -> None:
    unload_ollama("")


if __name__ == "__main__":
    import sys

    if "--watchdog" in sys.argv:  # independent safety net: unload every Ollama model at the hard limit
        hard = float(sys.argv[sys.argv.index("--watchdog") + 1]) if len(sys.argv) > sys.argv.index("--watchdog") + 1 else 100
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        g = ThermalGuard(soft=hard, resume=hard - 15, hard=hard, slow=hard,
                         csv=Path("out/gpu_watchdog.csv"), on_trip=_unload_all, interval=0.2).start()
        log.info("watchdog on: unload all Ollama models at %.0fC; now %s", hard, g.gpu.read())
        while True:
            time.sleep(30)
            if g.tripped.is_set():  # re-arm once cooled, so a later run is protected again
                if g.temp(g.last) <= hard - 20:
                    g.tripped.clear()
                    g.cool.set()
                    log.info("re-armed at %.0fC", g.temp(g.last))
            log.info("now %.0fC (max %.0fC)", g.temp(g.last), g.max_seen)
    g = AmdGpu()
    while True:
        print(time.strftime("%H:%M:%S"), g.read(), flush=True)
        time.sleep(1)
