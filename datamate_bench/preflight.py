import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def capture(command):
    try:
        run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
        return {"returncode": run.returncode, "stdout": run.stdout.strip(), "stderr": run.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def inspect_hardware():
    report = {"platform": platform.platform(), "python": sys.version,
              "python_executable": sys.executable, "cpu_count": os.cpu_count(),
              "nvidia_smi": capture(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
              "cuda_available": False,
              "model_server_binary": shutil.which("llama-server")}
    try:
        cuda = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
        status = cuda.cuInit(0)
        report["cuda_driver_status"] = status
        report["cuda_available"] = status == 0
        if status == 0:
            count = ctypes.c_int()
            cuda.cuDeviceGetCount(ctypes.byref(count))
            report["cuda_devices"] = []
            for index in range(count.value):
                device = ctypes.c_int()
                cuda.cuDeviceGet(ctypes.byref(device), index)
                name = ctypes.create_string_buffer(256)
                cuda.cuDeviceGetName(name, 256, device)
                report["cuda_devices"].append(name.value.decode(errors="replace"))
            report["cuda_available"] = count.value > 0
    except (OSError, AttributeError) as error:
        report["cuda_driver_error"] = str(error)
    if os.name == "nt":
        report["windows_video_controllers"] = capture([
            "powershell.exe", "-NoProfile", "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json"])
    report["notes"] = ["Preflight does not load weights or allocate a training job.",
                       "CUDA status 100 means CUDA_ERROR_NO_DEVICE in this process.",
                       "GPU names/visibility can differ across Windows sessions or GPU power modes."]
    return report


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
