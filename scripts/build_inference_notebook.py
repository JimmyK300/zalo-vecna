"""Build the GitHub + Drive submission notebook. Does not execute notebook cells."""
import ast
import json
import textwrap
from pathlib import Path

cells = []
def md(s):
    cells.append(dict(cell_type="markdown", metadata={}, source=textwrap.dedent(s).strip()+"\n"))
def code(s):
    s = textwrap.dedent(s).strip()+"\n"
    ast.parse(s)
    cells.append(dict(cell_type="code", metadata={}, source=s, execution_count=None, outputs=[]))

md('''
# DataMate — GitHub + Drive → submission.jsonl (không fine-tune)

Chọn **A100 + High RAM**. Chạy lần lượt M0–M5. Code được clone từ nhánh
`feat/datamate-colab-inference` của `JimmyK300/zalo-vecna`.
Model chạy local: `VikramPal/Qwen3.5-9B-Agentic-GRPO`, revision cố định, NF4 4-bit.
Đây là một checkpoint ứng viên; chưa có bằng chứng nó vượt stock Qwen trên Colab.
Giữ recipe của Jimmy; chỉ câu chưa có recipe mới gọi model. Không train và không gọi API LLM.

| Mốc | Output / gate |
|---|---|
| M0 | GitHub checkout + cài thư viện |
| M1 | Tải public test từ Drive, checksum dữ liệu |
| M2 | Khóa config, revision, dataset hash, GPU để resume |
| M3 | Smoke 3 câu buộc qua model; xem log lỗi trước lượt dài |
| M4 | Full pipeline; copy log và dự đoán sang Drive mỗi ~60s |
| M5 | Kiểm tra IDs/format, đếm abstain, lưu và tải submission |

Kết quả là file submission để bạn tự nộp. Notebook không upload lên leaderboard.
Chưa chạy thử GPU; đã kiểm tra CPU và cú pháp. Thời gian phải đo bằng M3.
''')
code('''
import os, sys, json, subprocess, time, shutil, hashlib
from pathlib import Path
REPO = "https://github.com/JimmyK300/zalo-vecna.git"
BRANCH = "feat/datamate-colab-inference"
# Lần đầu để trống. Khi resume phiên mới, dán commit_sha trong manifest.json của run cũ.
CODE_REVISION = ""
ROOT = Path("/content/zalo-vecna")
if not (ROOT / ".git").exists():
    subprocess.check_call(["git", "clone", "--branch", BRANCH, REPO, str(ROOT)])
if CODE_REVISION:
    subprocess.check_call(["git", "fetch", "origin", CODE_REVISION], cwd=ROOT)
    subprocess.check_call(["git", "checkout", "--detach", CODE_REVISION], cwd=ROOT)
COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt"),
    "transformers==5.15.0", "accelerate>=1.10,<2", "bitsandbytes>=0.49,<0.50",
    "google-api-python-client>=2,<3"])
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
print("M0: code commit", COMMIT)
''')
md('''
M1 dùng Drive API theo folder ID nên không phụ thuộc `/u/1` trong URL.
Đăng nhập tài khoản có quyền đọc Drive nhóm. Kết quả ghi vào My Drive của tài khoản này.
Để chạy train benchmark, đổi DATA_FOLDER_ID sang ID folder train và dùng RUN_NAME mới;
không thay public rulebook bằng train rulebook. Private test sau này cũng cần RUN_NAME mới.
''')
code('''
from google.colab import auth, drive as colab_drive
auth.authenticate_user()
import google.auth
from googleapiclient.discovery import build
from scripts.colab_drive import DriveStore, digest, write_json
DATA_FOLDER_ID = "1vipVjLz9QE2sYMpdRNS4Hs4ZEx6tG9Ml" # public_test của Drive nhóm
RUN_NAME = "public_agentic_nf4_v1"
assert RUN_NAME and all(c.isalnum() or c in "_-" for c in RUN_NAME)
DATA = Path("/content/datamate_input") / DATA_FOLDER_ID / RUN_NAME
credentials, _ = google.auth.default()
store = DriveStore(build("drive", "v3", credentials=credentials, cache_discovery=False))
source_manifest = store.pull(DATA_FOLDER_ID, DATA)
for name in ("questions.jsonl", "rulebook.md", "calendar.csv"):
    assert (DATA / name).is_file(), name
assert (DATA / "data").is_dir()
colab_drive.mount("/content/drive")
PERSIST = Path("/content/drive/MyDrive/DataMate_Submissions") / RUN_NAME
LOCAL = Path("/content/datamate_outputs") / RUN_NAME
PERSIST.mkdir(parents=True, exist_ok=True)
LOCAL.mkdir(parents=True, exist_ok=True)
shutil.copytree(PERSIST, LOCAL, dirs_exist_ok=True)
qs = [json.loads(s) for s in (DATA / "questions.jsonl").read_text(encoding="utf-8-sig").splitlines() if s.strip()]
assert len(qs) == len({str(q["id"]) for q in qs})
write_json(LOCAL / "data_manifest.json", source_manifest)
print("M1:", len(qs), "questions; output", PERSIST)
''')
code('''
import importlib.metadata as metadata
import torch
from datamate.hf_backend import MODEL, REVISION
assert torch.cuda.is_available(), "Chọn GPU runtime"
# Mọi thay đổi cấu hình dùng RUN_NAME mới, để tránh trộn hai submission.
CONSISTENCY = 1
VERIFY = False  # Chưa bật sampled verification vì có báo cáo loop ở checkpoint này.
QUANTIZED = True # False = BF16/FP16; phải dùng RUN_NAME mới.
MAX_TOKENS = 1536
CONTEXT = 32768
TIME_BUDGET_MIN = 330
environment = {p: metadata.version(p) for p in ["torch", "transformers", "bitsandbytes", "pandas", "duckdb"]}
manifest = dict(commit_sha=COMMIT, model=MODEL, revision=REVISION, dataset=digest(source_manifest),
    consistency=CONSISTENCY, verify=VERIFY, quantized=QUANTIZED, max_tokens=MAX_TOKENS,
    context=CONTEXT, time_budget_min=TIME_BUDGET_MIN, packages=environment,
    gpu=torch.cuda.get_device_name(), dtype="bf16" if torch.cuda.is_bf16_supported() else "fp16")
manifest_path = LOCAL / "manifest.json"
if manifest_path.exists():
    assert json.loads(manifest_path.read_text()) == manifest, "Config/code/data/environment changed: restore old settings or use new RUN_NAME."
write_json(manifest_path, manifest)
write_json(LOCAL / "M2.json", {"status": "complete"})
copied = {}
def sync():
    for src in LOCAL.rglob("*"):
        if not src.is_file() or src.suffix == ".tmp":
            continue
        signature = (src.stat().st_mtime_ns, src.stat().st_size)
        if copied.get(src) == signature:
            continue
        dest = PERSIST / src.relative_to(LOCAL)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        shutil.copy2(src, tmp)
        tmp.replace(dest)
        copied[src] = signature
sync()
env = os.environ.copy()
env.update(DATAMATE_MODEL_REVISION=REVISION, DATAMATE_4BIT="1" if QUANTIZED else "0",
    DATAMATE_MAX_TOKENS=str(MAX_TOKENS), DATAMATE_CONTEXT=str(CONTEXT),
    DATAMATE_REASONING="off", PYTHONUNBUFFERED="1")
def run_agent(folder, extra):
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "datamate.run", "--data", str(DATA), "--out", str(folder / "submission.jsonl"),
        "--backend", "transformers", "--model", MODEL, "--workers", "1",
        "--time-budget-min", str(TIME_BUDGET_MIN), *extra]
    print("RUN:", " ".join(command), flush=True)
    with (folder / "console.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        last_sync = 0
        try:
            while process.poll() is None:
                if time.monotonic() - last_sync >= 60:
                    sync()
                    last_sync = time.monotonic()
                    tail = (folder / "console.log").read_text(encoding="utf-8", errors="replace").splitlines()
                    print("\\n".join(tail[-2:]), flush=True)
                time.sleep(5)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            sync()
        if process.returncode:
            raise RuntimeError(f"Agent exited {process.returncode}; inspect {folder / 'console.log'}")
print("M2: manifest locked", manifest)
''')
md('''
M3: bật cờ để thử 3 câu với `candidate-policy=off`, nhằm chắc chắn model/parser/tool thực sự được gọi.
Đọc trace ở thư mục smoke. Thời gian smoke gồm load model; dùng `seconds` từng trace để tham khảo latency.
Smoke không có gold nên chỉ xác nhận kỹ thuật, không chứng minh đáp án đúng.
''')
code('''
RUN_SMOKE = False
if RUN_SMOKE:
    smoke = LOCAL / "smoke"
    run_agent(smoke, ["--sample", "3", "--seed", "3407", "--candidate-policy", "off"])
    traces = [json.loads(p.read_text()) for p in (smoke / "logs").glob("q_*.json")]
    assert len(traces) == min(3, len(qs))
    assert all(t.get("final_call") and not any("error" in x for x in t["trace"]) for t in traces), "Smoke failed: inspect logs before full run"
    write_json(LOCAL / "M3.json", {"status": "complete", "seconds": [t["seconds"] for t in traces]})
    sync()
    print("M3 passed", [(t["id"], t["type"], t["seconds"]) for t in traces])
''')
code('''
RUN_FULL = False
if RUN_FULL:
    assert (LOCAL / "M3.json").exists(), "Pass smoke M3 first"
    options = ["--candidate-policy", "answer", "--consistency", str(CONSISTENCY)]
    if VERIFY:
        assert CONSISTENCY > 1
        options.append("--verify")
    run_agent(LOCAL / "full", options)
    write_json(LOCAL / "M4.json", {"status": "complete"})
    sync()
''')
md('''
M5 kiểm tra đúng tập IDs và định dạng, không chấm độ chính xác vì public test không có gold.
`abstain` được phép, nhưng kiểm tra tỷ lệ và log trước khi nộp. Một lượt hết ngân sách có thể tạo nhiều abstain.
Chạy lại M4 sẽ tiếp tục các câu chưa trả lời và retry abstain; đổi config thì tạo run mới.
Khi Colab ngắt: giữ RUN_NAME, dán commit_sha cũ vào CODE_REVISION, chạy M0–M2 rồi M4/M5.
''')
code('''
EXPORT_SUBMISSION = False
if EXPORT_SUBMISSION:
    from datamate.io_contract import normalize_answer, ANSWER_TYPES
    assert (LOCAL / "M4.json").exists()
    path = LOCAL / "full" / "submission.jsonl"
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    ids = [str(r["id"]) for r in rows]
    assert len(ids) == len(set(ids)) == len(qs)
    assert set(ids) == {str(q["id"]) for q in qs}
    for r in rows:
        assert r["type"] in (*ANSWER_TYPES, "abstain", "unanswerable")
        if r["type"] in ANSWER_TYPES:
            assert normalize_answer(r["type"], r["answer"]) == r["answer"]
    from collections import Counter
    counts = dict(Counter(r["type"] for r in rows))
    logs = [json.loads(p.read_text()) for p in (LOCAL / "full/logs").glob("q_*.json")]
    backend_errors = [r["id"] for r in logs if r.get("type") == "error"]
    write_json(LOCAL / "M5.json", {"status": "complete", "n": len(rows), "types": counts,
        "backend_errors": backend_errors, "recipe_answers": sum(r.get("reason") == "recipe" for r in logs),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    sync()
    print("SUBMISSION:", PERSIST / "full/submission.jsonl", counts)
    print("Backend errors written as abstain:", len(backend_errors), "; inspect logs and rerun M4 before submitting if needed.")
    from google.colab import files
    files.download(str(path))
''')
for i,c in enumerate(cells):
    c['id'] = f'datamate-infer-{i:03}'
out=Path(__file__).resolve().parents[1]/'notebooks/DataMate_Submission_Colab.ipynb'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(dict(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),accelerator='GPU',colab=dict(name=out.name,provenance=[])),nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(out)
