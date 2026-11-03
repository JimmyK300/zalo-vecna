"""Drive API download and checkpoint sync helpers for Colab."""
import hashlib
import json
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()

def file_digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)

class DriveStore:
    """API mirror: no dependence on /u/1, shortcuts, or MyDrive mount paths."""
    def __init__(self, service):
        self.api = service

    def children(self, folder):
        result, token = [], None
        while True:
            page = self.api.files().list(
                q=f"'{folder}' in parents and trashed = false", pageSize=1000,
                pageToken=token, fields="nextPageToken,files(id,name,mimeType,size,md5Checksum)",
                supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
            result.extend(page.get("files", []))
            token = page.get("nextPageToken")
            if not token:
                return result

    def folder(self, parent, name):
        found = [f for f in self.children(parent) if f["name"] == name]
        if len(found) > 1:
            raise RuntimeError(f"Duplicate Drive name: {name}. Resolve duplicates before resuming.")
        if found:
            if found[0]["mimeType"] != "application/vnd.google-apps.folder":
                raise RuntimeError(f"Not a folder: {name}")
            return found[0]["id"]
        return self.api.files().create(body={"name": name, "parents": [parent],
            "mimeType": "application/vnd.google-apps.folder"}, fields="id",
            supportsAllDrives=True).execute()["id"]

    def pull(self, folder, destination):
        from googleapiclient.http import MediaIoBaseDownload
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        manifest = []
        seen = set()
        for f in sorted(self.children(folder), key=lambda x: x["name"]):
            name = f["name"]
            if name in seen or name in (".", "..") or "/" in name or "\\" in name:
                raise ValueError(f"Unsafe/duplicate Drive filename: {name!r}")
            seen.add(name)
            p = destination / name
            if f["mimeType"] == "application/vnd.google-apps.folder":
                for item in self.pull(f["id"], p):
                    manifest.append({**item, "path": name + "/" + item["path"]})
                continue
            if f["mimeType"].startswith("application/vnd.google-apps."):
                raise ValueError(f"Expected released binary/text file, got {f['mimeType']}: {name}")
            expected = f.get("md5Checksum")
            good = p.is_file() and expected and file_digest(p, "md5") == expected
            if not good:
                tmp = p.with_suffix(p.suffix + ".download")
                with tmp.open("wb") as stream:
                    download = MediaIoBaseDownload(stream, self.api.files().get_media(
                        fileId=f["id"], supportsAllDrives=True), chunksize=8 * 1024 * 1024)
                    done = False
                    while not done:
                        _, done = download.next_chunk(num_retries=5)
                if expected and file_digest(tmp, "md5") != expected:
                    raise IOError(f"Checksum mismatch: {name}")
                tmp.replace(p)
            manifest.append({"path": name, "id": f["id"], "md5": expected,
                             "size": p.stat().st_size})
        return manifest

    def push(self, source, folder):
        from googleapiclient.http import MediaFileUpload
        source = Path(source)
        children = self.children(folder)
        if len({f["name"] for f in children}) != len(children):
            raise RuntimeError("Duplicate output names on Drive; resolve before sync.")
        by_name = {f["name"]: f for f in children}
        # Completion marker is uploaded last within each checkpoint folder.
        for p in sorted(source.iterdir(), key=lambda p: (p.name in ("_COMPLETE.json", "milestone.json"), p.name)):
            if p.name.endswith((".tmp", ".download")) or p.name == "__pycache__":
                continue
            old = by_name.get(p.name)
            if p.is_dir():
                self.push(p, self.folder(folder, p.name))
                continue
            if old and old.get("md5Checksum") == file_digest(p, "md5"):
                continue
            media = MediaFileUpload(str(p), resumable=True, chunksize=8 * 1024 * 1024)
            if old:
                req = self.api.files().update(fileId=old["id"], media_body=media, supportsAllDrives=True)
            else:
                req = self.api.files().create(body={"name": p.name, "parents": [folder]},
                    media_body=media, fields="id", supportsAllDrives=True)
            response = None
            while response is None:
                _, response = req.next_chunk(num_retries=5)
