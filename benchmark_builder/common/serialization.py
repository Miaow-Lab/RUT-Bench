import json
from pathlib import Path
from typing import Any


def read_json_file(file_path: str | Path) -> Any:
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl_file(file_path: str | Path) -> list[dict[str, Any]]:
    path = Path(file_path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_json_file(file_path: str | Path, payload: Any) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_jsonl_file(file_path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str))
            handle.write("\n")
