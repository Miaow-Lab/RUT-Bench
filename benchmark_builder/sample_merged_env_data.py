import argparse
import random
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_builder.common.serialization import read_json_file, save_json_file
from benchmark_builder.config import DEFAULT_MERGED_ENV_DATA, PROJECT_ROOT


DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "benchmark_builder" / "output" / "test8" / "merged_env_data.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly sample environments from merged_env_data.json")
    parser.add_argument("--input-file", default=str(DEFAULT_MERGED_ENV_DATA), help="Path to source merged_env_data.json")
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE), help="Path to sampled merged_env_data.json")
    parser.add_argument("--sample-size", type=int, default=64, help="Number of environments to sample")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    return parser.parse_args()


def _sample_from_mapping(payload: dict[str, Any], sample_size: int, rng: random.Random) -> dict[str, Any]:
    keys = list(payload.keys())
    if sample_size > len(keys):
        raise ValueError(f"sample_size={sample_size} exceeds total environments={len(keys)}")
    selected_keys = rng.sample(keys, sample_size)
    return {key: payload[key] for key in selected_keys}


def _sample_from_items_list(payload: list[Any], sample_size: int, rng: random.Random) -> list[Any]:
    if sample_size > len(payload):
        raise ValueError(f"sample_size={sample_size} exceeds total environments={len(payload)}")
    return rng.sample(payload, sample_size)


def sample_payload(payload: Any, sample_size: int, seed: int | None) -> Any:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    rng = random.Random(seed)

    if isinstance(payload, dict):
        if isinstance(payload.get("items"), list):
            sampled_items = _sample_from_items_list(payload["items"], sample_size, rng)
            result = dict(payload)
            result["items"] = sampled_items
            return result
        return _sample_from_mapping(payload, sample_size, rng)

    if isinstance(payload, list):
        return _sample_from_items_list(payload, sample_size, rng)

    raise TypeError("Unsupported JSON structure: expected dict or list")


def main() -> None:
    args = parse_args()

    source_path = Path(args.input_file)
    output_path = Path(args.output_file)

    payload = read_json_file(source_path)
    sampled_payload = sample_payload(payload, sample_size=int(args.sample_size), seed=args.seed)
    save_json_file(output_path, sampled_payload)

    total_count: int
    sampled_count: int
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        total_count = len(payload["items"])
        sampled_count = len(sampled_payload.get("items", [])) if isinstance(sampled_payload, dict) else 0
    elif isinstance(payload, dict):
        total_count = len(payload)
        sampled_count = len(sampled_payload) if isinstance(sampled_payload, dict) else 0
    else:
        total_count = len(payload) if isinstance(payload, list) else 0
        sampled_count = len(sampled_payload) if isinstance(sampled_payload, list) else 0

    print(
        f"Sampled {sampled_count}/{total_count} environments from {source_path} to {output_path}."
    )


if __name__ == "__main__":
    main()
