"""
step0: collect tasks from existing instruction-following datasets (ToolAce, API-Bank, Dolci).
"""
# add project root directory to sys.path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
import polars as pl
from utils.process_file import read_file, save_file


BASE_DIR = Path(__file__).parent


def contains_non_english(s):
    """Check if task contains non-English characters."""
    # English letters + numbers + common English punctuation + whitespace
    allowed_pattern = r'^[A-Za-z0-9\s\.\,\?\!\;\:\'\"\(\)\[\]\{\}\-\_\*/\\@#\$%\^&\+\=<>\|~`]*$'
    return not re.match(allowed_pattern, s)


def is_multimodal_task(task_str):
    """Check if task string contains multimodal keywords."""
    keywords = ["image", "photo", "picture", "video", "audio", "sound", "speech", "clip"]
    return any(kw.lower() in task_str.lower() for kw in keywords)


def clean_task_text(task):
    """Normalize extracted task text and remove noisy wrapping quotes."""
    if not isinstance(task, str):
        return None

    task = task.strip()
    while len(task) >= 2 and task[0] == task[-1] and task[0] in {'"', "'"}:
        task = task[1:-1].strip()
    task = task.strip(" \t\n\r\"'`*")
    return task or None



def extract_task(item, dataset_name):
    """Extract task from a single sample based on dataset format."""
    if dataset_name == 'toolace':
        assert item['conversations'][0]['from'] == 'user'
        task = item['conversations'][0]['value']
    elif dataset_name == "api-bank":
        task = item["input"].split("User:")[1].split("\nGenerate API Request: ")[0].split("\nAPI-Request: ")[0].split("TIME: ")[0].strip()
        if '\nAI: ' in task:
            task = task.split('\nAI: ')[0]
    elif dataset_name == "dolci":
        task = extract_first_user_message(item.get("messages", []))
    else:
        return False, None

    task = clean_task_text(task)
    if not task:
        return False, None

    # Filter non-English tasks
    if contains_non_english(task):
        return False, None
    # Filter multimodal tasks
    if is_multimodal_task(task):
        return False, None
    # Filter tasks with special characters/keywords
    if "Role definition:" in task or 'USD' in task or 'ETH' in task or 'Bitcoin' in task or 'Ethereum' in task or '@' in task or '.com' in task or 'http:' in task or 'https:' in task:
        return False, None

    return True, task


def extract_first_user_message(messages):
    """Return the first user message content from a messages list."""
    if not isinstance(messages, list):
        return None

    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


def deduplicate_tasks(tasks, task_from):
    """Deduplicate tasks deterministically and wrap them with source metadata."""
    unique_tasks = sorted(set(tasks))
    return [{"task": task, "task_from": task_from} for task in unique_tasks]


def extract_api_bank():
    """Extract tasks from API-Bank dataset."""
    data = []
    data.extend(read_file("stage1_collect_env_from_task/source_data/api-bank/lv1-train.json"))
    data.extend(read_file("stage1_collect_env_from_task/source_data/api-bank/lv2-train.json"))
    data.extend(read_file("stage1_collect_env_from_task/source_data/api-bank/lv3-train.json"))
    tasks = []
    for item in data:
        success, task = extract_task(item, dataset_name='api-bank')
        if success:
            tasks.append(task)
    return deduplicate_tasks(tasks, task_from="api-bank")


def extract_toolace():
    """Extract tasks from ToolAce dataset."""
    data = read_file('stage1_collect_env_from_task/source_data/toolace/data.json')
    tasks = []
    for item in data:
        success, task = extract_task(item, dataset_name='toolace')
        if success:
            tasks.append(task)
    return deduplicate_tasks(tasks, task_from="toolace")


def extract_dolci():
    """Extract tasks from Dolci parquet files using polars."""
    dolci_dir = BASE_DIR / "source_data" / "dolci"
    parquet_paths = sorted(dolci_dir.rglob("*.parquet"))

    tasks = []
    for parquet_path in parquet_paths:
        df = pl.read_parquet(parquet_path)
        for item in df.select("messages").to_dicts():
            success, task = extract_task(item, dataset_name="dolci")
            if success:
                tasks.append(task)

    return deduplicate_tasks(tasks, task_from="dolci")
    

if __name__ == "__main__":
    import os as _os
    _os.chdir(Path(__file__).resolve().parents[1])
    task_data_1 = extract_api_bank()
    print(len(task_data_1))
    task_data_2 = extract_toolace()
    print(len(task_data_2))
    task_data_3 = extract_dolci()
    print(len(task_data_3))
    task_data = task_data_1 + task_data_2 + task_data_3
    print(len(task_data))
    save_file("stage1_collect_env_from_task/temp_result/step0_source_tasks.json", task_data)