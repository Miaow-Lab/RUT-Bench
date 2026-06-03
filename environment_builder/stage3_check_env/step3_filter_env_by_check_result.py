"""
step3: Filter environments based on check results.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.process_file import read_file, save_file


def _item_key(item):
    """Stable key for resume/deduplication."""
    env_class_name = str(item.get("env_class_name", "")).strip()
    environment_summary = str(item.get("environment_summary", "")).strip()
    return f"{environment_summary}|||{env_class_name}"


def _load_existing_results(file_path):
    try:
        existing = read_file(file_path)
    except Exception:
        return []

    if isinstance(existing, list):
        return existing
    if isinstance(existing, dict):
        return existing
    return []


def _next_env_id(existing_metadata):
    max_id = 0
    for env_id in existing_metadata.keys():
        if isinstance(env_id, str) and env_id.startswith("env_"):
            try:
                max_id = max(max_id, int(env_id.split("_")[-1]))
            except ValueError:
                continue
    return max_id + 1


def get_not_fail_accs(data):
    """Get not_fail accuracy (pass + warning rate) for each environment."""
    not_fail_acc_list = []
    for item in data:
        func_test_result_summary = item["func_test_result"]['func_test_cases']['summary']
        not_fail_acc = round((func_test_result_summary["pass_count"]+ func_test_result_summary["warning_count"]) / func_test_result_summary["total_count"], 2)
        not_fail_acc_list.append(not_fail_acc)
    return not_fail_acc_list

def get_postive_not_fail_accs(data):
    """Get positive not_fail accuracy (positive pass + warning rate) for each environment."""
    postive_not_fail_acc_list = []
    for item in data:
        func_test_result_summary = item["func_test_result"]['func_test_cases']['summary']
        postive_not_fail_acc = round((func_test_result_summary["positive_pass_count"]+ func_test_result_summary["positive_warning_count"]) / func_test_result_summary["positive_count"], 2)
        postive_not_fail_acc_list.append(postive_not_fail_acc)
    return postive_not_fail_acc_list

def get_pass_accs(data):
    """Get pass accuracy (pass rate only) for each environment."""
    pass_acc_list = []
    for item in data:
        func_test_result_summary = item["func_test_result"]['func_test_cases']['summary']
        pass_acc = round(func_test_result_summary["pass_count"] / func_test_result_summary["total_count"], 2)
        pass_acc_list.append(pass_acc)
    return pass_acc_list

def select_env(data, select_field, threshold):
    """Filter environments based on selected field and threshold."""
    new_data = []
    if select_field == "not_fail":
        # Filter by pass + warning rate
        for item in data:
            func_test_result_summary = item["func_test_result"]['func_test_cases']['summary']
            not_fail_acc = (func_test_result_summary["pass_count"]+ func_test_result_summary["warning_count"]) / func_test_result_summary["total_count"]
            if not_fail_acc >= threshold:
                new_data.append(item)
        return new_data
    elif select_field == "pass":
        # Filter by pass rate only
        for item in data:
            func_test_result_summary = item["func_test_result"]['func_test_cases']['summary']
            pass_acc = func_test_result_summary["pass_count"] / func_test_result_summary["total_count"]
            if pass_acc >= threshold:
                new_data.append(item)
        return new_data
    else:
        raise ValueError("unknown select_field: {}".format(select_field))
    
def brief_metadata(data):
    """Simplify environment metadata by removing detailed fields."""
    if isinstance(data, dict):
        iterable = data.values()
    else:
        iterable = data

    for item in iterable:
        if "env_func_details" in item:
            del item["env_func_details"]
        if "func_test_result" in item and "func_test_cases" in item["func_test_result"]:
            item["func_test_result"]["func_test_cases"].pop("details", None)
    return data

def process_env_metadata(data, existing_metadata=None):
    """Process data: append new env metadata while preserving existing env_ids."""
    new_data = dict(existing_metadata or {})
    env_count = _next_env_id(new_data)
    for item in data:
        env_key = _item_key(item)
        existing_env_id = None
        for candidate_env_id, candidate_item in new_data.items():
            if _item_key(candidate_item) == env_key:
                existing_env_id = candidate_env_id
                break

        if existing_env_id is not None:
            env_id = existing_env_id
        else:
            env_id = f"env_{env_count}"
            env_count += 1

        new_item = {
            "env_id": env_id, 
            "environment_summary": item["environment_summary"], 
            "environment_introduction": item["environment_introduction"], 
            "state_space_definition": item["state_space_definition"], 
            "constraints_rules": item["constraints_rules"], 
            "operation_list": item["operation_list"], 
            "env_class_name": item["env_class_name"], 
            "env_class_code": item["env_class_code"], 
            "env_class_def": item["env_class_def"],
            "env_structure": item["env_structure"], 
            "tools": item["tools"]
        }
        new_data[env_id] = new_item
    return new_data
        
    


if __name__ == "__main__":
    import os as _os
    _os.chdir(Path(__file__).resolve().parents[1])
    step2_data = read_file('stage3_check_env/temp_result/step2_roll_check.json')
    step2_data = [item for item in step2_data if "func_test_result" in item]

    existing_selected = _load_existing_results('stage3_check_env/temp_result/step3_selected_env_data.json')
    if isinstance(existing_selected, dict):
        existing_selected = list(existing_selected.values())

    existing_metadata = _load_existing_results('stage3_check_env/final_result/filtered_env_metadata.json')
    if not isinstance(existing_metadata, dict):
        existing_metadata = {}

    processed_keys = {_item_key(item) for item in existing_selected}
    processed_keys.update(_item_key(item) for item in existing_metadata.values())

    remaining_data = [item for item in step2_data if _item_key(item) not in processed_keys]
    print(f"Total checked envs: {len(step2_data)}")
    print(f"Already filtered envs: {len(processed_keys)}")
    print(f"Remaining envs to filter: {len(remaining_data)}")

    # not_fail_acc_list = get_not_fail_accs(step2_data)
    # count = ratio_by_auto_threshold(not_fail_acc_list, 0.05)
    # print(count)
    # postive_not_fail_acc_list = get_postive_not_fail_accs(step2_data)
    # count = ratio_by_auto_threshold(postive_not_fail_acc_list, 0.05)
    # print(count)
    # pass_acc_list = get_pass_accs(step2_data)
    # count = ratio_by_auto_threshold(pass_acc_list, 0.05)
    # print(count)

    threshold = 1.0
    new_selected_env_data = select_env(remaining_data, "not_fail", threshold)
    print(f"Newly selected envs: {len(new_selected_env_data)}")

    merged_selected_map = {}
    for item in existing_selected + new_selected_env_data:
        key = _item_key(item)
        if key and key not in merged_selected_map:
            merged_selected_map[key] = item
    merged_selected_env_data = list(merged_selected_map.values())
    save_file('stage3_check_env/temp_result/step3_selected_env_data.json', merged_selected_env_data)

    env_metadata = process_env_metadata(new_selected_env_data, existing_metadata=existing_metadata)
    save_file('stage3_check_env/final_result/filtered_env_metadata.json', brief_metadata(env_metadata))