"""
step1: Generate test initialization config for each environment to instantiate it. (Unlike ScenGenerator, no task scenario is needed here.)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from tqdm import tqdm
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.call_llm import llm_inference
from utils.process_file import read_file, save_file
from utils.tool_parser import extract_json_from_text


def _item_key(item):
    """Stable key for resume/deduplication."""
    env_class_name = str(item.get("env_class_name", "")).strip()
    environment_summary = str(item.get("environment_summary", "")).strip()
    return f"{environment_summary}|||{env_class_name}"


def _load_existing_results(save_file_path):
    """Load existing partial output for resume if available."""
    try:
        existing = read_file(save_file_path)
    except Exception:
        return []

    if isinstance(existing, list):
        return existing
    if isinstance(existing, dict):
        return list(existing.values())
    return []


def _is_completed(item, gen_config_num):
    init_config_list = item.get("init_config_list", [])
    return isinstance(init_config_list, list) and len(init_config_list) >= gen_config_num


def _build_ordered_results(raw_data, result_map):
    ordered = []
    for item in raw_data:
        key = _item_key(item)
        if key in result_map:
            ordered.append(result_map[key])
    return ordered


def _merge_resume_item(raw_item, existing_item):
    merged_item = deepcopy(raw_item)
    existing_configs = existing_item.get("init_config_list", [])
    if isinstance(existing_configs, list) and existing_configs:
        merged_item["init_config_list"] = deepcopy(existing_configs)
    return merged_item



# Prompt template for generating environment initialization config
input_template = \
"""You are an AI assistant.  
You will be given the complete definition of a Python class.  
This class represents an environment state in a specific domain and contains various attributes (such as dictionaries, lists, `TypedDict` objects, dataclasses, etc.) used to manage entities and their relationships within the system.

Based on the class definition, generate a JSON object that can serve directly as the class's initialization configuration (`config`), following these rules:

---

### 1. Structure and Type Matching  
- The JSON must strictly follow the attribute structure and data types required by the class.  
- Field names, nesting levels, and value types must match the class definition exactly.

### 2. Respect Constraints  
- Read the class methods and docstrings to identify constraints (e.g., valid status values, required fields, ID reference rules), ensuring all generated data complies.  
- All references (e.g., `reporter_id`, `location_id`, `disease_name`) must be cross-linked appropriately and valid.  
- Consider cross-entity relationships and constraints (e.g., a product must belong to an existing category).

### 3. Richness of Data  
- Each major dictionary-like attribute should contain multiple entities (recommended at least 3-5 entries) with differentiated content to avoid repetitive templates.  
- Cover the different states and value ranges supported by the class wherever possible.  
- Dates should be distributed over a reasonable time span to provide diversity.  
- Numerical fields (e.g., `case_count`) should vary in range to simulate realistic system data.

### 4. Realistic Simulation of Data  
- Name fields should use natural-language fictional content (e.g., `"Alice Chan"`, `"Central City District"`) rather than mechanical placeholders like `name1` or `user001`.  
- Description fields should be concise, natural, and logically consistent with the domain's context.  
- Date fields must be in ISO format (`YYYY-MM-DD`) or timestamps, with dates reasonably distributed in time.  
- ID fields may mix short codes (e.g., `LOC1`, `REP1`) and UUIDs, but all must be unique.  
- Data must be fictitious and must not contain any real-world personal or sensitive information.

### 5. Output Format  
- Output only the JSON, without any extra explanation.  
- The JSON must be a complete, ready-to-use initialization configuration that can be passed directly to the class constructor as the `config` parameter.

---

### Env Class Definition
```python
{env_class_code}
```

### All Containers
{all_containers}

---

Strictly follow the following output format:

# Analysis
[Your reasoning: what are the containers, what fields they require, what constraints apply, and how you chose the sample data, etc.]

# Init Config
```json
{{
    ...
}}
```"""

def parse_response(response):
    """Parse LLM response to extract analysis and initialization config JSON."""
    analysis = ""
    candidate = response

    if "# Analysis" in response and "# Init Config" in response:
        analysis = response.split("# Analysis", 1)[1].split("# Init Config", 1)[0].strip()
        candidate = response.split("# Init Config", 1)[1].strip()
    elif "# Init Config" in response:
        candidate = response.split("# Init Config", 1)[1].strip()

    try:
        init_config = extract_json_from_text(candidate)
        # Accept common variants and normalize to a dict init config.
        if isinstance(init_config, dict):
            # Some models wrap payload under a single key.
            for key in ("init_config", "config", "data"):
                if isinstance(init_config.get(key), dict):
                    return analysis or response, init_config[key]
            return analysis or response, init_config

        # Some models return a one-element list of configs.
        if isinstance(init_config, list) and init_config and isinstance(init_config[0], dict):
            return analysis or response, init_config[0]

        print("Error parsing response: extracted JSON is not an object")
        return response, None
    except Exception as e:
        preview = (candidate or "")[:300].replace("\n", " ")
        print(f"Error parsing response: {e} | preview: {preview}")
        return response, None
    
def gen_init_config(env_class_code, all_containers, model, temperature):
    """Generate initialization config using LLM, retry up to max_try times if parsing fails."""
    cur_try = 0
    max_try = 3
    init_config = None
    input_content = input_template.format(env_class_code=env_class_code,
                                          all_containers=all_containers)
    # Retry if parsing fails
    while cur_try < max_try:
        response = llm_inference(
            provider="openai",
            model=model,
            messages=[{"role": "user", "content": input_content}],
            response_format={"type": "json_object"},
            temperature=temperature)
        gen_init_config_analysis, init_config = parse_response(response)
        if init_config:
            break
        cur_try += 1
    return init_config
    

def process_env_item(env_item, gen_config_num, model, temperature):
    """Process environment item to generate required number of initialization configs."""
    env_class_code = env_item["env_class_code"]
    # Extract all containers except init_config
    all_containers = {
        k: v
        for k, v in env_item["env_structure"]["states"].items()
        if k != "init_config"
    }
    # Initialize or copy existing init_config_list
    if "init_config_list" not in env_item:
        init_config_list = []  
    else:
        init_config_list = deepcopy(env_item["init_config_list"])
    
    # Generate only the remaining configs needed
    gen_config_num = gen_config_num - len(init_config_list)
    for i in range(gen_config_num):
        init_config = gen_init_config(env_class_code=env_class_code, all_containers=all_containers, model=model, temperature=temperature)
        if init_config:
            init_config_list.append(init_config)
    new_item = deepcopy(env_item)
    new_item["init_config_list"] = init_config_list
    return new_item

# def main(read_file_path, save_file_path, gen_config_num, model, temperature):
#     raw_data = read_file(read_file_path)
#     new_data = []
#     for env_item in tqdm(raw_data, desc="Gen init config"):
#         new_item = process_env_item(env_item, gen_config_num, model, temperature)
#         new_data.append(new_item)
#         if len(new_data) % 1 == 0:
#             save_file(save_file_path, new_data)
#     save_file(save_file_path, new_data)

def main(read_file_path, save_file_path, gen_config_num, model, temperature, max_workers=20):
    """Process all environment items in parallel and save results periodically."""
    raw_data = read_file(read_file_path)
    existing_results = _load_existing_results(save_file_path)
    existing_map = {}
    result_map = {}
    for item in existing_results:
        key = _item_key(item)
        if key and key not in existing_map:
            existing_map[key] = item
        if key and key not in result_map and _is_completed(item, gen_config_num):
            result_map[key] = item

    remaining_data = []
    for item in raw_data:
        key = _item_key(item)
        if key in result_map:
            continue
        if key in existing_map:
            remaining_data.append(_merge_resume_item(item, existing_map[key]))
        else:
            remaining_data.append(item)

    if result_map:
        print(f"Resuming from existing output: loaded {len(result_map)} completed items")
    print(f"Remaining items to generate: {len(remaining_data)}")

    if not remaining_data:
        sorted_data = _build_ordered_results(raw_data, result_map)
        print("Save to file: {}".format(save_file_path))
        save_file(save_file_path, sorted_data)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(process_env_item, env_item, gen_config_num, model, temperature): env_item
            for env_item in remaining_data
        }
        for future in tqdm(as_completed(future_to_item), total=len(remaining_data)):
            result = future.result()
            result_map[_item_key(result)] = result

            if len(result_map) % 3 == 0:
                sorted_data = _build_ordered_results(raw_data, result_map)
                save_file(save_file_path, sorted_data)

    sorted_data = _build_ordered_results(raw_data, result_map)
    print("Save to file: {}".format(save_file_path))
    save_file(save_file_path, sorted_data)



if __name__ == "__main__":
    import os as _os
    _os.chdir(Path(__file__).resolve().parents[1])
    model = "gpt-4o-mini"
    read_file_path = "stage2_syn_env/final_result/env_with_code.json"
    save_file_path = "stage3_check_env/temp_result/step1_gen_test_init_config.json"
    gen_config_num = 1
    temperature = 0.6
    main(read_file_path, save_file_path, gen_config_num, model, temperature)
    