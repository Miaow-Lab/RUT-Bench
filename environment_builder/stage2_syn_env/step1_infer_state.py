import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import os
from tqdm import tqdm
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.call_llm import llm_inference
from utils.process_file import read_file, save_file

# --- System prompt for state space inference ---
system_prompt = \
"""You are an expert task and environment analyst.  
Given an environment description and a example task in this environment, infer the set of state variables (state space) that the environment maintained.  

The state should not be too broad (e.g. "all possible data in an e-commerce system"), nor too narrow (only for this single task).  Instead, reasonably design it to support this task and similar tasks in the same environment.  

The input format is:

# Environment Summary
[Environment summary]

# Environment Introduction
[Environment introduction]

# A Example Task in This Environment
[Example task]


Your output must follow the format below (do not include any other text):

# Analysis
[Your thought process: What states are involved in the environment? What entities/attributes need to be tracked? What constraints or rules exist in the environment? ……]

# State Space Definition
- Entity: EntityName1  
  - Attributes: Attribute1, Attribute2, ...
  - Description: The role of this entity in the environment

- Entity: EntityName2
  - Attributes: ...
  - Description: ...

# Constraints & Rules
- Constraint 1
- Constraint 2
...
"""

# --- Example cases for few-shot learning ---
input_case_1 = \
"""Analyze the following task and environment, and infer the set of state variables (state space) that the environment maintained.  

# Environment Summary
E-commerce order management system

# Environment Introduction
This environment consists of a stateful backend for an e-commerce platform, managing products, orders, inventory, and order statuses.  
It keeps records of which products have been purchased in each order, tracks real-time stock quantities for all products, and stores fulfillment information for each order.  
These features make it the natural setting for inventory adjustments and order status updates in response to customer purchases.

# A Example Task in This Environment
Reduce stock quantity by 1 for every product purchased in order #58291 and mark the order as fulfilled.
"""

output_case_1 = \
"""# Analysis
This task requires knowing which products belong to a given order, the quantity of each product, and their current stock levels.  
It also requires an order to have a modifiable status.  
Therefore, the environment must maintain entities for orders, products, and inventory, along with attributes like stock quantity and order state.  

# State Space Definition
- Entity: Product  
  - Attributes: product_id, name, category, price, stock_quantity  
  - Description: Represents a product sold on the platform, with inventory tracking via stock_quantity.

- Entity: Order  
  - Attributes: order_id, customer_id, status, order_items  
  - Description: Represents a purchase order placed by a customer. Includes current order status and the list of items purchased.

- Entity: OrderItem 
  - Attributes: order_id, product_id, quantity  
  - Description: Represents the many-to-many relationship between orders and products, with quantities ordered.

- Entity: Customer 
  - Attributes: customer_id, name, account_status  
  - Description: Represents the user placing the order, useful for related tasks.

# Constraints & Rules
- Stock quantity cannot drop below 0.  
- Only orders with status = "pending" can be marked as "fulfilled".  
- Each product in an order must exist in the product inventory."""

# --- Input template for state space inference ---
input_template = \
"""Analyze the following task and environment, and infer the set of state variables (state space) that the environment maintained.  

# Environment Summary
{env_summary}

# Environment Introduction
{env_introduction}

# A Example Task in This Environment
{task}
"""


def _item_key(item):
    """Stable key for resume/deduplication."""
    task = str(item.get("task", "")).strip()
    env_summary = str(item.get("environment_summary", "")).strip()
    return f"{env_summary}|||{task}"


def _load_existing_results(save_path):
    """Load existing partial output for resume if available."""
    try:
        existing = read_file(save_path)
    except Exception:
        return []

    if isinstance(existing, list):
        return existing
    if isinstance(existing, dict):
        return list(existing.values())
    return []


def _build_ordered_results(raw_data, result_map):
    ordered = []
    for item in raw_data:
        key = _item_key(item)
        if key in result_map:
            ordered.append(result_map[key])
    return ordered

def parse_state_space_definition(state_space_definition):
    """Parse state space definition into list of dictionaries with entity, attributes, description."""
    if "- Entity: " not in state_space_definition:
        print(f"Error parsing state space definition (missing markers): {state_space_definition[:100]}...")
        return []
    
    entities = state_space_definition.split("- Entity: ")[1:]
    parsed_results = []
    
    for entity_block in entities:
        lines = [line.strip() for line in entity_block.split("\n") if line.strip()]
        if len(lines) < 3:
            continue
        
        entity_name = lines[0]
        attributes = lines[1].replace("- Attributes: ", "").strip()
        description = lines[2].replace("- Description: ", "").strip()
        
        parsed_results.append({
            "entity": entity_name,
            "attributes": attributes,
            "description": description
        })
    return parsed_results

def parse_constraints_rules(constraints_rules):
    """Parse constraints and rules by splitting on '- ' pattern."""
    if "- " not in constraints_rules:
        return []
    constraints_list = constraints_rules.split("- ")[1:]
    return [c.strip() for c in constraints_list if c.strip()]

def parse_response(response):
    """Parse LLM response to extract analysis, state space definition, and constraints."""
    if "# Analysis" in response and "# State Space Definition" in response and "# Constraints & Rules" in response:
        try:
            analysis = response.split("# Analysis")[1].split("# State Space Definition")[0].strip()
            state_space_definition_raw = response.split("# State Space Definition")[1].split("# Constraints & Rules")[0].strip()
            constraints_rules_raw = response.split("# Constraints & Rules")[1].strip()
            
            ss_def = parse_state_space_definition(state_space_definition_raw)
            cons_rules = parse_constraints_rules(constraints_rules_raw)
            
            return analysis, ss_def, cons_rules
        except Exception as e:
            print(f"Exception during parsing: {e}")
            return None, None, None
    else:
        return None, None, None

def process_env_item(env_item, model):
    """Process a single environment item to infer state space with retry logic."""
    new_env_item = deepcopy(env_item)
    input_content = input_template.format(
        env_summary=env_item.get("environment_summary", ""),
        env_introduction=env_item.get("environment_introduction", ""),
        task=env_item.get("task", "")
    )
    
    cur_try = 0
    max_try = 3
    final_analysis, final_ss, final_rules = None, None, None
    
    while cur_try < max_try:
        cur_try += 1
        try:
            response = llm_inference(
                provider="openai",
                model=model, 
                messages=[
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": input_case_1},
                    {"role": "assistant", "content": output_case_1},
                    {"role": "user", "content": input_content}
                ]
            )
            analysis, state_space_definition, constraints_rules = parse_response(response)
            if state_space_definition and constraints_rules:
                final_analysis = analysis
                final_ss = state_space_definition
                final_rules = constraints_rules
                break
        except Exception as e:
            print(f"Try {cur_try} failed for item {env_item.get('environment_summary', 'unknown')}: {e}")

    new_env_item["state_space_analysis"] = final_analysis
    new_env_item["state_space_definition"] = final_ss
    new_env_item["constraints_rules"] = final_rules
    return new_env_item

def main(read_path, save_path, model, max_workers=10):
    """Main entry point to run inference in parallel."""
    raw_data = read_file(read_path)
    existing_results = _load_existing_results(save_path)
    result_map = {}
    for item in existing_results:
        key = _item_key(item)
        if key:
            result_map[key] = item

    remaining_data = [item for item in raw_data if _item_key(item) not in result_map]

    print(f"Starting state space inference for {len(raw_data)} items...")
    if result_map:
        print(f"Resuming from existing output: loaded {len(result_map)} completed items")
    print(f"Remaining items to infer: {len(remaining_data)}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        futures = {executor.submit(process_env_item, item, model): item for item in remaining_data}
        completed_this_run = 0
        
        # Track progress
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Inferring States")):
            try:
                result = future.result()
                result_map[_item_key(result)] = result
                completed_this_run += 1
            except Exception as e:
                orig_item = futures[future]
                print(f"Critical error processing {orig_item.get('environment_summary', 'unknown')}: {e}")

            # Save periodically
            if completed_this_run > 0 and completed_this_run % 10 == 0:
                save_file(save_path, _build_ordered_results(raw_data, result_map))

    # Final save
    save_file(save_path, _build_ordered_results(raw_data, result_map))
    print(f"Inference complete. Results saved to: {save_path}")

if __name__ == "__main__":
    import os as _os
    _os.chdir(Path(__file__).resolve().parents[1])
    # Settings
    TARGET_MODEL = "gpt-5-mini"
    READ_FILE_PATH = os.getenv(
        "STEP1_INFER_STATE_INPUT",
        "stage1_collect_env_from_task/final_result/env_description_daily_10.json",
    )
    SAVE_FILE_PATH = "stage2_syn_env/temp_result/step1_infer_state.json"
    MAX_CONCURRENCY = 10

    main(
        read_path=READ_FILE_PATH,
        save_path=SAVE_FILE_PATH,
        model=TARGET_MODEL,
        max_workers=MAX_CONCURRENCY
    )