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

# --- System prompt for converting environment spec to Python class ---
system_prompt = \
"""You are an AI coding assistant.  
Your job is to translate an environment specification into a Python environment class definition.  
The class should simulate the stateful environment structure (without methods yet).  

You should analyze first and then generate code.

You should follow the rules of Analysis and Code to generate the code.

Rules of Analysis
- Determine the environment class name. It should be EnvironmentSummary or an appropriate adaptation (e.g., `LinuxFileSystem`, `EcommerceOrderSystem`).  
- Extract attribute names (comma-separated) from each entity in `state_space_definition`.
- If needed, generate a corresponding `TypedDict` using the extracted attributes, with attribute name → key and attribute value type → inferred from the appropriate Python primitive type (e.g., `id`=str, `name`=str, `category`=str, `price`/`size`=float/int, `quantity`=int, `status`=str, `timestamps`=str/float).
- `constraints_rules` is left as a comment.

Rules of Code
- Generates each `TypedDict` definition if needed.
- Generates the environment class (with only `__init__` and attributes), with attributes of type `Dict[ID, TypedDict]`.
- Add comments mapping each attribute back to the state space entity/attributes.
- Annotates the constraints in the code comments.
- Do not implement any business logic or methods yet.  

The input format is:
# Environment Summary
<short label, e.g. Linux filesystem, E-commerce order system>"

# Environment Introduction
<paragraph intro>

# State Space Definition
[
    {
      "entity": "EntityName",
      "attributes": "attr1, attr2, ...",
      "description": "short description"
    },
    ...
]

# constraints_rules
constraint 1 ...
constraint 2 ...
}

Your output must follow the format below (do not include any other text):

# Analysis
[Explains how to design Python environment classes based on tasks and state spaces (including class name selection, mapping entities to data structures, which fields are stored as dict/list, and how constraints are expressed through annotations)]

# Class Definition
```python
[Python environment class definition]
```"""

# --- Example cases for few-shot learning ---
input_case_1 = \
"""Given the following Environment, State Space, and Constraints, generate a Python environment class definition accordingly.

# Environment Summary
E-commerce order management system

# Environment Introduction
This environment represents an e-commerce order management system, where users can place orders, view products, and manage their accounts.

# State Space Definition
[
    {
      "entity": "Product",
      "attributes": "product_id, name, category, price, stock_quantity",
      "description": "Represents a product sold on the platform."
    },
    {
      "entity": "Order",
      "attributes": "order_id, customer_id, status, order_items",
      "description": "Represents a purchase order placed by a customer."
    },
    {
      "entity": "OrderItem",
      "attributes": "order_id, product_id, quantity",
      "description": "Intermediate entity linking products to orders."
    }
]

# constraints_rules
- tock quantity cannot drop below 0.
- Only orders with status = 'pending' can be marked as 'fulfilled'."""

output_case_1 = """# Analysis
The task involves updating inventory and order status. The environment is summarized as an "e-commerce order management system," so the class is named `EcommerceOrderManagementSystem`.

Based on the state space:
- The Product entity needs to store a dict with key = product_id and value = metadata (including stock_quantity).
- The Order entity needs to store a dict with key = order_id and value = metadata (including customer_id, status, and order_items).
- The OrderItem entity represents a many-to-many relationship between Order and Product, ideally stored as {order_id: [{product_id, quantity}, ...]}.
 
Extract Entity：  
  • Product → {product_id: str, name: str, category: str, price: float, stock_quantity: int}  
  • Order → {order_id: str, customer_id: str, status: str, order_items: List[OrderItemInfo]}  
  • OrderItem → {order_id: str, product_id: str, quantity: int}  

Use TypedDict to define these structures
- In the environment：  
  self.products: Dict[str, ProductInfo]  
  self.orders: Dict[str, OrderInfo] 
 
Constraints such as "stock ≥ 0" and "order status can only transition from pending to fulfilled" are initially documented in the class as comments and later implemented in method implementations.

# Class Definition
```python
from typing import Dict, List, TypedDict

class ProductInfo(TypedDict):
    product_id: str
    name: str
    category: str
    price: float
    stock_quantity: int

class OrderItemInfo(TypedDict):
    order_id: str
    product_id: str
    quantity: int

class OrderInfo(TypedDict):
    order_id: str
    customer_id: str
    status: str
    order_items: List[OrderItemInfo]

class EcommerceOrderManagementSystem:
    def __init__(self):
        \"\"\"
        The environment for e-commerce order management.
        \"\"\"

        # Products: {product_id: ProductInfo}
        self.products: Dict[str, ProductInfo] = {}

        # Orders: {order_id: OrderInfo}
        self.orders: Dict[str, OrderInfo] = {}

        # Constraints reminder:
        # - Stock quantity cannot drop below 0
        # - Only orders with status = 'pending' can be marked as 'fulfilled'

        self.current_user: dict = {}
```"""

# --- Input template for environment class generation ---
input_template = """Given the following Environment, State Space, and Constraints, generate a Python environment class definition accordingly.

# Environment Summary
{env_summary}

# Environment Introduction
{env_introduction}

# State Space Definition
{state_space_definition}

# constraints_rules
{constraints_rules}
"""  


def _item_key(item):
    """Stable key for resume/deduplication."""
    task = str(item.get("task", "")).strip()
    env_summary = str(item.get("environment_summary", "")).strip()
    return f"{env_summary}|||{task}"


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


def _build_ordered_results(raw_data, result_map):
    ordered = []
    for item in raw_data:
        key = _item_key(item)
        if key in result_map:
            ordered.append(result_map[key])
    return ordered

def parse_response(response):
    """Parse LLM response to extract class definition."""
    if "# Analysis" in response and "# Class Definition" in response:
        try:
            analysis = response.split("# Analysis")[1].split("# Class Definition")[0].strip()
            # Handle potential markdown code block variations
            code_part = response.split("# Class Definition")[1].strip()
            if "```python" in code_part:
                class_definition = code_part.split("```python")[1].split("```")[0].strip()
            elif "```" in code_part:
                class_definition = code_part.split("```")[1].split("```")[0].strip()
            else:
                class_definition = code_part
            return True, class_definition
        except Exception as e:
            print(f"Error parsing response: {e}")
            return False, response
    else:
        print(f"Format mismatch in response: {response[:100]}...")
        return False, response

def construct_messages(env_item):
    """Construct messages for LLM inference based on environment item."""
    state_space_definition_str = json.dumps(env_item.get("state_space_definition", []), indent=4, ensure_ascii=False)
    constraint_str = ""
    for constraint in env_item.get("constraints_rules", []):
        constraint_str += f"- {constraint}\n"
        
    input_content = input_template.format(
        env_summary=env_item.get("environment_summary", ""), 
        env_introduction=env_item.get("environment_introduction", ""), 
        state_space_definition=state_space_definition_str,
        constraints_rules=constraint_str
    )
    
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": input_case_1},
        {"role": "assistant", "content": output_case_1},
        {"role": "user", "content": input_content}
    ]
    return messages

def process_env_item(env_item, model):
    """Process a single environment item to generate class definition with retries."""
    new_env_item = deepcopy(env_item)
    messages = construct_messages(env_item)
    
    cur_try = 0
    max_try = 5
    final_class_def = ""
    
    while cur_try < max_try:
        try:
            response = llm_inference(
                provider="openai",
                model=model,
                messages=messages
            )
            success, class_definition = parse_response(response)
            if success:
                final_class_def = class_definition
                break
        except Exception as e:
            print(f"LLM call error on try {cur_try + 1}: {e}")
        cur_try += 1
        
    new_env_item["class_definition"] = final_class_def
    return new_env_item

def main(read_file_path, save_file_path, model, max_workers=10):
    """Main function: generate class definitions in parallel."""
    raw_data = read_file(read_file_path)
    existing_results = _load_existing_results(save_file_path)
    result_map = {}
    for item in existing_results:
        key = _item_key(item)
        if key:
            result_map[key] = item

    remaining_data = [item for item in raw_data if _item_key(item) not in result_map]

    print(f"Generating Python classes for {len(raw_data)} environments using {max_workers} workers...")
    if result_map:
        print(f"Resuming from existing output: loaded {len(result_map)} completed items")
    print(f"Remaining items to generate: {len(remaining_data)}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a mapping of futures to their original environment items
        futures = {executor.submit(process_env_item, item, model): item for item in remaining_data}
        completed_this_run = 0
        
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Generating Classes")):
            try:
                result = future.result()
                result_map[_item_key(result)] = result
                completed_this_run += 1
            except Exception as e:
                orig_item = futures[future]
                print(f"Critical error on item '{orig_item.get('environment_summary')}': {e}")

            # Periodic save
            if completed_this_run > 0 and completed_this_run % 10 == 0:
                save_file(save_file_path, _build_ordered_results(raw_data, result_map))

    save_file(save_file_path, _build_ordered_results(raw_data, result_map))
    print(f"Done. Final results saved to: {save_file_path}")

if __name__ == "__main__":
    import os as _os
    _os.chdir(Path(__file__).resolve().parents[1])
    # --- Configuration ---
    MODEL = "gpt-5-mini"  # Or your specific model version
    READ_PATH = "stage2_syn_env/temp_result/step1_infer_state.json"
    SAVE_PATH = "stage2_syn_env/temp_result/step2_infer_state_code.json"
    WORKERS = 10

    main(
        read_file_path=READ_PATH, 
        save_file_path=SAVE_PATH, 
        model=MODEL,
        max_workers=WORKERS
    )