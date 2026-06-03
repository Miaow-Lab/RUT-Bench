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

# --- System prompt for operation list inference ---
system_prompt = \
"""You are an expert in building and analyzing agent environments.
Given an environment summary, introduction, state space definition, constraint rules, Python base class definition, and example task, your goal is to analyze the current environment and then generate the list of operations needed to support the task in this environment (including information query class and state modification class).
Each operation will be converted into a class function for the Agent to use in subsequent steps.


Key Points:  
- Operations are divided into 2 categories: **Information Query Class** and **State Change Class**.  
- Each operation includes: operation name + brief description.  
- Before output, you must first write **# Analysis**: explain task logic → which are query operations, which are state change operations → and how constraints are related.  

Input Format:
Based on the following environment specification, produce the operation list.

{
  "environment_summary": "...",
  "environment_introduction": "...",
  "state_space_definition": [...],
  "constraints_rules": [...],
  "environment_class_definition": "...",
  "environment_example_task": "...",
}



Strictly maintain the following Output Format:

# Analysis
[Explain operation requirements + classification logic + how constraints affect + ……]

# Operation List
## Information Query Class
- Operation: OperationName Description: xxxx  
- Operation: OperationName Description: xxxx 
- ……

## State Change Class
- Operation: OperationName Description: xxxx  
- Operation: OperationName Description: xxxx  
- ……"""

# --- Example cases for few-shot learning ---
input_case_1 = \
"""Based on the following environment specification, produce the operation list.

  {
    "environment_summary": "E-commerce order management system",
    "environment_introduction": "This environment models the backend of an e-commerce platform, where users can browse, place, and manage orders.  \\nIt keeps track of user accounts, their order histories, and order statuses such as pending, shipped, or completed.  \\nOperations like cancelling orders are tightly regulated by the system's business rules and state transitions, making it a natural setting for this kind of task.",
    "state_space_definition": [
      {
        "entity": "User",
        "attributes": "user_id, username, email, account_status, registration_date",
        "description": "Represents a registered customer who can place orders, with associated identity and status."
      },
      {
        "entity": "Order",
        "attributes": "order_id, user_id, order_date, status, order_items",
        "description": "Represents an order placed by a user, including time, current status (pending, shipped, completed, cancelled), and the items in the order."
      },
      {
        "entity": "OrderItem",
        "attributes": "order_id, product_id, quantity",
        "description": "Represents individual product items and quantities included in an order."
      }
    ],
    "constraints_rules": [
      "Only the user who placed an order may request its cancellation.",
      "Only orders with status \\"pending\\" (not \\"shipped,\\" \\"completed,\\" or \\"cancelled\\") are eligible for cancellation.",
      "There must be a clear ordering of orders per user (e.g., via order_date) to identify the most recent order.",
      "Changing an order's status to \\"cancelled\\" must follow the status transition rules (e.g., not possible if already shipped or completed)."
    ],
    "environment_class_definition": "\\nfrom typing import Dict, List, TypedDict\\n\\nclass UserInfo(TypedDict):\\n    user_id: str\\n    username: str\\n    email: str\\n    account_status: str\\n    registration_date: str\\n\\nclass OrderItemInfo(TypedDict):\\n    order_id: str\\n    product_id: str\\n    quantity: int\\n\\nclass OrderInfo(TypedDict):\\n    order_id: str\\n    user_id: str\\n    order_date: str  # ISO date or timestamp as string\\n    status: str      # e.g., \\"pending\\", \\"shipped\\", \\"completed\\", \\"cancelled\\"\\n    order_items: List[OrderItemInfo]\\n\\nclass EcommerceOrderManagementSystem:\\n    def __init__(self, init_config: dict):\\n        \\\"\\\"\\\"\\n        Backend environment for an e-commerce platform.\\n        \\\"\\\"\\\"\\n        # Users: {user_id: UserInfo}\\n        self.users: Dict[str, UserInfo] = {}\\n\\n        # Orders: {order_id: OrderInfo}\\n        self.orders: Dict[str, OrderInfo] = {}\\n\\n        # Constraints/rules:\\n        # - Only the user who placed an order may request its cancellation.\\n        # - Only orders with status \\"pending\\" (not \\"shipped\\", \\"completed\\", \\"cancelled\\") are eligible for cancellation.\\n        # - There must be a clear ordering of orders per user (e.g., via order_date) to identify the most recent order.\\n        # - Changing an order's status to \\"cancelled\\" must follow status transition rules (not possible if already shipped/completed).\\n\\n        self.init_config = init_config\\n",
    "environment_example_task": "Cancel the most recent order placed by the user `alice123` if it has not yet been shipped.",
  }"""

output_case_1 = \
"""# Analysis
An example task in an environment is: find user alice123, confirm their recent order, and cancel it when the order status is pending.  
Required operations include: get user → get order set → sort to confirm most recent → check status → update to cancelled.  

However, considering that the environment is an "e-commerce order management system" involving multi-dimensional information about users, orders, and order items, the system needs to support more operations, such as:  
- For user layer: besides querying users by username, it can also list all users, query users by user_id, and check user account status.  
- For order layer: besides querying order lists, it may also need to filter by status, get product lists for orders, view order history status changes, and count orders.  
- For order item layer: besides listing order items, it may also need to update product quantities or remove specific entries.  
- For constraints: additional validation operations may be defined, such as verifying whether an order belongs to the user, verifying whether cancellation complies with rules.  
- Environment redundant operations: such as restoring cancelled orders, admin deleting orders, bulk cancelling multiple orders, etc. - these operations may not be needed for the current task, but can improve the richness of the training environment.  

Therefore, the operation set should not only cover the current task path, but also reflect the diverse operational aspects of the environment.

# Operation List
## Information Query Class
- Operation: get_user_by_username  
  Description: Retrieve user info (id, email, status) by username.  

- Operation: get_user_by_id  
  Description: Retrieve user info by unique user_id.  

- Operation: list_all_users  
  Description: Retrieve the list of all registered users in the system.  

- Operation: check_user_account_status  
  Description: Query the account status of a user (active, suspended, etc.).  

- Operation: list_user_orders  
  Description: Retrieve all orders placed by a specific user.  

- Operation: list_orders_by_status  
  Description: Retrieve all orders of a user filtered by status (pending, shipped, etc.).  

- Operation: get_most_recent_order  
  Description: Identify the most recent order for a user based on order_date.  

- Operation: get_order_by_id  
  Description: Retrieve full details of an order given its order_id.  

- Operation: get_order_status  
  Description: Check the current status of an order.  

- Operation: get_order_items  
  Description: List all line items (products and quantities) in an order.  

- Operation: get_order_history  
  Description: Show the chronological status change history of a given order.  

## State Change Class
- Operation: cancel_order  
  Description: Update the status of an eligible order to "cancelled".  

- Operation: update_order_status  
  Description: Change the status of an order to any valid value (pending, shipped, completed, cancelled).  

- Operation: bulk_cancel_orders  
  Description: Cancel multiple pending orders from the same user.  

- Operation: reopen_cancelled_order  
  Description: Revert a cancelled order back to pending, if allowed.  

- Operation: modify_order_items  
  Description: Update or remove specific items in an order.  

- Operation: delete_order  
  Description: Remove an order permanently (admin-level action).  

- Operation: restore_order  
  Description: Restore a previously deleted order if metadata is retained."""

# --- Input template for operation list inference ---
input_template = "Based on the following environment specification, produce the operation list.\n\n{env_info}"


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

def parse_operation_list(operation_list_str, operation_type):
    """Parse operation list string into list of dictionaries with operation name, description, type."""
    raw_operations = operation_list_str.split("- Operation: ")
    raw_operations = [operation for operation in raw_operations if operation.strip()]
    operations = []
    for operation in raw_operations:
        try:
            if "Description: " not in operation:
                continue
            parts = operation.split("Description: ")
            operation_name = parts[0].strip()
            operation_description = parts[1].strip()
            operations.append({
                "operation_name": operation_name,
                "operation_description": operation_description,
                "operation_type": operation_type
            })
        except Exception as e:
            print(f"Error parsing specific operation: {e}")
    return operations


def parse_response(response):
    """Parse LLM response to extract analysis and operation list."""
    if "# Analysis" in response and "# Operation List" in response and "## Information Query Class" in response and "## State Change Class" in response:
        try:
            analysis = response.split("# Analysis")[1].split("# Operation List")[0].strip()
            query_operation_str = response.split("## Information Query Class")[1].split("## State Change Class")[0].strip()
            state_change_operation_str = response.split("## State Change Class")[1].strip()
            
            query_operation_list = parse_operation_list(query_operation_str, "query")
            state_change_operation_list = parse_operation_list(state_change_operation_str, "state_change")
            
            operation_list = query_operation_list + state_change_operation_list
            return analysis, operation_list
        except Exception as e:
            print(f"Error parsing response structure: {e}")
            return "Error parsing response: " + str(e), []
    else:
        print(f"Response format mismatch: {response[:100]}...")
        return "Error parsing response: Format Mismatch", []


def process_env_item(env_item, model):
    """Process a single environment item to infer operation list."""
    # Skip if operation_list already exists and is valid
    if "operation_list" in env_item and env_item["operation_list"]:
        return env_item

    new_env_item = deepcopy(env_item)
    env_info = {
        "environment_summary": env_item.get("environment_summary", ""),
        "environment_introduction": env_item.get("environment_introduction", ""),
        "state_space_definition": env_item.get("state_space_definition", []),
        "constraints_rules": env_item.get("constraints_rules", []),
        "environment_class_definition": env_item.get("class_definition", ""),
        "environment_example_task": env_item.get("task", "")
    }
    
    input_content = input_template.format(env_info=json.dumps(env_info, indent=2, ensure_ascii=False))
    
    cur_try = 0
    max_try = 3
    final_analysis, final_op_list = "", []

    while cur_try < max_try:
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
            analysis, operation_list = parse_response(response)
            if operation_list:
                final_analysis = analysis
                final_op_list = operation_list
                break
        except Exception as e:
            print(f"LLM call error on item {env_item.get('environment_summary')} try {cur_try}: {e}")
        cur_try += 1

    new_env_item["operation_analysis"] = final_analysis
    new_env_item["operation_list"] = final_op_list
    return new_env_item


def main(read_file_path, save_file_path, model, max_workers=10):
    """Main function: generate operation lists for all environments using parallel threads."""
    raw_data = read_file(read_file_path)
    existing_results = _load_existing_results(save_file_path)
    result_map = {}
    for item in existing_results:
        key = _item_key(item)
        if key and key not in result_map:
            result_map[key] = item

    if result_map:
        print(f"Resuming from existing output: loaded {len(result_map)} completed items")

    remaining_data = [item for item in raw_data if _item_key(item) not in result_map]

    print(f"Inferring operation lists for {len(raw_data)} environments using {max_workers} workers...")
    print(f"Remaining items to generate: {len(remaining_data)}")

    if not remaining_data:
        save_file(save_file_path, _build_ordered_results(raw_data, result_map))
        print(f"Processing complete. Final results saved to: {save_file_path}")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_env_item, env_item, model): env_item for env_item in remaining_data}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Inferring Operations"):
            try:
                result = future.result()
                result_map[_item_key(result)] = result
            except Exception as e:
                orig_item = futures[future]
                print(f"Critical error on item '{orig_item.get('environment_summary')}': {e}")

            if len(result_map) % 10 == 0:
                save_file(save_file_path, _build_ordered_results(raw_data, result_map))

    save_file(save_file_path, _build_ordered_results(raw_data, result_map))
    print(f"Processing complete. Final results saved to: {save_file_path}")


if __name__ == "__main__":
    import os as _os
    _os.chdir(Path(__file__).resolve().parents[1])
    # --- Configuration ---
    MODEL = "gpt-5-mini" # Update to your required model
    READ_PATH = "stage2_syn_env/temp_result/step2_infer_state_code.json"
    SAVE_PATH = "stage2_syn_env/temp_result/step3_infer_operation.json"
    WORKERS = 10

    main(
        read_file_path=READ_PATH, 
        save_file_path=SAVE_PATH, 
        model=MODEL,
        max_workers=WORKERS
    )