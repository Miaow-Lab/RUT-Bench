import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.call_llm import llm_inference
from utils.process_file import read_file, save_file

CONFLICT_TAXONOMY = {
    "T1": "State drift — resource mutates between observation and action (stale cache).",
    "T2": "Validity expiration — time-limited artefact expires between acquisition and use.",
    "T3": "Schedule/window violation — operation attempted outside permitted time window.",
    "S1": "Resource locality mismatch — target bound to wrong location/branch/node.",
    "S2": "Jurisdictional barrier — policy/licensing/regulatory boundary blocks operation.",
    "S3": "Topology disruption — physical/logical path blocked or re-routed.",
    "ST1": "Dynamic spatial impact — temporal event reshapes spatial landscape.",
    "ST2": "Cascading dependency — temporal failure propagates spatially downstream.",
    "ST3": "Moving-window resource — available only within joint time+location window.",
}

SYSTEM_PROMPT = """You are an expert judge for spatiotemporal dependency and multi-API conflict analysis on stateful tasks.

Your job is to judge whether a task should be selected as a benchmark candidate for the Step 3 conflict injection pipeline.

Judge the task against the actual Step 3 conflict taxonomy below. A task should be kept only if a competent agent solving it could naturally encounter at least one realistic conflict mechanism from this taxonomy during a normal multi-step workflow:

- T1: State drift — resource mutates between observation and action.
- T2: Validity expiration — a time-limited artefact expires between acquisition and use.
- T3: Schedule/window violation — an operation becomes invalid outside a permitted time window.
- S1: Resource locality mismatch — the target is bound to the wrong location / branch / node.
- S2: Jurisdictional barrier — a policy, licensing, or regulatory boundary blocks the operation.
- S3: Topology disruption — a physical or logical path becomes blocked or re-routed.
- ST1: Dynamic spatial impact — a temporal event reshapes the spatial landscape.
- ST2: Cascading dependency — a temporal failure propagates spatially to downstream resources.
- ST3: Moving-window resource — a resource is only available within a joint time + location window.

Evaluate the provided task using these dimensions:
1. Whether the task has real temporal affinity for T1/T2/T3 conflicts.
2. Whether the task has real spatial affinity for S1/S2/S3 conflicts.
3. Whether the task has joint spatiotemporal or strict dependent workflow structure that could support ST1/ST2/ST3 conflicts.
4. Whether the conflict would lie on a normal, competent execution path rather than requiring an unnatural contrived setup.
5. Whether the task has enough multi-step dependency that an injected conflict would be meaningful, observable, and benchmark-worthy.

Judgment rule:
- Answer YES only if the task naturally supports at least one concrete Step 3 conflict code from the taxonomy above.
- Answer NO if the task is stateful but does not clearly support any of those concrete conflict mechanisms on a realistic execution path.

Be strict:
- Prefer NO for simple CRUD-like stateful tasks.
- Prefer NO when time/space language is superficial but does not create a real conflict opportunity.
- Prefer YES only when the conflict opportunity is central to correct execution rather than incidental.
- A task with general multi-step logic but no plausible Step 3 conflict code should still be NO.

Output format (must follow exactly):
# Analysis
<detailed reasoning grounded in the Step 3 conflict taxonomy, explaining which conflict codes are or are not plausible>

# Dependency Type
<Temporal / Spatial / Joint / Sequential / None>

# Conflict Codes
<comma-separated codes such as T1, ST2, S3, or None>

# Answer
YES or NO
"""

USER_PROMPT = """Analyze the following stateful task for spatiotemporal dependencies and multi-API call conflicts.

Task:
{query}

Decide whether this task should be kept for Step 3 conflict injection, based on whether it supports one or more concrete Step 3 conflict codes.
"""

def parse_response(response: str):
    """Parse LLM response to extract analysis and spatiotemporal judgment."""
    try:
        if "# Analysis" not in response or "# Answer" not in response:
            return "parsed_failed", "Unknown", [], False
        
        # Split logic based on the strict output format
        analysis = response.split("# Analysis")[1].split("# Dependency Type")[0].strip()
        dependency_block = response.split("# Dependency Type")[1]
        if "# Conflict Codes" in dependency_block:
            dep_type = dependency_block.split("# Conflict Codes")[0].strip()
            codes_block = dependency_block.split("# Conflict Codes")[1].split("# Answer")[0].strip()
        else:
            dep_type = dependency_block.split("# Answer")[0].strip()
            codes_block = "None"

        if codes_block.lower() == "none":
            conflict_codes = []
        else:
            conflict_codes = [code.strip() for code in codes_block.split(",") if code.strip() in CONFLICT_TAXONOMY]

        answer_part = response.split("# Answer")[1].strip().upper()
        
        answer = True if "YES" in answer_part else False
        return analysis, dep_type, conflict_codes, answer
    except Exception:
        return "parsed_failed", "Error", [], False

def process_query(item, model):
    """Process a single task to judge spatiotemporal complexity."""
    query = item["task"]
    context_query = f"Original Task: {query}"
    
    response = llm_inference(
        provider="openai",
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(query=context_query)}
        ],
    )
    
    analysis, dep_type, conflict_codes, answer = parse_response(response)
    
    # Update the item with new step2 results
    item.update({
        "st_analysis": analysis,
        "dependency_type": dep_type,
        "matched_conflict_codes": conflict_codes,
        "st_judge_result": answer
    })
    return item

def main(input_file, save_file_path, model, max_workers=20):
    """Filter results and perform step2 judgment."""
    # 1. Read step1 results
    source_data = read_file(input_file)
    
    # 2. STRICT FILTER: Only process items where step1 judge_result is True
    # Items with False are completely filtered out from this stage
    tasks_to_process = [item for item in source_data if item.get("judge_result") is True]
    
    print(f"Total tasks from Step 1: {len(source_data)}")
    print(f"Stateful tasks to evaluate (Filtered): {len(tasks_to_process)}")
    
    final_data = []

    # 3. Process the filtered tasks
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_query, item, model): item for item in tasks_to_process}
        
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing ST-Dependencies")):
            try:
                result = future.result()
                final_data.append(result)
            except Exception as e:
                print(f"Error processing item: {e}")

            # Intermediate save for safety
            if len(final_data) % 50 == 0:
                save_file(save_file_path, final_data)

    # 4. Final save
    save_file(save_file_path, final_data)
    print(f"Done. Step 2 results (only ST-relevant) saved to: {save_file_path}")

if __name__ == "__main__":
    import os as _os
    _os.chdir(Path(__file__).resolve().parents[1])
    # Path Configuration
    step1_output = "stage1_collect_env_from_task/temp_result/step1_stateful_task_judge.json"
    step2_output = "stage1_collect_env_from_task/temp_result/step2_spatiotemporal_task_judge.json"
    model_name = "gpt-4o-mini" 

    main(
        input_file=step1_output,
        save_file_path=step2_output,
        model=model_name,
        max_workers=50
    )