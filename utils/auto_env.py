"""
Automatically convert environment source code to gym/gem-like interactive environment.
"""
import types
import traceback
import json
from copy import deepcopy


def _to_json_safe(obj):
    """Round-trip an object through JSON to ensure all values are plain JSON types.
    This prevents datetime objects, custom classes, etc. from leaking into state dicts.
    Dates/times are preserved as the strings they serialize to (via default=str).
    """
    return json.loads(json.dumps(obj, ensure_ascii=False, default=str))


def get_state_diff(old_state: dict, new_state: dict) -> dict:
    """Compare two state dictionaries and return difference details."""
    diff_result = {}

    # Find union of all keys
    all_keys = set(old_state.keys()) | set(new_state.keys())

    for key in all_keys:
        old_val = old_state.get(key)
        new_val = new_state.get(key)

        if key not in old_state:
            # Added key
            diff_result[key] = {"added": new_val}
        elif key not in new_state:
            # Removed key
            diff_result[key] = {"removed": old_val}
        else:
            # Both exist → compare values
            if isinstance(old_val, dict) and isinstance(new_val, dict):
                # Recursively compare dictionaries
                sub_diff = get_state_diff(old_val, new_val)
                if sub_diff:  # Only record if there are changes
                    diff_result[key] = sub_diff
            else:
                # Simple type comparison
                if old_val != new_val:
                    diff_result[key] = {"changed": {"old":old_val, "new":new_val}}

    return diff_result



class InteractiveEnv:
    """Interactive environment wrapper with gym-like API."""

    def __init__(self, env_class, max_steps):
        self.env_class = env_class
        self.env_instance = None
        self.current_step = 0
        self.max_steps = max_steps
        self.trajectory = []  # Record trajectory for each step
        self.dynamic_event_rules = []
        self.dynamic_event_runtime = {}
        self.dynamic_event_log = []
        self.tool_call_history = []

    def env_init(self, init_config=None):
        """Initialize environment instance."""
        self.current_step = 0
        try:
            self.env_instance = self.env_class({})
        except Exception as e:
            self.env_instance = self.env_class()

        if init_config:
            self._apply_init_config(self.env_instance, init_config)
        self._load_dynamic_event_rules()
        # Reset trajectory (step 0 represents initial state)
        self.trajectory = [
                {
                    "step": 0, 
                    "state_snapshot": deepcopy(self.get_state_info())
                }
            ]

    def _load_dynamic_event_rules(self):
        """Load deterministic dynamic-event rules from the environment instance."""
        raw_rules = getattr(self.env_instance, "_dynamic_event_rules", []) or []
        safe_rules = _to_json_safe(raw_rules)
        if not isinstance(safe_rules, list):
            safe_rules = []

        self.dynamic_event_rules = []
        self.dynamic_event_runtime = {}
        self.dynamic_event_log = []
        self.tool_call_history = []

        for index, rule in enumerate(safe_rules, 1):
            if not isinstance(rule, dict):
                continue
            normalized = deepcopy(rule)
            event_id = str(normalized.get("event_id") or f"dynamic_event_{index}")
            normalized["event_id"] = event_id
            activation_tool = str(normalized.get("activation_tool", "")).strip()
            initially_armed = not activation_tool
            self.dynamic_event_rules.append(normalized)
            self.dynamic_event_runtime[event_id] = {
                "armed": initially_armed,
                "fired": False,
                "arm_history_len": 0 if initially_armed else None,
                "fire_step": None,
            }

    def _matches_arg_subset(self, actual_args, expected_args):
        if not expected_args:
            return True
        if not isinstance(actual_args, dict) or not isinstance(expected_args, dict):
            return False
        for key, value in expected_args.items():
            if actual_args.get(key) != value:
                return False
        return True

    def _matches_tool_rule(self, action, tool_key, args_key, rule):
        expected_tool = rule.get(tool_key)
        if isinstance(expected_tool, list):
            allowed_tools = {str(item).strip() for item in expected_tool if str(item).strip()}
            if allowed_tools and action.get("name") not in allowed_tools:
                return False
        else:
            expected_tool = str(expected_tool or "").strip()
            if expected_tool and action.get("name") != expected_tool:
                return False

        return self._matches_arg_subset(action.get("params", {}), rule.get(args_key, {}))

    def _safe_int(self, value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _resolve_state_path(self, path, create_missing=False):
        path = str(path or "").strip()
        if not path:
            return None, None
        if path.startswith("self."):
            path = path[5:]

        parts = [part for part in path.split(".") if part]
        if not parts:
            return None, None

        current = self.env_instance
        for part in parts[:-1]:
            if isinstance(current, dict):
                if part not in current:
                    if not create_missing:
                        return None, None
                    current[part] = {}
                current = current[part]
                continue

            if not hasattr(current, part):
                if not create_missing:
                    return None, None
                setattr(current, part, {})
            current = getattr(current, part)

        return current, parts[-1]

    def _apply_single_state_update(self, update):
        if not isinstance(update, dict):
            return False

        op = str(update.get("op", "set")).strip() or "set"
        value = update.get("value")
        target, leaf = self._resolve_state_path(update.get("path", ""), create_missing=(op == "set"))
        if target is None or not leaf:
            return False

        if isinstance(target, dict):
            current_value = target.get(leaf)
            if op == "set":
                target[leaf] = value
                return True
            if op == "delete":
                target.pop(leaf, None)
                return True
            if op == "append":
                if leaf not in target or not isinstance(target[leaf], list):
                    target[leaf] = []
                target[leaf].append(value)
                return True
            if op == "increment":
                target[leaf] = (current_value or 0) + (value or 0)
                return True
            return False

        if op == "set":
            setattr(target, leaf, value)
            return True
        if op == "delete":
            if hasattr(target, leaf):
                delattr(target, leaf)
            return True
        if op == "append":
            current_value = getattr(target, leaf, None)
            if not isinstance(current_value, list):
                current_value = []
            current_value.append(value)
            setattr(target, leaf, current_value)
            return True
        if op == "increment":
            current_value = getattr(target, leaf, 0) or 0
            setattr(target, leaf, current_value + (value or 0))
            return True
        return False

    def _apply_dynamic_state_updates(self, updates):
        applied = []
        for update in updates or []:
            if self._apply_single_state_update(update):
                applied.append(update)
        return applied

    def _fire_dynamic_event(self, rule, action):
        event_id = rule.get("event_id", "dynamic_event")
        runtime = self.dynamic_event_runtime.setdefault(
            event_id,
            {"armed": True, "fired": False, "arm_history_len": 0, "fire_step": None},
        )
        if runtime.get("fired"):
            return None

        applied_updates = self._apply_dynamic_state_updates(rule.get("state_updates", []))
        runtime["fired"] = True
        runtime["fire_step"] = self.current_step

        log_entry = {
            "event_id": event_id,
            "step": self.current_step,
            "action": deepcopy(action),
            "effect_type": rule.get("effect_type", "error_with_state_update"),
            "error_message": rule.get("error_message", ""),
            "applied_updates": deepcopy(applied_updates),
            "observability": deepcopy(rule.get("observability", [])),
        }
        self.dynamic_event_log.append(log_entry)

        if rule.get("effect_type", "error_with_state_update") == "error_with_state_update":
            return {"error": rule.get("error_message") or "Dynamic environment change invalidated the current plan."}
        return None

    def _maybe_fire_dynamic_event_before_action(self, action):
        history_len = len(self.tool_call_history)
        for rule in self.dynamic_event_rules:
            event_id = rule.get("event_id", "dynamic_event")
            runtime = self.dynamic_event_runtime.get(event_id, {})
            if runtime.get("fired") or not runtime.get("armed"):
                continue

            arm_history_len = runtime.get("arm_history_len")
            if arm_history_len is None:
                continue

            required_delay = max(0, self._safe_int(rule.get("fire_after_tool_calls", 0), 0))
            if history_len - arm_history_len < required_delay:
                continue

            if not self._matches_tool_rule(action, "fire_on_tool", "fire_on_args", rule):
                continue

            injected_observation = self._fire_dynamic_event(rule, action)
            if injected_observation is not None:
                return injected_observation
        return None

    def _maybe_arm_dynamic_event_after_action(self, action):
        history_len = len(self.tool_call_history)
        for rule in self.dynamic_event_rules:
            event_id = rule.get("event_id", "dynamic_event")
            runtime = self.dynamic_event_runtime.get(event_id, {})
            if runtime.get("fired") or runtime.get("armed"):
                continue
            if not self._matches_tool_rule(action, "activation_tool", "activation_args", rule):
                continue
            runtime["armed"] = True
            runtime["arm_history_len"] = history_len

    def get_dynamic_event_log(self):
        return deepcopy(self.dynamic_event_log)

    def get_dynamic_event_summary(self):
        triggered_ids = [entry.get("event_id", "") for entry in self.dynamic_event_log]
        return {
            "defined_event_count": len(self.dynamic_event_rules),
            "triggered_event_count": len(triggered_ids),
            "triggered_event_ids": triggered_ids,
        }

    def env_step(self, action: dict):
        """Execute one step of interaction. Returns: observation, reward, terminated, truncated, info."""
        self.current_step += 1
        terminated = False
        truncated = False
        info = {}

        # Exceeded maximum steps
        if self.current_step > self.max_steps:
            self._record_step(action, {}, 0, terminated, truncated, info)
            return {"error": "Max steps reached"}, 0, True, True, info

        try:
            method = getattr(self.env_instance, action["name"])
        except AttributeError:
            # Method not found — this IS a fatal error
            error_log = traceback.format_exc()
            observation = {"error": "<Exception>\n" + error_log}
            reward = 0.0
            terminated = True
            truncated = True
            info = {}
            self._record_step(action, observation, reward, terminated, truncated, info)
            return observation, reward, terminated, truncated, info

        try:
            # Check parameters
            if "params" not in action:
                return {"error": "No params in action"}, 0, terminated, truncated, info

            injected_observation = self._maybe_fire_dynamic_event_before_action(action)
            if injected_observation is not None:
                observation = injected_observation
                reward = 0.0
                self.tool_call_history.append(deepcopy(action))
                self._record_step(action, observation, reward, terminated, truncated, info)
                return observation, reward, terminated, truncated, info

            observation = method(**action.get("params", {}))
            self.tool_call_history.append(deepcopy(action))
            self._maybe_arm_dynamic_event_after_action(action)

            # Termination condition: reached max steps
            if self.current_step == self.max_steps:
                truncated = True
                terminated = True

            # Calculate reward when task ends
            if terminated:
                reward = self.calculate_reward()
            else:
                reward = 0.0

            self._record_step(action, observation, reward, terminated, truncated, info)
            return observation, reward, terminated, truncated, info

        except Exception:
            # Business-logic exceptions (ValueError, KeyError, AttributeError, etc.)
            # These are NORMAL tool-call errors — NOT terminal.
            error_log = traceback.format_exc()
            observation = {"error": "<Exception>\n" + error_log}
            reward = 0.0
            info = {}
            self._record_step(action, observation, reward, terminated, truncated, info)
            return observation, reward, terminated, truncated, info

    def _record_step(self, action, observation, reward, terminated, truncated, info):
        """Record current step information to trajectory."""
        last_state = self.trajectory[-1]["state_snapshot"]
        current_state = deepcopy(self.get_state_info())
        state_diff = get_state_diff(last_state, current_state)
        step_record = {
            "step": self.current_step,
            "action": action,
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "state_snapshot": current_state,
            "state_diff": state_diff,
        }
        self.trajectory.append(step_record)

    def _apply_init_config(self, env, init_config):
        """Apply initial configuration to environment instance.
        Values are normalized to plain JSON types so that dates/times
        remain as strings exactly as they appeared in the original config.
        """
        safe_config = _to_json_safe(init_config)
        for key, value in safe_config.items():
            setattr(env, key, value)

    def get_state_info(self):
        """Return instance state as plain JSON-safe types (no datetime objects, etc.)."""
        obj = self.env_instance
        raw = {
            k: v
            for k, v in vars(obj).items()
            if not (k.startswith("__") and k.endswith("__"))
        }
        return _to_json_safe(raw)

    def calculate_reward(self):
        """Placeholder reward function: currently returns 0."""
        return 0


def build_env_from_str(env_str: str, class_name: str, max_steps: int) -> InteractiveEnv:
    """Build interactive environment from source code string."""
    module = types.ModuleType("dynamic_env")
    exec(env_str, module.__dict__)
    env_class = getattr(module, class_name)
    return InteractiveEnv(env_class, max_steps)