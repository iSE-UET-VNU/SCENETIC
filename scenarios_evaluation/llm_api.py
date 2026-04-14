import argparse
import json
import os
import re
import time
import traceback
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
try:
    import tiktoken
except ImportError:
    tiktoken = None
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from rubrics import RUBRICS, OUTPUT_FORMAT


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[0]

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(CURRENT_DIR / ".env", override=False)


class LLMAPI(object):

    def __init__(self):
        self.frame_interval_seconds = 1.0
        self.max_scenarios_per_folder = 1
        self.project_root = PROJECT_ROOT
        self.outputs_root = Path(__file__).resolve().parent / "outputs_results"
        self.max_workers = max(1, int(os.getenv("LLM_MAX_WORKERS", "4")))
        self.prompt_token_threshold = int(os.getenv("PROMPT_TOKEN_THRESHOLD", "35000"))
        self.request_timeout_seconds = float(os.getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "300"))
        self.request_max_retries = max(1, int(os.getenv("OPENAI_MAX_RETRIES", "3")))
        self.retry_backoff_seconds = float(os.getenv("OPENAI_RETRY_BACKOFF_SECONDS", "2"))
        self._encoding_cache = {}
        self.road_description = {
            ("sanfrancisco", "road1"): (
                "In the following scenario, Ego starts from the lower part of a north-south road in San Francisco and drives north through a signalized intersection. "
                "It then continues upward and follows a left-curving connector onto a slanted side road, forming an L-shaped residential junction. "
                "The surrounding environment includes buildings, sidewalks, and trees.\n"
            ),
            ("sanfrancisco", "road2"): (
                "In the following scenario, Ego starts from the right-side branch road of a signalized intersection in San Francisco and drives west into the junction. "
                "It turns left onto the long north-south arterial road, continues north, and then turns left again onto the upper-left slanted road. "
                "The surrounding environment includes buildings, sidewalks, and trees.\n"
            ),
            ("tartu", "road1"): (
                "In the following scenario, Ego starts from the lower part of a long north-south arterial road in Tartu and drives straight toward a large multi-lane signalized intersection. "
                "After reaching the junction, it turns onto a connecting road and merges into a lane that is perpendicular to its original direction of travel. "
                "The surrounding environment includes buildings, sidewalks, parking areas, and some trees.\n"
            ),
            ("tartu", "road2"): (
                "In the following scenario, Ego starts from the lower part of a short north-south side road in Tartu and drives north into a signalized junction. "
                "It then turns left onto a long curved boulevard and continues west along the road. "
                "The surrounding environment includes buildings, sidewalks, trees, and a broad separator area.\n"
            ),
            (None, "road3"): "In the following scenario, Ego's driving intention is to first perform a left turn to switch to a straight downhill lane and then drive without turning.\n",
            (None, "road4"): "In the following scenario, Ego's driving intention is to first turn left and then drive on the right-hand side of the road.\n",
        }
        self.rubrics = RUBRICS
        self.output_format = OUTPUT_FORMAT
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "XXX")

    def get_road_description(self, city, road):
        return self.road_description.get((city, road), self.road_description.get((None, road), ""))

    def create_openai_client(self):
        client_kwargs = {"api_key": self.openai_api_key}
        if self.openai_base_url:
            client_kwargs["base_url"] = self.openai_base_url
        return OpenAI(**client_kwargs)

    def get_token_encoding(self, model):
        model_name = model.split("/")[-1]
        if model_name in self._encoding_cache:
            return self._encoding_cache[model_name]

        if tiktoken is None:
            return None

        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            try:
                encoding = tiktoken.get_encoding("o200k_base")
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")

        self._encoding_cache[model_name] = encoding
        return encoding

    def count_text_tokens(self, text, model):
        encoding = self.get_token_encoding(model)
        if encoding is None:
            return ceil(len(text) / 4)
        return len(encoding.encode(text or ""))

    def estimate_text_tokens(self, text, model=None):
        if model is not None:
            return self.count_text_tokens(text, model)
        return ceil(len(text) / 4)

    def estimate_message_tokens(self, messages, model):
        encoding = self.get_token_encoding(model)
        if encoding is None:
            return sum(self.estimate_text_tokens(message.get("content", "")) for message in messages)

        model_name = model.split("/")[-1]
        tokens_per_message = 3
        tokens_per_name = 1

        if model_name.startswith("gpt-3.5-turbo"):
            tokens_per_message = 4
            tokens_per_name = -1

        total_tokens = 0
        for message in messages:
            total_tokens += tokens_per_message
            for key, value in message.items():
                total_tokens += len(encoding.encode(value or ""))
                if key == "name":
                    total_tokens += tokens_per_name

        total_tokens += 3
        return total_tokens

    def extract_usage(self, completion, model, messages, output):
        usage = getattr(completion, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)

            if prompt_tokens is not None and completion_tokens is not None:
                if total_tokens is None:
                    total_tokens = prompt_tokens + completion_tokens
                return {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "source": "api",
                }

        prompt_tokens = self.estimate_message_tokens(messages, model)
        completion_tokens = self.estimate_text_tokens(output or "", model)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "source": "estimated",
        }

    def get_results_path(self, scenario_folder, scenario_path):
        return self.outputs_root / scenario_folder.name / scenario_path.stem

    def get_result_file_stem(self, model, index):
        return f"{model.split('/')[-1]}_{index}"

    def openai_chat_completion(self, model, messages, temperature):
        client = self.create_openai_client()
        last_error = None

        for attempt in range(1, self.request_max_retries + 1):
            start_time = time.time()
            try:
                request_kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if self.request_timeout_seconds > 0:
                    request_kwargs["timeout"] = self.request_timeout_seconds

                completion = client.chat.completions.create(**request_kwargs)
                create_time = time.time() - start_time
                middle_time = time.time()
                output = completion.choices[0].message.content
                output_time = time.time() - middle_time
                total_time = time.time() - start_time
                usage = self.extract_usage(completion, model, messages, output)
                return output, create_time, output_time, total_time, usage, attempt
            except Exception as exc:
                last_error = exc
                if attempt >= self.request_max_retries:
                    break
                if self.retry_backoff_seconds > 0:
                    time.sleep(self.retry_backoff_seconds * attempt)

        raise RuntimeError(
            f"OpenAI request failed after {self.request_max_retries} attempt(s): {last_error}"
        ) from last_error

    def write_result_file(self, results_path, model, index, messages, output, create_time, output_time, total_time, usage, has_collision):
        results_path.mkdir(parents=True, exist_ok=True)
        result_file_stem = self.get_result_file_stem(model, index)
        result_file_path = results_path / f"{result_file_stem}.txt"
        error_file_path = results_path / f"{result_file_stem}.error.txt"

        if error_file_path.exists():
            error_file_path.unlink()

        with result_file_path.open("w", encoding="utf-8") as file:
            file.write(f"model: {model}\n\n")
            file.write(f"has_collision: {has_collision}\n\n")
            file.write(messages[0]["content"] + "\n\n" + messages[1]["content"] + "\n\n\n")
            file.write(output + "\n\n\n")
            file.write(
                f"prompt_tokens: {usage['prompt_tokens']} completion_tokens: {usage['completion_tokens']} "
                f"total_tokens: {usage['total_tokens']} usage_source: {usage['source']}\n"
            )
            file.write(
                f"create_time: {create_time}s output_time: {output_time}s total_time: {total_time}s\n"
            )

    def write_error_file(self, results_path, model, index, messages, error_message, traceback_text, has_collision):
        results_path.mkdir(parents=True, exist_ok=True)
        error_file_path = results_path / f"{self.get_result_file_stem(model, index)}.error.txt"

        with error_file_path.open("w", encoding="utf-8") as file:
            file.write(f"model: {model}\n")
            file.write(f"index: {index}\n\n")
            file.write(f"has_collision: {has_collision}\n\n")
            file.write("system_message:\n")
            file.write(messages[0]["content"] + "\n\n")
            file.write("user_message:\n")
            file.write(messages[1]["content"] + "\n\n")
            file.write("error:\n")
            file.write(error_message + "\n\n")
            file.write("traceback:\n")
            file.write(traceback_text)

    def evaluate_single_run(self, scenario_folder, scenario_path, messages, model, index, temperature, has_collision):
        results_path = self.get_results_path(scenario_folder, scenario_path)

        try:
            output, create_time, output_time, total_time, usage, attempts = self.openai_chat_completion(model, messages, temperature)
            self.write_result_file(
                results_path=results_path,
                model=model,
                index=index,
                messages=messages,
                output=output,
                create_time=create_time,
                output_time=output_time,
                total_time=total_time,
                usage=usage,
                has_collision=has_collision,
            )
            return {
                "scenario_folder_name": scenario_folder.name,
                "scenario_stem": scenario_path.stem,
                "index": index,
                "usage": usage,
                "success": True,
                "attempts": attempts,
                "error": None,
            }
        except Exception as exc:
            traceback_text = traceback.format_exc()
            self.write_error_file(
                results_path=results_path,
                model=model,
                index=index,
                messages=messages,
                error_message=str(exc),
                traceback_text=traceback_text,
                has_collision=has_collision,
            )
            return {
                "scenario_folder_name": scenario_folder.name,
                "scenario_stem": scenario_path.stem,
                "index": index,
                "usage": None,
                "success": False,
                "attempts": self.request_max_retries,
                "error": str(exc),
            }

    def parse_scenario_folder_name(self, folder_name):
        if not folder_name.endswith("-scenarios"):
            raise ValueError(f"Invalid scenario folder name: {folder_name}")

        base_name = folder_name[:-len("-scenarios")]
        parts = base_name.split("-")
        if len(parts) < 2:
            raise ValueError(f"Invalid scenario folder name: {folder_name}")

        city = parts[0]
        road = parts[-1]
        method = "-".join(parts[1:-1]) or None
        return {
            "city": city,
            "method": method,
            "road": road,
        }

    def get_scenario_folders(self):
        return sorted(
            [path for path in self.project_root.iterdir() if path.is_dir() and path.name.endswith("-scenarios")],
            key=lambda path: path.name,
        )

    def resolve_scenario_folder(self, scenario_folder):
        if not scenario_folder:
            raise ValueError("A scenario folder must be provided.")

        requested_path = Path(scenario_folder).expanduser()
        candidate_paths = []

        if requested_path.is_absolute():
            candidate_paths.append(requested_path)
        else:
            candidate_paths.append(self.project_root / requested_path)
            candidate_paths.append(Path.cwd() / requested_path)

        matched_folder = next(
            (folder for folder in self.get_scenario_folders() if folder.name == scenario_folder),
            None,
        )
        if matched_folder is not None:
            candidate_paths.append(matched_folder)

        for candidate_path in candidate_paths:
            if candidate_path.exists() and candidate_path.is_dir():
                resolved_path = candidate_path.resolve()
                if not resolved_path.name.endswith("-scenarios"):
                    raise ValueError(
                        f"Scenario folder must end with '-scenarios': {resolved_path}"
                    )
                return resolved_path

        raise FileNotFoundError(f"Scenario folder not found: {scenario_folder}")

    def resolve_output_path(self, output_path):
        if not output_path:
            return self.outputs_root

        requested_path = Path(output_path).expanduser()
        if requested_path.is_absolute():
            return requested_path.resolve()

        project_relative_path = (self.project_root / requested_path)
        if project_relative_path.exists():
            return project_relative_path.resolve()

        return project_relative_path

    def scenario_sort_key(self, scenario_path):
        match = re.search(r"scenario(\d+)\.json$", scenario_path.name)
        if match:
            return int(match.group(1)), scenario_path.name
        return 10 ** 9, scenario_path.name

    def get_sorted_frame_names(self, scenario_data):
        return sorted(
            [frame_name for frame_name in scenario_data.keys() if re.fullmatch(r"timestep_\d+", frame_name)],
            key=lambda frame_name: int(frame_name.split("_")[1]),
        )

    def get_prompt_frame_names(self, scenario_data):
        frame_names = self.get_sorted_frame_names(scenario_data)
        if not frame_names:
            return []

        if len(frame_names) <= 60:
            return frame_names

        return frame_names[:-60]

    def scenario_has_collision(self, scenario_data):
        if not isinstance(scenario_data, dict):
            return False

        for frame_data in scenario_data.values():
            if not isinstance(frame_data, dict):
                continue
            for actor_data in frame_data.values():
                if isinstance(actor_data, dict) and actor_data.get("Collided_With_Ego") is True:
                    return True

        return False

    def format_vector(self, vector_data):
        if not isinstance(vector_data, dict):
            return "NA"

        def format_number(value):
            if isinstance(value, (int, float)):
                rounded_value = round(value, 2)
                formatted = f"{rounded_value:.2f}".rstrip("0").rstrip(".")
                return formatted
            return value

        x = format_number(vector_data.get("x", "NA"))
        y = format_number(vector_data.get("y", "NA"))
        z = format_number(vector_data.get("z", "NA"))
        return f"({x},{y},{z})"

    def format_actor(self, actor_name, actor_data):
        params = []
        short_names = {
            "position": "p",
            "rotation": "r",
            "velocity": "v",
            "angular_velocity": "av",
        }

        if actor_name.startswith("NPC"):
            actor_type = actor_data.get("type")
            if actor_type is not None:
                params.append(f"type={actor_type}")

            collided_with_ego = actor_data.get("Collided_With_Ego")
            if isinstance(collided_with_ego, bool):
                params.append(f"Collided_With_Ego={str(collided_with_ego).lower()}")

        for parameter_name in ["position", "rotation", "velocity", "angular_velocity"]:
            vector_data = actor_data.get(parameter_name)
            if not isinstance(vector_data, dict):
                continue
            params.append(f"{short_names[parameter_name]}={self.format_vector(vector_data)}")

        if not params:
            return ""

        return f"{actor_name} " + " ".join(params)

    def format_frame(self, frame_name, frame_data):
        frame_index = int(frame_name.split("_")[1])
        timestamp_seconds = frame_index * self.frame_interval_seconds
        lines = [f"t={timestamp_seconds:.1f}"]

        ego_data = frame_data.get("Ego")
        if isinstance(ego_data, dict):
            formatted_ego = self.format_actor("Ego", ego_data)
            if formatted_ego:
                lines.append(formatted_ego)

        npc_names = sorted(name for name in frame_data.keys() if name.startswith("NPC"))
        if npc_names:
            for npc_name in npc_names:
                npc_data = frame_data.get(npc_name)
                if isinstance(npc_data, dict):
                    formatted_npc = self.format_actor(npc_name, npc_data)
                    if formatted_npc:
                        lines.append(formatted_npc)

        return "\n".join(lines)

    def merge_actor_frames(self, actor_name, first_actor_data, second_actor_data):
        if not isinstance(first_actor_data, dict) and not isinstance(second_actor_data, dict):
            return None

        if isinstance(second_actor_data, dict):
            merged_actor = deepcopy(second_actor_data)
        else:
            merged_actor = deepcopy(first_actor_data)

        if actor_name.startswith("NPC"):
            first_collision = bool(first_actor_data.get("Collided_With_Ego")) if isinstance(first_actor_data, dict) else False
            second_collision = bool(second_actor_data.get("Collided_With_Ego")) if isinstance(second_actor_data, dict) else False
            merged_actor["Collided_With_Ego"] = first_collision or second_collision

            if merged_actor.get("type") is None and isinstance(first_actor_data, dict):
                merged_actor["type"] = first_actor_data.get("type")

        return merged_actor

    def merge_frame_pair(self, first_frame_data, second_frame_data):
        if not isinstance(first_frame_data, dict) and not isinstance(second_frame_data, dict):
            return {}

        later_frame_data = second_frame_data if isinstance(second_frame_data, dict) else first_frame_data
        merged_frame = {}

        for actor_name in sorted(later_frame_data.keys() if isinstance(later_frame_data, dict) else []):
            if actor_name.startswith("NPC") or actor_name == "Ego":
                merged_actor = self.merge_actor_frames(
                    actor_name,
                    first_frame_data.get(actor_name) if isinstance(first_frame_data, dict) else None,
                    second_frame_data.get(actor_name) if isinstance(second_frame_data, dict) else None,
                )
                if merged_actor is not None:
                    merged_frame[actor_name] = merged_actor

        if isinstance(first_frame_data, dict):
            for actor_name in sorted(first_frame_data.keys()):
                if actor_name in merged_frame or (not actor_name.startswith("NPC") and actor_name != "Ego"):
                    continue
                merged_actor = self.merge_actor_frames(actor_name, first_frame_data.get(actor_name), None)
                if merged_actor is not None:
                    merged_frame[actor_name] = merged_actor

        return merged_frame

    def get_merged_prompt_frames(self, scenario_data):
        frame_names = self.get_prompt_frame_names(scenario_data)
        merged_frames = []

        for merged_index, start_index in enumerate(range(0, len(frame_names), 2)):
            first_frame_name = frame_names[start_index]
            second_frame_name = frame_names[start_index + 1] if start_index + 1 < len(frame_names) else None
            first_frame_data = scenario_data.get(first_frame_name)
            second_frame_data = scenario_data.get(second_frame_name) if second_frame_name is not None else None

            merged_frame_name = f"timestep_{merged_index}"
            merged_frame_data = self.merge_frame_pair(first_frame_data, second_frame_data)
            merged_frames.append((merged_frame_name, merged_frame_data))

        return merged_frames

    def format_scenario_frames(self, scenario_data):
        return "\n\n".join(
            self.format_frame(frame_name, frame_data)
            for frame_name, frame_data in self.get_merged_prompt_frames(scenario_data)
            if isinstance(frame_data, dict)
        )

    def build_messages(self, scenario_folder, scenario_path, scenario_data):
        metadata = self.parse_scenario_folder_name(scenario_folder.name)
        merged_frames = self.get_merged_prompt_frames(scenario_data)
        road_description = self.get_road_description(metadata["city"], metadata["road"])
        timestamp_lines = [
            f'"{int(frame_name.split("_")[1]) * self.frame_interval_seconds:.1f} seconds": <realism score>'
            for frame_name, _ in merged_frames
        ]
        timestamp_json_example = ",\n".join(timestamp_lines[:5])
        frames_block = self.format_scenario_frames(scenario_data)

        messages = []
        sys_message = {
            "role": "system",
            "content": """You know a lot about autonomous driving and some realistic autonomous driving crash datasets.
You are a helpful scenario realism evaluation assistant.
You will evaluate the following autonomous driving scenario, and check whether the scenario is realistic.
You will provide the corresponding realism score. The scale is 1.0-10.0, where 1.0 is unrealistic and 10.0 is realistic.
You will also provide the Probability and Confidence of the outputs as a percentage. Probability refers to the likelihood of your output, while Confidence indicates the certainty in the prediction for your output.
You know the scenario is from the LGSVL simulator.""",
        }
        messages.append(sys_message)

        prompt = {
            "role": "user",
            "content": f"""{road_description}The scenario id is {scenario_path.stem}. The city is {metadata['city']}. The road is {metadata['road']}.
The scenario starts at 0.0 seconds.
Frame notation: t=time in seconds, p=position, r=rotation, v=velocity, av=angular velocity, type=NPC actor type, Collided_With_Ego=whether this NPC/Pedestrian collides with Ego Vehicle.

{frames_block}

Important note about collisions:
- A collision can still be realistic and should not be penalized by default.
- If a collision occurs, judge whether it develops naturally from the prior motion, timing, and interaction context.
- Penalize it only when the collision feels forced, implausible, or weakly supported by earlier frames.

# Evaluating Rubrics:
{self.rubrics}

# Output Format:
{self.output_format}
"""
        }
        messages.append(prompt)
        return messages

    def evaluate_R_MR_extra_4full_experiments(self, model, repeat_time, scenario_folder):
        scenario_folders = [self.resolve_scenario_folder(scenario_folder)]
        jobs = []
        prompt_token_stats = []

        for scenario_folder in scenario_folders:
            print(scenario_folder)
            scenario_paths = sorted(
                scenario_folder.glob("scenario*.json"),
                key=self.scenario_sort_key,
            )[:self.max_scenarios_per_folder]
            for scenario_path in scenario_paths:
                with scenario_path.open("r", encoding="utf-8") as json_file:
                    scenario_data = json.load(json_file)

                has_collision = self.scenario_has_collision(scenario_data)
                messages = self.build_messages(scenario_folder, scenario_path, scenario_data)
                system_tokens = self.estimate_text_tokens(messages[0]["content"], model)
                user_tokens = self.estimate_text_tokens(messages[1]["content"], model)
                total_tokens = self.estimate_message_tokens(messages, model)
                prompt_token_stats.append(
                    {
                        "scenario_folder": scenario_folder.name,
                        "scenario_name": scenario_path.stem,
                        "system_tokens": system_tokens,
                        "user_tokens": user_tokens,
                        "total_tokens": total_tokens,
                    }
                )

                for index in range(repeat_time):
                    jobs.append((scenario_folder, scenario_path, messages, model, index, 0, has_collision))

            # break

        if prompt_token_stats:
            total_values = [item["total_tokens"] for item in prompt_token_stats]
            system_values = [item["system_tokens"] for item in prompt_token_stats]
            user_values = [item["user_tokens"] for item in prompt_token_stats]
            scenarios_above_threshold = [
                item
                for item in prompt_token_stats
                if item["total_tokens"] > self.prompt_token_threshold
            ]

            min_total_item = min(prompt_token_stats, key=lambda item: item["total_tokens"])
            max_total_item = max(prompt_token_stats, key=lambda item: item["total_tokens"])

            print(f"scenarios_read: {len(prompt_token_stats)}")
            print(
                f"system_tokens avg/min/max: "
                f"{sum(system_values) / len(system_values):.2f}/"
                f"{min(system_values)}/{max(system_values)}"
            )
            print(
                f"user_tokens avg/min/max: "
                f"{sum(user_values) / len(user_values):.2f}/"
                f"{min(user_values)}/{max(user_values)}"
            )
            print(
                f"total_prompt_tokens avg/min/max: "
                f"{sum(total_values) / len(total_values):.2f}/"
                f"{min(total_values)}/{max(total_values)}"
            )
            print(
                f"min_total_scenario: {min_total_item['scenario_folder']} / "
                f"{min_total_item['scenario_name']} ({min_total_item['total_tokens']} tokens)"
            )
            print(
                f"max_total_scenario: {max_total_item['scenario_folder']} / "
                f"{max_total_item['scenario_name']} ({max_total_item['total_tokens']} tokens)"
            )
            print(
                f"scenarios_above_{self.prompt_token_threshold}_tokens: "
                f"{len(scenarios_above_threshold)}"
            )
            for item in scenarios_above_threshold[:10]:
                print(
                    f"above_threshold: {item['scenario_folder']} / "
                    f"{item['scenario_name']} ({item['total_tokens']} tokens)"
                )
            # sys.exit()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_job = {
                executor.submit(self.evaluate_single_run, *job): job
                for job in jobs
            }
            total_jobs = len(future_to_job)
            total_prompt_tokens_used = 0
            total_completion_tokens_used = 0
            total_tokens_used = 0
            api_usage_count = 0
            estimated_usage_count = 0
            success_count = 0
            failed_count = 0
            failed_jobs = []
            progress = None
            if tqdm is not None:
                progress = tqdm(total=total_jobs, desc="LLM requests", unit="req")

            for completed_jobs, future in enumerate(as_completed(future_to_job), start=1):
                scenario_folder, scenario_path, messages, _, index, _, has_collision = future_to_job[future]
                result = future.result()
                scenario_folder_name = result["scenario_folder_name"]
                scenario_stem = result["scenario_stem"]

                if result["success"]:
                    usage = result["usage"]
                    success_count += 1
                    total_prompt_tokens_used += usage["prompt_tokens"]
                    total_completion_tokens_used += usage["completion_tokens"]
                    total_tokens_used += usage["total_tokens"]
                    if usage["source"] == "api":
                        api_usage_count += 1
                    else:
                        estimated_usage_count += 1
                else:
                    failed_count += 1
                    failed_jobs.append(
                        {
                            "scenario_folder_name": scenario_folder_name,
                            "scenario_stem": scenario_stem,
                            "index": result["index"],
                            "error": result["error"],
                        }
                    )

                if progress is not None:
                    status = "ok" if result["success"] else "failed"
                    progress.set_postfix_str(f"{scenario_folder_name}/{scenario_stem}#{index} {status}")
                    progress.update(1)
                else:
                    print(f"{scenario_folder_name} {scenario_stem} ================================================")
                    print(f"model: {model}")
                    print(f"index: {index}")
                    print(f"has_collision: {has_collision}")
                    print(f"status: {'success' if result['success'] else 'failed'}")
                    print(f"completed: {completed_jobs}/{total_jobs}\n")

            if progress is not None:
                progress.close()

            print(f"requests_completed: {total_jobs}")
            print(f"requests_succeeded: {success_count}")
            print(f"requests_failed: {failed_count}")
            print(f"prompt_tokens_used: {total_prompt_tokens_used}")
            print(f"completion_tokens_used: {total_completion_tokens_used}")
            print(f"total_tokens_used: {total_tokens_used}")
            print(f"usage_count_api: {api_usage_count}")
            print(f"usage_count_estimated: {estimated_usage_count}")
            for failed_job in failed_jobs[:10]:
                print(
                    f"failed_request: {failed_job['scenario_folder_name']} / "
                    f"{failed_job['scenario_stem']} #{failed_job['index']} -> {failed_job['error']}"
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate scenarios from a specific scenario folder.")
    parser.add_argument(
        "--input_path",
        default=None,
        help="Specific scenario folder to evaluate. Accepts an absolute path, a path relative to the project root, or a folder name.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Directory where evaluation results will be written. Accepts an absolute path or a path relative to the project root.",
    )
    args = parser.parse_args()

    input_path = args.input_path or os.getenv("INPUT_PATH") or os.getenv("SCENARIO_FOLDER")
    output_path = args.output_path or os.getenv("OUTPUT_PATH")

    if not input_path:
        parser.error("input_path is required. Pass --input_path or set INPUT_PATH.")

    llmapi = LLMAPI()
    llmapi.outputs_root = llmapi.resolve_output_path(output_path)

    model = os.getenv("MODEL", "gpt-oss-20b")

    repeat_time = int(os.getenv("REPEAT_TIME", "10"))

    llmapi.evaluate_R_MR_extra_4full_experiments(model, repeat_time, input_path)
