from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from openpyxl import Workbook


REQUIRED_KEYS = [
    "realistic",
    "realistic_probability",
    "realistic_confidence",
    "scenario",
    "scenario_probability",
    "scenario_confidence",
]

RUBRIC_OUTPUT_KEYS = [
    "overall_realism_score",
    "probability",
    "confidence",
]

RUBRIC_SCORE_KEYS = [
    "kinematic_plausibility",
    "map_and_junction_consistency",
    "agent_behavioral_realism",
    "interaction_realism",
    "temporal_consistency",
    "realistic_edge_case_formation",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input_root = script_dir / "outputs_results"
    default_output_dir = default_input_root / "metrics_summary"

    parser = argparse.ArgumentParser(
        description=(
            "Extract realism metrics from LLM response .txt files and compute averages by folder."
        )
    )
    parser.add_argument(
        "input_root",
        nargs="?",
        default=None,
        help="Root folder that contains LLM result .txt files.",
    )
    parser.add_argument(
        "--input-root",
        "--input_root",
        dest="input_root_flag",
        default=None,
        help="Root folder that contains LLM result .txt files.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir_flag",
        default=None,
        help="Directory where CSV, XLSX, and JSON summaries will be written.",
    )
    args = parser.parse_args()
    args.input_root = args.input_root_flag or args.input_root or str(default_input_root)
    delattr(args, "input_root_flag")
    args.output_dir = args.output_dir_flag or str(default_output_dir)
    delattr(args, "output_dir_flag")
    return args


def iter_braced_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    stack_depth = 0
    block_start: int | None = None

    for index, char in enumerate(text):
        if char == "{":
            if stack_depth == 0:
                block_start = index
            stack_depth += 1
        elif char == "}":
            if stack_depth == 0:
                continue
            stack_depth -= 1
            if stack_depth == 0 and block_start is not None:
                blocks.append(text[block_start:index + 1])
                block_start = None

    return blocks


def sanitize_candidate(candidate: str) -> str:
    normalized = candidate.strip()
    normalized = normalized.replace("\\_", "_")
    normalized = normalized.replace("\r\n", "\n")
    normalized = normalized.replace("\r", "\n")
    normalized = normalized.replace("“", '"').replace("”", '"')
    normalized = normalized.replace("‘", "'").replace("’", "'")
    normalized = re.sub(r"```(?:json)?", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bTrue\b", "true", normalized)
    normalized = re.sub(r"\bFalse\b", "false", normalized)
    normalized = re.sub(r"(:\s*)(-?\d+(?:\.\d+)?)\s*%", r"\g<1>\g<2>", normalized)
    normalized = re.sub(r",\s*}", "\n}", normalized)

    lines = normalized.splitlines()
    fixed_lines: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            fixed_lines.append(line)
            continue

        fixed_line = line
        next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if (
            stripped not in {"{", "}"}
            and not stripped.endswith(",")
            and next_line.startswith('"')
            and not stripped.endswith("{")
        ):
            fixed_line = f"{line},"
        fixed_lines.append(fixed_line)

    normalized = "\n".join(fixed_lines)
    normalized = re.sub(r",\s*}", "\n}", normalized)
    return normalized


def canonicalize_key(key: str) -> str:
    return key.replace("\\", "").strip().lower()


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def parse_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("%", "")
        cleaned = cleaned.replace("/10.0", "").replace("/10", "")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def normalize_probability(value: float | None) -> float | None:
    if value is None:
        return None
    if 0.0 <= value <= 1.0:
        return value * 100.0
    return value


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    normalized_map: dict[str, Any] = {}
    for key, value in payload.items():
        normalized_map[canonicalize_key(key)] = value

    if all(required_key in normalized_map for required_key in REQUIRED_KEYS):
        realistic_value = parse_bool(normalized_map["realistic"])
        scenario_value = parse_number(normalized_map["scenario"])
        realistic_probability = normalize_probability(parse_number(normalized_map["realistic_probability"]))
        realistic_confidence = normalize_probability(parse_number(normalized_map["realistic_confidence"]))
        scenario_probability = normalize_probability(parse_number(normalized_map["scenario_probability"]))
        scenario_confidence = normalize_probability(parse_number(normalized_map["scenario_confidence"]))

        if realistic_value is None or scenario_value is None:
            return None

        return {
            "realistic": realistic_value,
            "realistic_probability": realistic_probability,
            "realistic_confidence": realistic_confidence,
            "scenario": scenario_value,
            "scenario_probability": scenario_probability,
            "scenario_confidence": scenario_confidence,
        }

    if all(required_key in normalized_map for required_key in RUBRIC_OUTPUT_KEYS):
        scenario_value = parse_number(normalized_map["overall_realism_score"])
        probability = normalize_probability(parse_number(normalized_map["probability"]))
        confidence = normalize_probability(parse_number(normalized_map["confidence"]))

        if scenario_value is None:
            return None

        rubric_fields: dict[str, Any] = {}
        for rubric_key in RUBRIC_SCORE_KEYS:
            rubric_value = normalized_map.get(rubric_key)
            rubric_score = None
            rubric_reasoning = None
            if isinstance(rubric_value, dict):
                rubric_score = parse_number(rubric_value.get("score"))
                reasoning_value = rubric_value.get("reasoning")
                if reasoning_value is not None:
                    rubric_reasoning = str(reasoning_value)

            rubric_fields[f"{rubric_key}_score"] = rubric_score
            rubric_fields[f"{rubric_key}_reasoning"] = rubric_reasoning

        return {
            "realistic": scenario_value >= 3.0,
            "realistic_probability": probability,
            "realistic_confidence": confidence,
            "scenario": scenario_value,
            "scenario_probability": probability,
            "scenario_confidence": confidence,
            "overall_realism_score": scenario_value,
            "probability": probability,
            "confidence": confidence,
            **rubric_fields,
        }

    return None


def try_parse_candidate(candidate: str) -> dict[str, Any] | None:
    if "<" in candidate and ">" in candidate:
        return None

    sanitized = sanitize_candidate(candidate)

    try:
        parsed = json.loads(sanitized)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    return normalize_payload(parsed)


def extract_metrics_from_text(text: str) -> tuple[dict[str, Any] | None, str | None]:
    for candidate in reversed(iter_braced_blocks(text)):
        parsed = try_parse_candidate(candidate)
        if parsed is not None:
            return parsed, candidate
    return None, None


def build_block_preview(matched_block: str | None, max_length: int = 240) -> str | None:
    if matched_block is None:
        return None
    compact = " ".join(matched_block.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3] + "..."


def extract_create_time_seconds(text: str) -> float | None:
    match = re.search(r"create_time:\s*([0-9]+(?:\.[0-9]+)?)s", text)
    if match is None:
        return None
    return float(match.group(1))


def resolve_scenario_json_path(file_path: Path, input_root: Path) -> Path | None:
    relative_path = file_path.relative_to(input_root)
    if input_root.name.endswith("-scenarios"):
        if len(relative_path.parts) < 2:
            return None
        scenario_folder_name = input_root.name
        scenario_stem = relative_path.parts[0]
    else:
        if len(relative_path.parts) < 3:
            return None
        scenario_folder_name = relative_path.parts[0]
        scenario_stem = relative_path.parts[1]

    repo_root = Path(__file__).resolve().parents[1]
    search_roots: list[Path] = []
    for candidate_root in [input_root.parent, input_root.parent.parent, repo_root]:
        if candidate_root not in search_roots:
            search_roots.append(candidate_root)

    for search_root in search_roots:
        scenario_json_path = search_root / scenario_folder_name / f"{scenario_stem}.json"
        if scenario_json_path.exists():
            return scenario_json_path
    return None


def scenario_has_collision(scenario_json_path: Path | None) -> bool | None:
    if scenario_json_path is None:
        return None

    try:
        with scenario_json_path.open("r", encoding="utf-8") as file:
            scenario_data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(scenario_data, dict):
        return None

    for frame_data in scenario_data.values():
        if not isinstance(frame_data, dict):
            continue
        for actor_data in frame_data.values():
            if isinstance(actor_data, dict) and actor_data.get("Collided_With_Ego") is True:
                return True

    return False


def relative_parent(file_path: Path, input_root: Path) -> str:
    relative_path = file_path.relative_to(input_root)
    parent = relative_path.parent.as_posix()
    return parent if parent else "."


def top_level_folder(file_path: Path, input_root: Path) -> str:
    relative_path = file_path.relative_to(input_root)
    if len(relative_path.parts) <= 1:
        return "."
    return relative_path.parts[0]


def compute_mean_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    parsed_records = [record for record in records if record["parsed"]]
    summary: dict[str, Any] = {
        "scenario_count": len(records),
        "parsed_count": len(parsed_records),
        "mean_realistic": None,
        "mean_realistic_probability": None,
        "mean_realistic_confidence": None,
        "mean_scenario": None,
        "mean_scenario_probability": None,
        "mean_scenario_confidence": None,
        "mean_overall_realism_score": None,
        "mean_probability": None,
        "mean_confidence": None,
        "mean_kinematic_plausibility_score": None,
        "mean_map_and_junction_consistency_score": None,
        "mean_agent_behavioral_realism_score": None,
        "mean_interaction_realism_score": None,
        "mean_temporal_consistency_score": None,
        "mean_realistic_edge_case_formation_score": None,
    }

    if not parsed_records:
        return summary

    realistic_values = [1.0 if record["realistic"] else 0.0 for record in parsed_records if record["realistic"] is not None]
    summary["mean_realistic"] = sum(realistic_values) / len(realistic_values) if realistic_values else None

    for metric_name in [
        "realistic_probability",
        "realistic_confidence",
        "scenario",
        "scenario_probability",
        "scenario_confidence",
        "overall_realism_score",
        "probability",
        "confidence",
        "kinematic_plausibility_score",
        "map_and_junction_consistency_score",
        "agent_behavioral_realism_score",
        "interaction_realism_score",
        "temporal_consistency_score",
        "realistic_edge_case_formation_score",
    ]:
        values = [record[metric_name] for record in parsed_records if record[metric_name] is not None]
        summary[f"mean_{metric_name}"] = sum(values) / len(values) if values else None

    return summary


def summarize_records(records: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[group_key])].append(record)

    summaries: list[dict[str, Any]] = []
    for group_name, group_records in sorted(grouped.items()):
        parsed_records = [record for record in group_records if record["parsed"]]
        summary: dict[str, Any] = {
            group_key: group_name,
            "file_count": len(group_records),
            "parsed_count": len(parsed_records),
            "parse_rate": (len(parsed_records) / len(group_records) * 100.0) if group_records else None,
            "collision_true_count": sum(1 for record in group_records if record["has_collision"] is True),
            "collision_false_count": sum(1 for record in group_records if record["has_collision"] is False),
            "collision_unknown_count": sum(1 for record in group_records if record["has_collision"] is None),
            "realistic_true_count": sum(1 for record in parsed_records if record["realistic"] is True),
            "realistic_false_count": sum(1 for record in parsed_records if record["realistic"] is False),
        }
        summary["realistic_true_rate"] = (
            summary["realistic_true_count"] / len(parsed_records) * 100.0
            if parsed_records
            else None
        )
        known_collision_count = summary["collision_true_count"] + summary["collision_false_count"]
        summary["collision_true_rate"] = (
            summary["collision_true_count"] / known_collision_count * 100.0
            if known_collision_count
            else None
        )

        for metric_name in [
            "realistic_probability",
            "realistic_confidence",
            "scenario",
            "scenario_probability",
            "scenario_confidence",
            "overall_realism_score",
            "probability",
            "confidence",
            "kinematic_plausibility_score",
            "map_and_junction_consistency_score",
            "agent_behavioral_realism_score",
            "interaction_realism_score",
            "temporal_consistency_score",
            "realistic_edge_case_formation_score",
            "create_time_seconds",
        ]:
            values = [record[metric_name] for record in parsed_records if record[metric_name] is not None]
            summary[f"mean_{metric_name}"] = sum(values) / len(values) if values else None

        summaries.append(summary)

    return summaries


def write_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with output_path.open("w", encoding="utf-8", newline="") as file:
            file.write("")
        return

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_xlsx(output_path: Path, rows: list[dict[str, Any]], sheet_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = sheet_name

    if not rows:
        workbook.save(output_path)
        return

    headers = list(rows[0].keys())
    worksheet.append(headers)
    for row in rows:
        worksheet.append([row.get(header) for header in headers])

    for column_cells in worksheet.columns:
        max_length = 0
        column_letter = column_cells[0].column_letter
        for cell in column_cells:
            value = "" if cell.value is None else str(cell.value)
            if len(value) > max_length:
                max_length = len(value)
        worksheet.column_dimensions[column_letter].width = min(max_length + 2, 80)

    workbook.save(output_path)


def build_record(file_path: Path, input_root: Path) -> dict[str, Any]:
    text = file_path.read_text(encoding="utf-8")
    extracted_metrics, matched_block = extract_metrics_from_text(text)
    create_time_seconds = extract_create_time_seconds(text)
    scenario_json_path = resolve_scenario_json_path(file_path, input_root)
    has_collision = scenario_has_collision(scenario_json_path)
    workspace_root = Path(__file__).resolve().parents[1]

    record: dict[str, Any] = {
        "relative_file": file_path.relative_to(input_root).as_posix(),
        "parent_folder": relative_parent(file_path, input_root),
        "top_level_folder": top_level_folder(file_path, input_root),
        "scenario_json": scenario_json_path.relative_to(workspace_root).as_posix() if scenario_json_path else None,
        "has_collision": has_collision,
        "parsed": extracted_metrics is not None,
        "matched_block_preview": build_block_preview(matched_block),
        "create_time_seconds": create_time_seconds,
        "realistic": None,
        "realistic_probability": None,
        "realistic_confidence": None,
        "scenario": None,
        "scenario_probability": None,
        "scenario_confidence": None,
        "overall_realism_score": None,
        "probability": None,
        "confidence": None,
        "kinematic_plausibility_score": None,
        "kinematic_plausibility_reasoning": None,
        "map_and_junction_consistency_score": None,
        "map_and_junction_consistency_reasoning": None,
        "agent_behavioral_realism_score": None,
        "agent_behavioral_realism_reasoning": None,
        "interaction_realism_score": None,
        "interaction_realism_reasoning": None,
        "temporal_consistency_score": None,
        "temporal_consistency_reasoning": None,
        "realistic_edge_case_formation_score": None,
        "realistic_edge_case_formation_reasoning": None,
    }

    if extracted_metrics is not None:
        record.update(extracted_metrics)

    return record


def collect_txt_files(input_root: Path) -> list[Path]:
    return sorted(
        [path for path in input_root.rglob("*.txt") if path.is_file()],
        key=lambda path: path.as_posix(),
    )


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_root}")

    txt_files = collect_txt_files(input_root)
    if not txt_files:
        print(f"No .txt files found under: {input_root}")
        return

    records = [build_record(file_path, input_root) for file_path in txt_files]
    folder_summaries = summarize_records(records, "parent_folder")
    top_folder_summaries = summarize_records(records, "top_level_folder")
    collision_records = [record for record in records if record["has_collision"] is True]

    overall_summary = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "file_count": len(records),
        "parsed_count": sum(1 for record in records if record["parsed"]),
        "collision_true_count": sum(1 for record in records if record["has_collision"] is True),
        "collision_false_count": sum(1 for record in records if record["has_collision"] is False),
        "collision_unknown_count": sum(1 for record in records if record["has_collision"] is None),
        "folder_count": len({record["parent_folder"] for record in records}),
        "top_level_folder_count": len({record["top_level_folder"] for record in records}),
        "overall_means": compute_mean_metrics(records),
        "collision_means": compute_mean_metrics(collision_records),
    }

    write_csv(output_dir / "extracted_metrics.csv", records)
    write_csv(output_dir / "folder_summary.csv", folder_summaries)
    write_csv(output_dir / "top_folder_summary.csv", top_folder_summaries)
    write_xlsx(output_dir / "extracted_metrics.xlsx", records, "extracted_metrics")
    write_xlsx(output_dir / "folder_summary.xlsx", folder_summaries, "folder_summary")
    write_xlsx(output_dir / "top_folder_summary.xlsx", top_folder_summaries, "top_folder_summary")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "overall_summary.json").open("w", encoding="utf-8") as file:
        json.dump(overall_summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(overall_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()