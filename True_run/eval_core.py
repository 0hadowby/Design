import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from llava_video_runner import LLaVANextVideoRunner
from prompt_methods import PROMPT_BUILDERS, PROMPT_METHODS, map_answer_to_letter


BASE_DIR = os.path.dirname(__file__)
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ====== 魔塔服务器默认路径 ======
SERVER_DATA_ROOT = "/mnt/workspace/Data/data"
JSON_BGT_DIR = os.path.join(SERVER_DATA_ROOT, "json_bgt")
VIDEO_NAME_DIR = os.path.join(SERVER_DATA_ROOT, "video_name")
ACTION_OBJECT_VIDEO_ROOT_DEFAULT = "/mnt/workspace/Data/data/video/video_action&object"
COUNTERFACTUAL_VIDEO_ROOT_DEFAULT = "/mnt/workspace/Data/data/video/video_counterfactual"


@dataclass
class TaskConfig:
    task_name: str
    answer_json_path: str
    video_names_path: str
    video_root_path: str


def _env_or(default: str, env_key: str) -> str:
    return os.environ.get(env_key, default)


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "action_prediction": TaskConfig(
        task_name="action_prediction",
        answer_json_path=_env_or(
            os.path.join(JSON_BGT_DIR, "action_prediction.json"),
            "ACTION_JSON_PATH",
        ),
        video_names_path=_env_or(
            os.path.join(VIDEO_NAME_DIR, "action_prediction_video_names.txt"),
            "ACTION_NAMES_PATH",
        ),
        video_root_path=_env_or(
            ACTION_OBJECT_VIDEO_ROOT_DEFAULT,
            "ACTION_OBJECT_VIDEO_ROOT",
        ),
    ),
    "object_interaction": TaskConfig(
        task_name="object_interaction",
        answer_json_path=_env_or(
            os.path.join(JSON_BGT_DIR, "object_interaction.json"),
            "OBJECT_JSON_PATH",
        ),
        video_names_path=_env_or(
            os.path.join(VIDEO_NAME_DIR, "object_interaction_video_names.txt"),
            "OBJECT_NAMES_PATH",
        ),
        video_root_path=_env_or(
            ACTION_OBJECT_VIDEO_ROOT_DEFAULT,
            "ACTION_OBJECT_VIDEO_ROOT",
        ),
    ),
    "counterfactual_inference": TaskConfig(
        task_name="counterfactual_inference",
        answer_json_path=_env_or(
            os.path.join(JSON_BGT_DIR, "counterfactual_inference.json"),
            "COUNTERFACTUAL_JSON_PATH",
        ),
        video_names_path=_env_or(
            os.path.join(VIDEO_NAME_DIR, "counterfactual_inference_video_names.txt"),
            "COUNTERFACTUAL_NAMES_PATH",
        ),
        video_root_path=_env_or(
            COUNTERFACTUAL_VIDEO_ROOT_DEFAULT,
            "COUNTERFACTUAL_VIDEO_ROOT",
        ),
    ),
}


def load_video_name_set(path: str) -> set:
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_samples(task_cfg: TaskConfig) -> List[Dict]:
    with open(task_cfg.answer_json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    allow_names = load_video_name_set(task_cfg.video_names_path)
    return [s for s in samples if s.get("video", "").strip() in allow_names]


def build_prompt(method: str, question: str, candidates: List[str]) -> str:
    return PROMPT_BUILDERS[method](question, candidates)


def get_gt_letter(sample: Dict) -> str:
    labels = ["A", "B", "C", "D", "E", "F"]
    answer = (sample.get("answer") or "").strip().lower()
    candidates = sample.get("candidates", [])

    for i, cand in enumerate(candidates):
        if i >= len(labels):
            break
        if cand.strip().lower() == answer:
            return labels[i]

    for i, cand in enumerate(candidates):
        if i >= len(labels):
            break
        c = cand.strip().lower()
        if answer in c or c in answer:
            return labels[i]

    return ""


def resolve_video_path(task_cfg: TaskConfig, video_name: str) -> str:
    return os.path.join(task_cfg.video_root_path, video_name)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def evaluate_one_method(
    runner: LLaVANextVideoRunner,
    task_cfg: TaskConfig,
    method: str,
    max_samples: Optional[int],
    result_dir: str,
) -> Dict:
    samples = load_samples(task_cfg)
    total = len(samples) if max_samples is None else min(max_samples, len(samples))
    correct = 0
    details = []

    print(f"\n===== task={task_cfg.task_name}, method={method}, total={total} =====")
    for idx in range(total):
        sample = samples[idx]
        video_name = sample.get("video", "")
        candidates = sample.get("candidates", [])

        prompt = build_prompt(method, sample.get("question", ""), candidates)
        video_path = resolve_video_path(task_cfg, video_name)
        raw = runner.generate_answer(video_path, prompt, sample=sample)

        pred = map_answer_to_letter(raw, candidates)
        gt = get_gt_letter(sample)
        ok = int(pred == gt and gt != "")
        correct += ok

        details.append(
            {
                "idx": idx,
                "id": sample.get("id"),
                "video": video_name,
                "video_path": video_path,
                "question": sample.get("question"),
                "candidates": candidates,
                "answer_text": sample.get("answer"),
                "gt_letter": gt,
                "pred_letter": pred,
                "correct": ok,
                "raw_output": raw,
            }
        )

        if (idx + 1) % 10 == 0 or idx + 1 == total:
            print(f"progress: {idx + 1}/{total}, acc={correct / (idx + 1):.4f}")

    acc = correct / total if total > 0 else 0.0
    result = {
        "task": task_cfg.task_name,
        "method": method,
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "details": details,
    }

    ensure_dir(result_dir)
    out_json = os.path.join(result_dir, f"results_{task_cfg.task_name}_{method}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {out_json}")
    return result


def save_summary_and_table(
    task_name: str, rows: List[Dict], is_smoke: bool, result_dir: str
) -> Tuple[str, str]:
    ensure_dir(result_dir)
    base = "smoke" if is_smoke else "full"
    summary_json = os.path.join(result_dir, f"summary_{task_name}_{base}.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({"summary": rows}, f, ensure_ascii=False, indent=2)

    vanilla_acc = 0.0
    for r in rows:
        if r["method"] == "vanilla":
            vanilla_acc = r["accuracy"]
            break

    table_csv = os.path.join(result_dir, f"table_{task_name}_{base}.csv")
    with open(table_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "method", "accuracy", "correct", "total", "delta_vs_vanilla"])
        for r in rows:
            writer.writerow(
                [
                    r["task"],
                    r["method"],
                    f"{r['accuracy']:.6f}",
                    r["correct"],
                    r["total"],
                    f"{(r['accuracy'] - vanilla_acc):.6f}",
                ]
            )
    return summary_json, table_csv


def print_table(rows: List[Dict]):
    vanilla_acc = 0.0
    for r in rows:
        if r["method"] == "vanilla":
            vanilla_acc = r["accuracy"]
            break
    print("\nmethod\t\taccuracy\tcorrect/total\tdelta_vs_vanilla")
    for r in rows:
        delta = r["accuracy"] - vanilla_acc
        print(f"{r['method']}\t\t{r['accuracy']:.4f}\t\t{r['correct']}/{r['total']}\t\t{delta:+.4f}")


def run_task_experiment(
    task_key: str,
    is_smoke: bool,
    smoke_samples: int = 3,
    max_frames_num: int = 32,
    result_dir: str = DEFAULT_RESULTS_DIR,
):
    if task_key not in TASK_CONFIGS:
        raise ValueError(f"未知任务: {task_key}, 可选={list(TASK_CONFIGS.keys())}")

    task_cfg = TASK_CONFIGS[task_key]
    max_samples = smoke_samples if is_smoke else None

    runner = LLaVANextVideoRunner(max_frames_num=max_frames_num)
    runner.load_model()

    rows = []
    for method in PROMPT_METHODS:
        out = evaluate_one_method(
            runner=runner,
            task_cfg=task_cfg,
            method=method,
            max_samples=max_samples,
            result_dir=result_dir,
        )
        rows.append(
            {
                "task": out["task"],
                "method": out["method"],
                "accuracy": out["accuracy"],
                "correct": out["correct"],
                "total": out["total"],
            }
        )

    print_table(rows)
    summary_json, table_csv = save_summary_and_table(
        task_name=task_key,
        rows=rows,
        is_smoke=is_smoke,
        result_dir=result_dir,
    )
    print(f"[Saved] {summary_json}")
    print(f"[Saved] {table_csv}")
