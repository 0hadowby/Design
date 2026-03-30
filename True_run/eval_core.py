import csv
import gc
import json
import os
import random
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from llava_video_runner import LLaVANextVideoRunner
from prompt_methods import (
    PROMPT_BUILDERS,
    PROMPT_METHODS,
    map_answer_to_candidate,
    map_answer_to_letter,
    normalize_text,
)


BASE_DIR = os.path.dirname(__file__)
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, "results")

SERVER_DATA_ROOT = "/mnt/workspace/Data/data"
JSON_BGT_DIR = os.path.join(SERVER_DATA_ROOT, "json_bgt")
VIDEO_NAME_DIR = os.path.join(SERVER_DATA_ROOT, "video_name")
ACTION_OBJECT_VIDEO_ROOT_DEFAULT = "/mnt/workspace/Data/data/video/video_action&object"
COUNTERFACTUAL_VIDEO_ROOT_DEFAULT = "/mnt/workspace/Data/data/video/video_counterfactual"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_video_name_set(path: str) -> set:
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def resolve_video_path(task_cfg: TaskConfig, video_name: str) -> str:
    return os.path.join(task_cfg.video_root_path, video_name)


def load_samples(task_cfg: TaskConfig) -> List[Dict]:
    with open(task_cfg.answer_json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    allow_names = load_video_name_set(task_cfg.video_names_path)
    filtered = [s for s in samples if s.get("video", "").strip() in allow_names]
    filtered = sorted(filtered, key=lambda x: str(x.get("video", "")))
    return filtered


def preflight_check(task_cfg: TaskConfig):
    if not os.path.exists(task_cfg.answer_json_path):
        raise FileNotFoundError(f"answer_json_path 不存在: {task_cfg.answer_json_path}")
    if not os.path.exists(task_cfg.video_names_path):
        raise FileNotFoundError(f"video_names_path 不存在: {task_cfg.video_names_path}")
    if not os.path.isdir(task_cfg.video_root_path):
        raise FileNotFoundError(f"video_root_path 不存在: {task_cfg.video_root_path}")

    samples = load_samples(task_cfg)
    missing = []
    for s in samples:
        video_name = s.get("video", "").strip()
        video_path = resolve_video_path(task_cfg, video_name)
        if not os.path.exists(video_path):
            missing.append(video_name)

    if missing:
        print(f"[Warn] {task_cfg.task_name} 缺失视频 {len(missing)} 个，前10个: {missing[:10]}")
    else:
        print(f"[OK] {task_cfg.task_name} 数据检查通过，共 {len(samples)} 条样本")


def build_prompt(method: str, question: str, candidates: List[str]) -> str:
    return PROMPT_BUILDERS[method](question, candidates)


def get_gt_candidate(sample: Dict) -> str:
    answer = (sample.get("answer") or "").strip()
    candidates = sample.get("candidates", [])

    if not answer:
        return ""

    answer_norm = normalize_text(answer)

    for cand in candidates:
        cand_text = (cand or "").strip()
        if not cand_text:
            continue
        if normalize_text(cand_text) == answer_norm:
            return cand_text

    for cand in candidates:
        cand_text = (cand or "").strip()
        if not cand_text:
            continue
        cand_norm = normalize_text(cand_text)
        if answer_norm and (answer_norm in cand_norm or cand_norm in answer_norm):
            return cand_text

    return answer


def get_gt_letter_from_candidate(gt_candidate: str, candidates: List[str]) -> str:
    labels = ["A", "B", "C", "D", "E", "F"]
    gt_norm = normalize_text(gt_candidate)

    if not gt_norm:
        return ""

    for i, cand in enumerate(candidates):
        if i >= len(labels):
            break
        if normalize_text(cand) == gt_norm:
            return labels[i]

    return ""


def evaluate_one_method(
    runner: LLaVANextVideoRunner,
    task_cfg: TaskConfig,
    method: str,
    max_samples: Optional[int],
    result_dir: str,
) -> Dict:
    samples = load_samples(task_cfg)
    if max_samples is not None:
        samples = samples[:max_samples]

    total = len(samples)
    correct = 0
    invalid = 0
    errors = 0
    details = []

    print(f"\n===== task={task_cfg.task_name}, method={method}, total={total} =====")

    for idx, sample in enumerate(samples):
        video_name = sample.get("video", "")
        candidates = sample.get("candidates", [])
        prompt = build_prompt(method, sample.get("question", ""), candidates)
        video_path = resolve_video_path(task_cfg, video_name)

        raw = ""
        pred_text = ""
        pred_letter = ""
        err_msg = ""

        try:
            raw = runner.generate_answer(video_path, prompt, sample=sample)
            pred_text = map_answer_to_candidate(raw, candidates)
            pred_letter = map_answer_to_letter(raw, candidates)
        except Exception:
            err_msg = traceback.format_exc()
            errors += 1

            print("\n" + "=" * 80)
            print(f"[ERROR] task={task_cfg.task_name}, method={method}, idx={idx}")
            print(f"[ERROR] video={video_name}")
            print(f"[ERROR] video_path={video_path}")
            print(err_msg)
            print("=" * 80 + "\n")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        gt_text = get_gt_candidate(sample)
        gt_letter = get_gt_letter_from_candidate(gt_text, candidates)

        if not pred_text:
            invalid += 1

        ok = int(
            bool(pred_text)
            and bool(gt_text)
            and normalize_text(pred_text) == normalize_text(gt_text)
        )
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
                "gt_text": gt_text,
                "gt_letter": gt_letter,
                "pred_text": pred_text,
                "pred_letter": pred_letter,
                "correct": ok,
                "raw_output": raw,
                "error": err_msg,
            }
        )

        if (idx + 1) % 10 == 0 or idx + 1 == total:
            print(
                f"progress: {idx + 1}/{total}, "
                f"acc={correct / (idx + 1):.4f}, invalid={invalid}, errors={errors}"
            )

    acc = correct / total if total > 0 else 0.0
    result = {
        "task": task_cfg.task_name,
        "method": method,
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "errors": errors,
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
        writer.writerow(
            ["task", "method", "accuracy", "correct", "total", "invalid", "errors", "delta_vs_vanilla"]
        )
        for r in rows:
            writer.writerow(
                [
                    r["task"],
                    r["method"],
                    f"{r['accuracy']:.6f}",
                    r["correct"],
                    r["total"],
                    r["invalid"],
                    r["errors"],
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

    print("\nmethod\t\taccuracy\tcorrect/total\tinvalid\terrors\tdelta_vs_vanilla")
    for r in rows:
        delta = r["accuracy"] - vanilla_acc
        print(
            f"{r['method']}\t\t{r['accuracy']:.4f}\t\t"
            f"{r['correct']}/{r['total']}\t\t{r['invalid']}\t{r['errors']}\t{delta:+.4f}"
        )


def run_task_experiment(
    task_key: str,
    is_smoke: bool,
    smoke_samples: int = 3,
    max_frames_num: int = 8,
    result_dir: str = DEFAULT_RESULTS_DIR,
    debug: bool = False,
):
    if task_key not in TASK_CONFIGS:
        raise ValueError(f"未知任务: {task_key}, 可选={list(TASK_CONFIGS.keys())}")

    set_seed(42)

    task_cfg = TASK_CONFIGS[task_key]
    preflight_check(task_cfg)

    max_samples = smoke_samples if is_smoke else None

    print(
        f"[Run] task={task_key}, mode={'smoke' if is_smoke else 'full'}, "
        f"samples={'all' if max_samples is None else max_samples}, max_frames={max_frames_num}, debug={debug}"
    )

    runner = LLaVANextVideoRunner(
        max_frames_num=max_frames_num,
        debug=debug,
        verbose=True,
    )
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
                "invalid": out["invalid"],
                "errors": out["errors"],
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