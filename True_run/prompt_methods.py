import re
from typing import Callable, Dict, List


PROMPT_METHODS = ["vanilla", "zero_shot", "few_shot", "cot", "few_shot_cot"]


def format_options_with_labels(candidates: List[str]) -> str:
    labels = ["A", "B", "C", "D", "E", "F"]
    lines = []
    for idx, cand in enumerate(candidates):
        if idx >= len(labels):
            break
        lines.append(f"{labels[idx]}. {cand}")
    return "\n".join(lines)


def vanilla_prompt(question: str, candidates: List[str]) -> str:
    return (
        "Please answer the question based on the video and choose the correct option.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{format_options_with_labels(candidates)}\n\n"
        "Reply with only one option text or one option letter."
    )


def zero_shot_prompt(question: str, candidates: List[str]) -> str:
    return (
        "You are an expert in video understanding and temporal reasoning.\n"
        "Choose the best option according to the video.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{format_options_with_labels(candidates)}\n\n"
        "Reply with only one option text or one option letter."
    )


def few_shot_prompt(question: str, candidates: List[str]) -> str:
    return (
        "You are an expert in video understanding.\n\n"
        "Example 1:\n"
        "Question: What will the person do next?\n"
        "A. Put down the cup\n"
        "B. Take the book\n"
        "C. Open the door\n"
        "D. Sit down\n"
        "Answer: Take the book\n\n"
        "Example 2:\n"
        "Question: Which object was taken by the person?\n"
        "A. The pillow\n"
        "B. The blanket\n"
        "C. The book\n"
        "D. The cup\n"
        "Answer: The blanket\n\n"
        "Now solve the new question.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{format_options_with_labels(candidates)}\n\n"
        "Reply with only one option text or one option letter."
    )


def cot_prompt(question: str, candidates: List[str]) -> str:
    return (
        "You are an expert in video understanding.\n"
        "Think step by step, then provide the final answer.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{format_options_with_labels(candidates)}\n\n"
        "Format:\n"
        "Reasoning: ...\n"
        "Answer: <option text or letter>"
    )


def few_shot_cot_prompt(question: str, candidates: List[str]) -> str:
    return (
        "You are an expert in video understanding.\n\n"
        "Example:\n"
        "Question: What will the person do next?\n"
        "A. Put down the cup\n"
        "B. Take the book\n"
        "C. Open the door\n"
        "D. Sit down\n"
        "Reasoning: The person reaches toward the book and does not move to the door.\n"
        "Answer: Take the book\n\n"
        "Now solve the new question with reasoning.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{format_options_with_labels(candidates)}\n\n"
        "Format:\n"
        "Reasoning: ...\n"
        "Answer: <option text or letter>"
    )


PROMPT_BUILDERS: Dict[str, Callable[[str, List[str]], str]] = {
    "vanilla": vanilla_prompt,
    "zero_shot": zero_shot_prompt,
    "few_shot": few_shot_prompt,
    "cot": cot_prompt,
    "few_shot_cot": few_shot_cot_prompt,
}


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s/]", "", text)
    return text


def extract_choice_letter(text: str, num_candidates: int) -> str:
    if not text:
        return ""

    labels = ["A", "B", "C", "D", "E", "F"][: max(1, num_candidates)]
    if not labels:
        return ""

    label_group = "".join(labels)
    text = text.strip()

    patterns = [
        rf"Answer\s*[:：]\s*([{label_group}])\b",
        rf"Option\s*[:：]?\s*([{label_group}])\b",
        rf"Choice\s*[:：]?\s*([{label_group}])\b",
        rf"\(([{label_group}])\)",
        rf"\b([{label_group}])\b",
    ]

    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    return ""


def extract_choice_text(text: str) -> str:
    if not text:
        return ""

    text = str(text).strip()

    patterns = [
        r"Answer\s*[:：]\s*(.+)",
        r"Option\s*[:：]?\s*(.+)",
        r"Choice\s*[:：]?\s*(.+)",
    ]

    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()

    return text


def map_answer_to_candidate(model_output: str, candidates: List[str]) -> str:
    if model_output is None:
        return ""

    model_output = str(model_output).strip()
    if not model_output:
        return ""

    if not candidates:
        return ""

    labels = ["A", "B", "C", "D", "E", "F"]

    # 1) 优先按字母解析
    letter = extract_choice_letter(model_output, len(candidates))
    if letter and letter in labels:
        idx = labels.index(letter)
        if idx < len(candidates):
            return candidates[idx]

    # 2) 再提取 Answer: 后面的文本
    answer_text = extract_choice_text(model_output)
    answer_norm = normalize_text(answer_text)
    output_norm = normalize_text(model_output)

    if not answer_norm and not output_norm:
        return ""

    # 3) 先做精确标准化匹配
    for cand in candidates:
        cand_norm = normalize_text(cand)
        if cand_norm and (cand_norm == answer_norm or cand_norm == output_norm):
            return cand

    # 4) 再做包含匹配
    for cand in candidates:
        cand_norm = normalize_text(cand)
        if not cand_norm:
            continue
        if answer_norm and (cand_norm in answer_norm or answer_norm in cand_norm):
            return cand
        if output_norm and (cand_norm in output_norm or output_norm in cand_norm):
            return cand

    # 5) 最后做词重叠匹配
    best_cand = ""
    best_overlap = 0

    answer_tokens = set(answer_norm.split()) if answer_norm else set()
    output_tokens = set(output_norm.split()) if output_norm else set()

    for cand in candidates:
        cand_norm = normalize_text(cand)
        if not cand_norm:
            continue
        cand_tokens = set(cand_norm.split())

        overlap = 0
        if answer_tokens:
            overlap = max(overlap, len(cand_tokens & answer_tokens))
        if output_tokens:
            overlap = max(overlap, len(cand_tokens & output_tokens))

        if overlap > best_overlap:
            best_overlap = overlap
            best_cand = cand

    return best_cand if best_overlap > 0 else ""


def map_answer_to_letter(model_output: str, candidates: List[str]) -> str:
    pred_candidate = map_answer_to_candidate(model_output, candidates)
    if not pred_candidate:
        return ""

    labels = ["A", "B", "C", "D", "E", "F"]
    for idx, cand in enumerate(candidates):
        if idx >= len(labels):
            break
        if normalize_text(cand) == normalize_text(pred_candidate):
            return labels[idx]

    return ""