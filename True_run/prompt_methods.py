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
        "Output only the answer letter."
    )


def zero_shot_prompt(question: str, candidates: List[str]) -> str:
    return (
        "You are an expert in video understanding and temporal reasoning.\n"
        "Choose the best option according to the video.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{format_options_with_labels(candidates)}\n\n"
        "Answer with only one letter."
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
        "Answer: B\n\n"
        "Example 2:\n"
        "Question: Which object was taken by the person?\n"
        "A. The pillow\n"
        "B. The blanket\n"
        "C. The book\n"
        "D. The cup\n"
        "Answer: B\n\n"
        "Now solve the new question.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{format_options_with_labels(candidates)}\n\n"
        "Answer:"
    )


def cot_prompt(question: str, candidates: List[str]) -> str:
    return (
        "You are an expert in video understanding.\n"
        "Think step by step, then provide the final option letter.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{format_options_with_labels(candidates)}\n\n"
        "Format:\n"
        "Reasoning: ...\n"
        "Answer: <letter>"
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
        "Answer: B\n\n"
        "Now solve the new question with reasoning.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{format_options_with_labels(candidates)}\n\n"
        "Format:\n"
        "Reasoning: ...\n"
        "Answer: <letter>"
    )


PROMPT_BUILDERS: Dict[str, Callable[[str, List[str]], str]] = {
    "vanilla": vanilla_prompt,
    "zero_shot": zero_shot_prompt,
    "few_shot": few_shot_prompt,
    "cot": cot_prompt,
    "few_shot_cot": few_shot_cot_prompt,
}


def extract_choice_letter(text: str, num_candidates: int) -> str:
    if not text:
        return ""

    labels = ["A", "B", "C", "D", "E", "F"][: max(1, num_candidates)]
    label_group = "".join(labels)
    text = text.strip()

    m = re.search(rf"Answer\s*[:：]\s*([{label_group}])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(rf"\b([{label_group}])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(rf"\(([{label_group}])\)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return ""


def map_answer_to_letter(model_output: str, candidates: List[str]) -> str:
    letter = extract_choice_letter(model_output, len(candidates))
    if letter:
        return letter

    labels = ["A", "B", "C", "D", "E", "F"]
    output_lower = model_output.lower().strip()
    best = ""
    max_overlap = 0

    for idx, cand in enumerate(candidates):
        if idx >= len(labels):
            break
        cand_lower = cand.lower().strip()
        if cand_lower in output_lower or output_lower in cand_lower:
            return labels[idx]

        overlap = len(set(cand_lower.split()) & set(output_lower.split()))
        if overlap > max_overlap:
            max_overlap = overlap
            best = labels[idx]

    return best if max_overlap > 0 else ""
