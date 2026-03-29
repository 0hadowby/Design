import copy
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class LLaVANextVideoRunner:
    def __init__(
        self,
        llava_repo_path: Optional[str] = None,
        model_path: Optional[str] = None,
        max_frames_num: int = 32,
        max_new_tokens: int = 256,
    ):
        local_repo_guess = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "LLaVA-NeXT-main"))
        self.llava_repo_path = llava_repo_path or os.environ.get(
            "LLAVA_REPO_PATH", local_repo_guess
        )
        self.model_path = model_path or os.environ.get(
            "LLAVA_MODEL_PATH", "/mnt/workspace/models/llava"
        )
        self.max_frames_num = max_frames_num
        self.max_new_tokens = max_new_tokens

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.conv_mode = None

        self._conv_templates = None
        self._tokenizer_image_token = None
        self._IMAGE_TOKEN_INDEX = None
        self._DEFAULT_IMAGE_TOKEN = None
        self._DEFAULT_IM_START_TOKEN = None
        self._DEFAULT_IM_END_TOKEN = None

    @staticmethod
    def _guess_conv_mode(model_path: str) -> str:
        return "qwen_1_5"

    def load_model(self):
        try:
            from llava.constants import (
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IM_END_TOKEN,
                DEFAULT_IM_START_TOKEN,
                IMAGE_TOKEN_INDEX,
            )
            from llava.conversation import conv_templates
            from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
            from llava.model.builder import load_pretrained_model
        except ImportError:
            if not os.path.isdir(self.llava_repo_path):
                raise FileNotFoundError(
                    f"无法导入 llava，且 LLAVA_REPO_PATH 不存在: {self.llava_repo_path}。"
                    "请在魔塔设置 LLAVA_REPO_PATH 后重试。"
                )
            if self.llava_repo_path not in sys.path:
                sys.path.insert(0, self.llava_repo_path)
            from llava.constants import (
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IM_END_TOKEN,
                DEFAULT_IM_START_TOKEN,
                IMAGE_TOKEN_INDEX,
            )
            from llava.conversation import conv_templates
            from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
            from llava.model.builder import load_pretrained_model

        model_name = "llava_qwen"
        self.conv_mode = "qwen_1_5"

        torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=self.model_path,
            model_base=None,
            model_name=model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.model.eval()

        self._conv_templates = conv_templates
        self._tokenizer_image_token = tokenizer_image_token
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self._DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self._DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN

        print(f"[LLaVA] model={self.model_path}")
        print(f"[LLaVA] conv_mode={self.conv_mode}, device={self.device}")

    @staticmethod
    def _uniform_indices(total: int, n: int) -> List[int]:
        if total <= 0:
            return [0]
        if n >= total:
            return list(range(total))
        return np.linspace(0, total - 1, n, dtype=int).tolist()

    def _load_video_decord(
        self, video_path: str, start_sec: Optional[float], end_sec: Optional[float]
    ) -> Tuple[np.ndarray, str, float]:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        fps = float(vr.get_avg_fps()) if vr.get_avg_fps() else 0.0
        duration = total_frames / fps if fps > 0 else 0.0

        start_idx = 0
        end_idx = max(total_frames - 1, 0)
        if fps > 0:
            if start_sec is not None:
                start_idx = min(max(int(start_sec * fps), 0), end_idx)
            if end_sec is not None:
                end_idx = min(max(int(end_sec * fps), start_idx), end_idx)

        clip_len = end_idx - start_idx + 1
        if clip_len <= 0:
            indices = [start_idx]
        else:
            rel = self._uniform_indices(clip_len, self.max_frames_num)
            indices = [start_idx + i for i in rel]

        frames = vr.get_batch(indices).asnumpy()
        frame_times = ",".join([f"{(i / fps):.2f}s" if fps > 0 else "0.00s" for i in indices])
        return frames, frame_times, duration

    def _load_video_cv2(
        self, video_path: str, start_sec: Optional[float], end_sec: Optional[float]
    ) -> Tuple[np.ndarray, str, float]:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法用 OpenCV 打开视频: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0

        start_idx = 0
        end_idx = max(frame_count - 1, 0)
        if fps > 0:
            if start_sec is not None:
                start_idx = min(max(int(start_sec * fps), 0), end_idx)
            if end_sec is not None:
                end_idx = min(max(int(end_sec * fps), start_idx), end_idx)

        clip_len = end_idx - start_idx + 1
        rel = self._uniform_indices(max(clip_len, 1), self.max_frames_num)
        indices = [start_idx + i for i in rel]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if not frames:
            raise RuntimeError(f"OpenCV 读取失败: {video_path}")

        frame_times = ",".join([f"{(i / fps):.2f}s" if fps > 0 else "0.00s" for i in indices])
        return np.asarray(frames), frame_times, duration

    def load_video(
        self, video_path: str, sample: Optional[Dict] = None
    ) -> Tuple[np.ndarray, str, float]:
        sample = sample or {}
        start_sec = sample.get("accurate_start", sample.get("start"))
        end_sec = sample.get("accurate_end", sample.get("end"))

        try:
            return self._load_video_decord(video_path, start_sec, end_sec)
        except Exception as decord_err:
            print(f"[Warn] decord 读取失败，尝试 OpenCV: {decord_err}")
            return self._load_video_cv2(video_path, start_sec, end_sec)

    def generate_answer(self, video_path: str, prompt: str, sample: Optional[Dict] = None) -> str:
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        if not os.path.exists(video_path):
            print(f"[Warn] 视频不存在: {video_path}")
            return ""

        frames, frame_time, duration = self.load_video(video_path, sample)
        video_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        if self.device == "cuda":
            video_tensor = video_tensor.cuda()
        if video_tensor.dtype != getattr(self.model, "dtype", video_tensor.dtype):
            video_tensor = video_tensor.to(dtype=getattr(self.model, "dtype", video_tensor.dtype))
        video = [video_tensor]

        time_instruction = (
            f"The video lasts for {duration:.2f} seconds, and {len(video[0])} frames are uniformly sampled "
            f"from it. These frames are located at {frame_time}. "
            f"Please answer the following multiple-choice question about this video.\n"
        )
        user_prompt = time_instruction + prompt

        if getattr(self.model.config, "mm_use_im_start_end", False):
            question = (
                self._DEFAULT_IM_START_TOKEN
                + self._DEFAULT_IMAGE_TOKEN
                + self._DEFAULT_IM_END_TOKEN
                + "\n"
                + user_prompt
            )
        else:
            question = self._DEFAULT_IMAGE_TOKEN + "\n" + user_prompt

        conv = copy.deepcopy(self._conv_templates[self.conv_mode])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        final_prompt = conv.get_prompt()

        input_ids = self._tokenizer_image_token(
            final_prompt,
            self.tokenizer,
            self._IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0)
        if self.device == "cuda":
            input_ids = input_ids.cuda()

        if self.tokenizer.pad_token_id is None and "qwen" in self.tokenizer.name_or_path.lower():
            self.tokenizer.pad_token_id = 151643

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        if self.device == "cuda":
            attention_mask = attention_mask.cuda()

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=video,
                modalities=["video"],
                attention_mask=attention_mask,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=self.max_new_tokens,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.startswith(final_prompt):
            outputs = outputs[len(final_prompt):].strip()
        return outputs
