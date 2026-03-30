import copy
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*copying from a non-meta parameter in the checkpoint to a meta parameter.*",
)

warnings.filterwarnings(
    "ignore",
    message=r"`do_sample` is set to `False`\. However, `temperature` is set to `.*` -- this flag is only used in sample-based generation modes\..*",
)

warnings.filterwarnings(
    "ignore",
    message=r"`do_sample` is set to `False`\. However, `top_p` is set to `.*` -- this flag is only used in sample-based generation modes\..*",
)

warnings.filterwarnings(
    "ignore",
    message=r"`do_sample` is set to `False`\. However, `top_k` is set to `.*` -- this flag is only used in sample-based generation modes\..*",
)


class LLaVANextVideoRunner:
    def __init__(
        self,
        llava_repo_path: Optional[str] = None,
        model_path: Optional[str] = None,
        max_frames_num: int = 32,
        max_new_tokens: int = 64,
        debug: bool = False,
        verbose: bool = True,
    ):
        local_repo_guess = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "LLaVA-NeXT-main")
        )
        self.llava_repo_path = llava_repo_path or os.environ.get(
            "LLAVA_REPO_PATH", local_repo_guess
        )
        self.model_path = model_path or os.environ.get(
            "LLAVA_MODEL_PATH", "/mnt/workspace/models/llava"
        )
        self.max_frames_num = max_frames_num
        self.max_new_tokens = max_new_tokens
        self.debug = debug
        self.verbose = verbose

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.image_processor = None

        self.model_name = "llava_qwen"
        self.conv_mode = "qwen_1_5"

        self._conv_templates = None
        self._tokenizer_image_token = None
        self._IMAGE_TOKEN_INDEX = None
        self._DEFAULT_IMAGE_TOKEN = None
        self._DEFAULT_IM_START_TOKEN = None
        self._DEFAULT_IM_END_TOKEN = None

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _debug(self, msg: str):
        if self.debug:
            print(msg)

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
            from llava.mm_utils import tokenizer_image_token
            from llava.model.builder import load_pretrained_model
        except ImportError:
            if not os.path.isdir(self.llava_repo_path):
                raise FileNotFoundError(
                    f"无法导入 llava，且 LLAVA_REPO_PATH 不存在: {self.llava_repo_path}"
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
            from llava.mm_utils import tokenizer_image_token
            from llava.model.builder import load_pretrained_model

        if self.device == "cuda":
            torch_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            device_map = "auto"
            attn_implementation = "sdpa"
        else:
            torch_dtype = "float16"
            device_map = "cpu"
            attn_implementation = "sdpa"

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=self.model_path,
            model_base=None,
            model_name=self.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )

        self.model.eval()

        # 与贪心解码对齐；某些 transformers 版本即使设为 None 仍会 warning，
        # 所以上面已经加了 warnings.filterwarnings 做最终兜底。
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.do_sample = False
            for attr in ["temperature", "top_p", "top_k"]:
                if hasattr(self.model.generation_config, attr):
                    try:
                        setattr(self.model.generation_config, attr, None)
                    except Exception:
                        pass

        self._conv_templates = conv_templates
        self._tokenizer_image_token = tokenizer_image_token
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self._DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self._DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN

        self._log(f"[LLaVA] model={self.model_path}")
        self._log(f"[LLaVA] model_name={self.model_name}")
        self._log(
            f"[LLaVA] conv_mode={self.conv_mode}, device={self.device}, dtype={torch_dtype}"
        )
        self._log(
            f"[LLaVA] attn_implementation={attn_implementation}, device_map={device_map}"
        )
        self._debug(f"[DEBUG] model_class={self.model.__class__.__name__}")

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
        frame_times = ",".join(
            [f"{(i / fps):.2f}s" if fps > 0 else "0.00s" for i in indices]
        )
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

        frame_times = ",".join(
            [f"{(i / fps):.2f}s" if fps > 0 else "0.00s" for i in indices]
        )
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
            self._log(f"[Warn] decord 读取失败，尝试 OpenCV: {decord_err}")
            return self._load_video_cv2(video_path, start_sec, end_sec)

    @staticmethod
    def _postprocess_model_output(text: str) -> str:
        if not text:
            return ""

        text = str(text).strip()

        m = re.search(r"Answer\s*[:：]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()

        text = re.sub(r"^\s*assistant\s*[:：]\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*response\s*[:：]\s*", "", text, flags=re.IGNORECASE)

        return text.strip()

    def generate_answer(
        self, video_path: str, prompt: str, sample: Optional[Dict] = None
    ) -> str:
        from PIL import Image

        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频不存在: {video_path}")

        frames = None
        pil_frames = None
        processed = None
        video_tensor = None
        video = None
        input_ids = None
        attention_mask = None
        output_ids = None
        decode_tokens = None

        try:
            frames, frame_time, duration = self.load_video(video_path, sample)

            if frames is None:
                raise RuntimeError("load_video 返回 frames=None")
            if not isinstance(frames, np.ndarray):
                raise TypeError(f"frames 不是 np.ndarray，而是 {type(frames)}")
            if len(frames) == 0:
                raise RuntimeError("frames 为空")

            pil_frames = [Image.fromarray(frame.astype("uint8")) for frame in frames]

            processed = self.image_processor.preprocess(pil_frames, return_tensors="pt")
            if processed is None:
                raise RuntimeError("image_processor.preprocess 返回 None")
            if "pixel_values" not in processed:
                raise RuntimeError(
                    f"preprocess 输出不含 pixel_values，keys={list(processed.keys())}"
                )

            video_tensor = processed["pixel_values"]
            if video_tensor is None:
                raise RuntimeError("video_tensor 为 None")

            self._debug(f"[DEBUG] video_path={video_path}")
            self._debug(f"[DEBUG] frames.shape={frames.shape}")
            self._debug(
                f"[DEBUG] video_tensor.shape={video_tensor.shape}, dtype={video_tensor.dtype}"
            )

            if self.device == "cuda":
                video_tensor = video_tensor.cuda()

            model_dtype = getattr(self.model, "dtype", video_tensor.dtype)
            if video_tensor.dtype != model_dtype:
                video_tensor = video_tensor.to(dtype=model_dtype)

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

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
            if self.device == "cuda":
                attention_mask = attention_mask.cuda()

            self._debug(f"[DEBUG] input_ids.shape={input_ids.shape}")
            self._debug(f"[DEBUG] attention_mask.shape={attention_mask.shape}")

            generate_kwargs = {
                "inputs": input_ids,
                "images": video,
                "modalities": ["video"],
                "attention_mask": attention_mask,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
            }

            with torch.inference_mode():
                output_ids = self.model.generate(**generate_kwargs)

            self._debug(f"[DEBUG] output_ids.shape={tuple(output_ids.shape)}")
            self._debug(f"[DEBUG] input_len={input_ids.shape[1]}")

            if output_ids.shape[1] > input_ids.shape[1]:
                decode_tokens = output_ids[0, input_ids.shape[1]:]
                decode_source = "sliced_new_tokens"
            else:
                decode_tokens = output_ids[0]
                decode_source = "full_output_ids"

            self._debug(f"[DEBUG] decode_source={decode_source}")
            self._debug(f"[DEBUG] decode_tokens.shape={tuple(decode_tokens.shape)}")
            self._debug(f"[DEBUG] decode_tokens.tolist()={decode_tokens.tolist()}")

            decoded_raw = self.tokenizer.decode(
                decode_tokens, skip_special_tokens=False
            )
            decoded_clean = self.tokenizer.decode(
                decode_tokens, skip_special_tokens=True
            ).strip()

            self._debug(f"[DEBUG] decoded_raw={repr(decoded_raw)}")
            self._debug(f"[DEBUG] decoded_clean={repr(decoded_clean)}")

            final_output = self._postprocess_model_output(decoded_clean)
            self._debug(f"[DEBUG] final_output={repr(final_output)}")

            return final_output

        finally:
            del frames
            del pil_frames
            del processed
            del video_tensor
            del video
            del input_ids
            del attention_mask
            del output_ids
            del decode_tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()