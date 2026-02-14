import asyncio
import logging
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


logger = logging.getLogger(__name__)


class _SoundFileCompat:
    """Minimal compatibility shim for transformers Voxtral processor."""

    @staticmethod
    def write(file_obj, array, samplerate: int, format: str = "WAV"):
        fmt = (format or "WAV").upper()
        if fmt != "WAV":
            raise ValueError(f"Unsupported audio format for compat writer: {format}")
        pcm = np.asarray(array, dtype=np.float32).reshape(-1)
        pcm = np.clip(pcm, -1.0, 1.0)
        pcm_i16 = (pcm * 32767.0).astype(np.int16, copy=False)
        with wave.open(file_obj, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(samplerate))
            wf.writeframes(pcm_i16.tobytes())
        file_obj.seek(0)


def _resolve_soundfile_backend():
    try:
        import soundfile as sf_backend
        return sf_backend
    except Exception:
        return _SoundFileCompat


class UserSTTEngine:
    """Streaming-friendly user STT engine backed by Voxtral."""

    def __init__(
        self,
        model_id: str = "mistralai/Voxtral-Mini-3B-2507",
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        language: str = "en",
        chunk_seconds: float = 2.0,
        stride_seconds: float = 1.0,
        max_new_tokens: int = 128,
    ):
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be > 0")
        if stride_seconds <= 0:
            raise ValueError("stride_seconds must be > 0")
        if stride_seconds > chunk_seconds:
            stride_seconds = chunk_seconds

        self.model_id = model_id
        self.language = language.strip()
        self.chunk_seconds = chunk_seconds
        self.stride_seconds = stride_seconds
        self.max_new_tokens = max_new_tokens
        self._dtype = dtype
        self._device = torch.device(device) if isinstance(device, str) else device
        self._thread_pool = ThreadPoolExecutor(max_workers=1)
        self._lock = asyncio.Lock()
        self._last_transcript: str = ""

        from transformers import AutoProcessor

        try:
            from transformers import VoxtralForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Voxtral classes are unavailable. Please install transformers with Voxtral support (>= 4.57)."
            ) from exc

        logger.info("Loading user STT model: %s", model_id)
        # Work around a transformers Processor __repr__/deepcopy crash seen
        # with some Voxtral tokenizer artifacts on startup.
        from transformers.processing_utils import ProcessorMixin
        original_repr = ProcessorMixin.__repr__
        ProcessorMixin.__repr__ = lambda self: f"{self.__class__.__name__}()"
        try:
            self._processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        finally:
            ProcessorMixin.__repr__ = original_repr

        # Work around transformers Voxtral ndarray path bug when soundfile isn't installed:
        # processing_voxtral.py expects module-global `sf`.
        import transformers.models.voxtral.processing_voxtral as processing_voxtral
        sf_backend = _resolve_soundfile_backend()
        if getattr(processing_voxtral, "sf", None) is None:
            processing_voxtral.sf = sf_backend

        try:
            import mistral_common.audio as mistral_audio
            if getattr(mistral_audio, "sf", None) is None:
                mistral_audio.sf = sf_backend
        except Exception:
            pass

        processor_sr = getattr(getattr(self._processor, "feature_extractor", None), "sampling_rate", None)
        self._target_sample_rate = int(processor_sr) if processor_sr is not None else 16000
        self._model = VoxtralForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(self._device)
        self._model.eval()
        logger.info("User STT model loaded on %s (expected_sr=%d)", self._device, self._target_sample_rate)

    def window_samples(self, sample_rate: int) -> int:
        return max(1, int(self.chunk_seconds * sample_rate))

    def stride_samples(self, sample_rate: int) -> int:
        return max(1, int(self.stride_seconds * sample_rate))

    @staticmethod
    def _resample_linear(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr:
            return pcm
        if pcm.size == 0:
            return pcm
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        out_len = int(round(pcm.shape[0] * float(dst_sr) / float(src_sr)))
        out_len = max(1, out_len)
        if out_len == pcm.shape[0]:
            return pcm
        x_old = np.arange(pcm.shape[0], dtype=np.float32)
        x_new = np.linspace(0.0, float(pcm.shape[0] - 1), num=out_len, dtype=np.float32)
        return np.interp(x_new, x_old, pcm).astype(np.float32, copy=False)

    def _transcribe_blocking(self, pcm: np.ndarray, sample_rate: int) -> str | None:
        audio = pcm.astype(np.float32, copy=False).reshape(-1)
        sr = int(sample_rate)
        if sr != self._target_sample_rate:
            audio = self._resample_linear(audio, sr, self._target_sample_rate)
            sr = self._target_sample_rate

        kwargs = {
            "audio": [audio],
            "model_id": self.model_id,
            "sampling_rate": sr,
            "format": ["WAV"],
            "return_tensors": "pt",
        }
        if self.language:
            kwargs["language"] = self.language

        inputs = self._processor.apply_transcription_request(**kwargs)
        inputs = inputs.to(self._device, dtype=self._dtype)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        text = self._processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        if not text:
            return None
        transcript = " ".join(text[0].strip().split())
        return transcript or None

    async def transcribe(self, pcm: np.ndarray, sample_rate: int) -> str | None:
        if pcm.size == 0:
            return None

        async with self._lock:
            loop = asyncio.get_running_loop()
            try:
                text = await loop.run_in_executor(
                    self._thread_pool,
                    self._transcribe_blocking,
                    pcm.copy(),
                    sample_rate,
                )
            except Exception:
                logger.exception("User STT transcription failed")
                return None

            if not text:
                return None

            normalized = text.lower()
            if self._last_transcript and (
                normalized == self._last_transcript
                or normalized in self._last_transcript
                or self._last_transcript in normalized
            ):
                return None

            self._last_transcript = normalized
            return text
