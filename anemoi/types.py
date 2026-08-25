from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class AnemoiEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class TaskType(AnemoiEnum):
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_IMAGE = "image-to-image"
    TEXT_TO_VIDEO = "text-to-video"
    IMAGE_TO_VIDEO = "image-to-video"
    TEXT_IMAGE_TO_VIDEO = "text-image-to-video"
    VIDEO_TO_VIDEO = "video-to-video"
    VIDEO_EDITING = "video-editing"
    VIDEO_CONTINUATION = "video-continuation"
    REFERENCE_TO_VIDEO = "reference-to-video"
    AUDIO_TO_VIDEO = "audio-to-video"
    SPEECH_TO_VIDEO = "speech-to-video"
    TALKING_AVATAR = "talking-avatar"
    TEXT_TO_VIDEO_WITH_SOUND = "text-to-video-with-sound"
    ACTION_CONDITIONED = "action-conditioned"
    SUPER_RESOLUTION = "super-resolution"
    REFINEMENT = "refinement"


class RuntimeKind(AnemoiEnum):
    DIFFUSERS = "diffusers"
    FASTVIDEO = "fastvideo"
    NATIVE = "native"
    OMNI = "omni"
    UPSTREAM_SCRIPT = "upstream-script"
    HYBRID = "hybrid"


class SupportStatus(AnemoiEnum):
    METADATA_ONLY = "metadata-only"
    ADAPTER_SCAFFOLDED = "adapter-scaffolded"
    EXPERIMENTAL = "experimental"
    SUPPORTED = "supported"


@dataclass(frozen=True)
class MediaInput:
    path: str
    kind: str


@dataclass(frozen=True)
class GenerationRequest:
    model: str
    prompt: str
    output: Path
    task: TaskType = TaskType.TEXT_TO_VIDEO
    variant: str | None = None
    negative_prompt: str | None = None
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    num_frames: int | None = None
    fps: float | None = None
    duration: float | None = None
    media: tuple[MediaInput, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedArtifact:
    path: Path
    task: TaskType
    model: str
    variant: str
    metadata: dict[str, Any] = field(default_factory=dict)
