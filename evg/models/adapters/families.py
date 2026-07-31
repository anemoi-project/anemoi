from __future__ import annotations

from evg.models.adapters.base import AdapterError, VideoModelAdapter
from evg.models.adapters.diffusers import DiffusersVideoAdapter
from evg.types import GeneratedArtifact, GenerationRequest


class Wan22Adapter(DiffusersVideoAdapter):
    backend = "wan2.2-diffusers"


class HunyuanVideo15Adapter(DiffusersVideoAdapter):
    backend = "hunyuanvideo-1.5-diffusers"


class LingBotVideoAdapter(DiffusersVideoAdapter):
    backend = "lingbot-video-diffusers"


class LongCatVideoAdapter(DiffusersVideoAdapter):
    backend = "longcat-video-diffusers"


class Cosmos3Adapter(DiffusersVideoAdapter):
    backend = "cosmos3-omni-diffusers"
    entrypoint = "diffusers.Cosmos3OmniPipeline"


class SkyReelsV3Adapter(DiffusersVideoAdapter):
    backend = "skyreels-v3-diffusers"


class BerniniAdapter(VideoModelAdapter):
    backend = "bernini-upstream-script"
    install_extras = ("runtime",)
    entrypoint = "bytedance/Bernini case-file runner"
    can_execute = False

    def generate(self, request: GenerationRequest) -> GeneratedArtifact:
        self.validate_request(request)
        raise AdapterError(
            "Bernini is scaffolded as a hybrid planner-renderer integration. "
            "The next implementation step is to wrap its case-file runner and map "
            "EVG GenerationRequest fields to assets/testcases JSON."
        )
