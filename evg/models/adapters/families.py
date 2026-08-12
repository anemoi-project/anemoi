from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from evg.models.adapters.base import AdapterError, VideoModelAdapter
from evg.models.adapters.diffusers import DiffusersVideoAdapter
from evg.types import GeneratedArtifact, GenerationRequest


class Wan22Adapter(DiffusersVideoAdapter):
    backend = "wan2.2-diffusers"


class MiniMaxH3Adapter(VideoModelAdapter):
    backend = "minimax-h3-sm89-mixed-attention"
    install_extras = ("minimax-h3",)
    entrypoint = "scripts/run_minimax_h3.sh"
    can_execute = True

    def generate(self, request: GenerationRequest) -> GeneratedArtifact:
        self.validate_request(request)
        if request.prompt != "official-example-1":
            raise AdapterError(
                "the current MiniMax-H3 release accepts the bundled "
                "'official-example-1' conditioning only"
            )
        candidate = str(
            request.extra.get("candidate", "mpa-sm89-regular2d-mixed")
        )
        if candidate not in {
            "dense",
            "official-sol",
            "mpa-sm89-regular2d-mixed",
        }:
            raise AdapterError(f"unsupported MiniMax-H3 candidate: {candidate}")
        repository = Path(__file__).resolve().parents[3]
        script = repository / "scripts/run_minimax_h3.sh"
        artifact_dir = request.output.parent / f"{request.output.stem}.artifacts"
        subprocess.run(
            (str(script), candidate, str(artifact_dir)),
            cwd=repository,
            check=True,
        )
        generated = artifact_dir / "out.mp4"
        if not generated.is_file():
            raise AdapterError(f"MiniMax-H3 runner did not produce {generated}")
        request.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated, request.output)
        return GeneratedArtifact(
            path=request.output,
            task=request.task,
            model=self.spec.id,
            variant=self.variant.id,
            metadata={"backend": self.backend, "candidate": candidate},
        )


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
