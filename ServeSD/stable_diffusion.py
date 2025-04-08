from io import BytesIO
from fastapi import FastAPI
from fastapi.responses import Response
import torch

from ray import serve
from ray.serve.handle import DeploymentHandle


app = FastAPI()


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, diffusion_model_handle: DeploymentHandle) -> None:
        self.handle = diffusion_model_handle

    @app.get(
        "/imagine",
        responses={200: {"content": {"image/png": {}}}},
        response_class=Response,
    )
    async def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"

        image = await self.handle.generate.remote(prompt, img_size=img_size)
        file_stream = BytesIO()
        image.save(file_stream, "PNG")
        return Response(content=file_stream.getvalue(), media_type="image/png")


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 0, "max_replicas": 2},
)
class StableDiffusionV2:
    def __init__(self):
        # Install compatible versions if needed (uncomment one of these options)
        # import os
        # os.system("pip install huggingface_hub==0.12.1 diffusers==0.14.0")  # Option 1: Downgrade both to compatible versions
        # os.system("pip install -U diffusers transformers")                  # Option 2: Upgrade diffusers
        
        from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
        import importlib
        import sys
        
        # Verify imports will work before attempting to load models
        try:
            model_id = "stabilityai/stable-diffusion-2"
            
            scheduler = EulerDiscreteScheduler.from_pretrained(
                model_id, subfolder="scheduler"
            )
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
            )
            self.pipe = self.pipe.to("cuda")
        except ImportError as e:
            print(f"Error loading dependencies: {e}")
            print("Try installing compatible versions with:")
            print("pip install huggingface_hub==0.12.1 diffusers==0.14.0")
            print("Or update with: pip install -U diffusers transformers huggingface_hub")
            raise

    def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"

        with torch.autocast("cuda"):
            image = self.pipe(prompt, height=img_size, width=img_size).images[0]
            return image


entrypoint = APIIngress.bind(StableDiffusionV2.bind())
