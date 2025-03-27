import os
import sys
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams

import ray
from ray import serve

# Define request and response models
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt for text generation")
    max_tokens: int = Field(512, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Temperature for sampling")
    top_p: float = Field(0.95, description="Top-p sampling parameter")
    top_k: int = Field(50, description="Top-k sampling parameter")
    stop_sequences: Optional[List[str]] = Field(None, description="Sequences that stop generation")
    
class GenerationResponse(BaseModel):
    generated_text: str

# Define the base class without Ray Serve decoration
class LlamaTextGeneratorBase:
    def __init__(self):
        # Initialize the model
        self.model_id = "meta-llama/Llama-2-7b-hf"
        self.llm = None
        self.app = FastAPI(title="Llama2-7b API", description="API for generating text with Llama2-7b")
        self._setup_routes()
        print(f"Initializing LlamaTextGenerator with model {self.model_id}")
        
        # Get HuggingFace token from environment
        self.hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not self.hf_token:
            print("WARNING: HUGGING_FACE_HUB_TOKEN environment variable is not set")
            
        # Load the model
        print(f"Loading model {self.model_id}...")
        try:
            self.llm = LLM(
                model=self.model_id,
                trust_remote_code=True,
                download_dir=None, 
                dtype="float16",  # Use half precision for efficiency
                gpu_memory_utilization=0.85,
            )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # We don't raise here - will return 503 on requests until model loads

    def _setup_routes(self):
        @self.app.post("/generate", response_model=GenerationResponse)
        async def generate_text(request: GenerationRequest):
            """Generate text based on the input prompt"""
            if self.llm is None:
                raise HTTPException(status_code=503, detail="Model not loaded yet")
                
            try:
                # Configure sampling parameters
                sampling_params = SamplingParams(
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    stop=request.stop_sequences,
                )
                
                # Generate text
                outputs = self.llm.generate([request.prompt], sampling_params)
                generated_text = outputs[0].outputs[0].text
                
                return GenerationResponse(generated_text=generated_text)
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "ok", "model_loaded": self.llm is not None}

        @self.app.get("/-/healthz", response_class=PlainTextResponse)
        async def kubernetes_health_check():
            """Standard health check endpoint for Kubernetes"""
            print("Health check endpoint hit")
            return "success"

    async def __call__(self, request):
        """Handle FastAPI requests"""
        return await self.app(request)

# Create a Ray Serve deployment from the base class
@serve.deployment(
    name="llama2-7b-service",
    num_replicas=1,  # Set to number of workers with GPUs
    ray_actor_options={"num_gpus": 1}  # Request 1 GPU per replica
)
class LlamaTextGenerator(LlamaTextGeneratorBase):
    pass

# Create a deployment graph to expose to Ray Serve
deployment_graph = LlamaTextGenerator.bind()

# Use inline deployment approach instead of module imports
def deploy():
    """Deploy the application to Ray Serve"""
    # Make sure current directory is in the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Define files to exclude from the runtime environment
    exclude_patterns = [
        # Pip cache files
        "**/.cache/pip/**/*.body",
        "**/.cache/pip/**/*.whl",
        # Any other large files you don't need
        "**/__pycache__/**",
        "**/*.pyc",
    ]
    
    # Initialize Ray with runtime environment configuration
    if not ray.is_initialized():
        ray.init(runtime_env={"excludes": exclude_patterns})
    
    # Deploy directly without referencing the module
    serve.run(deployment_graph)
    print("LLama2-7b service deployed to Ray Serve!")
    print("Access the service at: http://localhost:8000/")
    print("Health check: curl http://localhost:8000/-/healthz")
    print("Generate text: curl -X POST http://localhost:8000/generate -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"max_tokens\": 50}'")

# For direct testing without ray serve
if __name__ == "__main__":
    # Check if we're being asked to deploy to Ray Serve
    if len(sys.argv) > 1 and sys.argv[1] == "deploy":
        deploy()
    else:
        # Run as standalone FastAPI app
        import uvicorn
        app = FastAPI()
        
        @app.get("/")
        def redirect_to_docs():
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url="/docs")
        
        # Use the base class directly for standalone mode
        generator = LlamaTextGeneratorBase()
        
        # Mount the FastAPI app at the root
        app.mount("/", generator.app)
        
        uvicorn.run(app, host="0.0.0.0", port=8000)