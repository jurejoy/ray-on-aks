import os
import ray
from ray import serve
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
from vllm import LLM, SamplingParams
import traceback
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger("llama-service")

# Initialize Ray
ray.init(address="auto")

# Define models
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt for text generation")
    max_tokens: int = Field(512, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Temperature for sampling")
    top_p: float = Field(0.95, description="Top-p sampling parameter")
    top_k: int = Field(50, description="Top-k sampling parameter")
    stop_sequences: Optional[List[str]] = Field(None, description="Sequences that stop generation")
    
class GenerationResponse(BaseModel):
    generated_text: str

# Create a FastAPI app without routes - we'll add them in the deployment
app = FastAPI()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Server error: {str(exc)}"}
    )

# Define deployment
@serve.deployment(
    name="llama2-7b-service",
    num_replicas=2,  # 2 independent replicas 
    ray_actor_options={"num_gpus": 1}  # Each gets 1 GPU
)
@serve.ingress(app)  # Key change: use serve.ingress
class LlamaTextGenerator:
    def __init__(self):
        # Initialize model
        self.model_id = "meta-llama/Llama-2-7b-hf"
        self.llm = None
        logger.info(f"Initializing LlamaTextGenerator with model {self.model_id}")
        
        # Get HuggingFace token
        self.hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not self.hf_token:
            logger.warning("HUGGING_FACE_HUB_TOKEN environment variable is not set")
            
        # Load model
        logger.info(f"Loading model {self.model_id}...")
        try:
            self.llm = LLM(
                model=self.model_id,
                trust_remote_code=True,
                download_dir=None, 
                dtype="float16",
                gpu_memory_utilization=0.85,
                tensor_parallel_size=1  # Use only 1 GPU per replica
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}\n{traceback.format_exc()}")
    
    @app.post("/generate", response_model=GenerationResponse)
    async def generate(self, request: GenerationRequest):
        logger.info(f"Received request: {request.prompt[:30]}...")
        if self.llm is None:
            logger.error("Model not loaded")
            raise HTTPException(status_code=503, detail="Model not loaded yet")
            
        try:
            # Create sampling parameters
            sampling_params = SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stop=request.stop_sequences,
            )
            
            # Generate text
            logger.info("Generating text...")
            outputs = self.llm.generate([request.prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            logger.info(f"Generated text (first 30 chars): {generated_text[:30]}...")
            
            return GenerationResponse(generated_text=generated_text)
        
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=error_msg)

    @app.get("/health")
    async def health(self):
        return {"status": "ok", "model_loaded": self.llm is not None}

    @app.get("/-/healthz", response_class=PlainTextResponse)
    async def healthz(self):
        logger.info("Health check endpoint hit")
        return "success"

# Deploy
serve.run(LlamaTextGenerator.bind())