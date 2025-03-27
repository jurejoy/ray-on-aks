import os
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse  # Add this import
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams

# Get HuggingFace token from environment
hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not hf_token:
    raise ValueError("HUGGING_FACE_HUB_TOKEN environment variable is not set")

# Initialize FastAPI app
app = FastAPI(title="Llama2-7b API", description="API for generating text with Llama2-7b")

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

# Initialize the model
model_id = "meta-llama/Llama-2-7b-hf"
llm = None

@app.on_event("startup")
async def startup_event():
    global llm
    print(f"Loading model {model_id}...")
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        download_dir=None,
                dtype="float16",  # Use half precision for efficiency
        gpu_memory_utilization=0.85,
    )
    print("Model loaded successfully")

@app.on_event("shutdown")
async def shutdown_event():
    # Properly clean up resources
    global llm
    if llm is not None:
        del llm
    
@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text based on the input prompt"""
    global llm
    if llm is None:
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
        outputs = llm.generate([request.prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        return GenerationResponse(generated_text=generated_text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global llm
    return {"status": "ok", "model_loaded": llm is not None}

# Add the standard Kubernetes health check endpoint
from fastapi.responses import PlainTextResponse

@app.get("/-/healthz", response_class=PlainTextResponse)
async def kubernetes_health_check():
    """Standard health check endpoint for Kubernetes"""
    return "success"

# Changed how the app is run
def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()