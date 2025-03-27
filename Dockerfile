FROM anyscale/ray-llm:2.44.0-py311-cu124

# Install required packages
RUN pip install accelerate transformers deepspeed datasets tensorboard huggingface_hub peft

# Clone Ray repository
RUN git clone https://github.com/ray-project/ray

# Set environment variable in the image
ENV ANYSCALE_ARTIFACT_STORAGE="/shared"
