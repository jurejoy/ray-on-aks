FROM anyscale/ray-llm:2.44.0-py311-cu124

# Install required packages
RUN pip install accelerate transformers deepspeed datasets tensorboard huggingface_hub peft
RUN pip install ray\[serve\] requests torch diffusers==0.12.1 transformers
RUN pip install -U diffusers transformers huggingface_hub

# Clone Ray repository
RUN git clone https://github.com/ray-project/ray

# Set environment variable in the image
ENV ANYSCALE_ARTIFACT_STORAGE="/shared"