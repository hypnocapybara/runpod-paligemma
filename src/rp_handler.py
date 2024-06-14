"""
Contains the handler function that will be called by the serverless.
"""
import os
import torch
import runpod
import concurrent.futures

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from diffusers.utils import load_image

from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA

torch.cuda.empty_cache()

# ------------------------------- Model Handler ------------------------------ #


class ModelHandler:
    def __init__(self):
        self.model = None
        self.processor = None
        self.load_models()

    def _do_load_model(self):
        model_id = "google/paligemma-3b-mix-448"
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            revision="bfloat16",
            token=os.environ.get("HF_TOKEN", None),
        ).eval()

        processor = AutoProcessor.from_pretrained(model_id, token=os.environ.get("HF_TOKEN", None))

        return model, processor

    def load_models(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_model = executor.submit(self._do_load_model)

            self.model, self.processor = future_model.result()


MODELS = ModelHandler()

# ---------------------------------- Helper ---------------------------------- #


@torch.inference_mode()
def handler(job):
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}

    job_input = validated_input['validated_input']

    prompt = job_input["prompt"]
    max_new_tokens = job_input["max_new_tokens"]
    image_url = job_input["image_url"]

    init_image = load_image(image_url).convert("RGB")
    model_inputs = MODELS.processor(text=prompt, images=init_image, return_tensors="pt").to(MODELS.model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = MODELS.model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generation = generation[0][input_len:]
        decoded = MODELS.processor.decode(generation, skip_special_tokens=True)

    return {
        "output": decoded,
        "refresh_worker": True
    }


runpod.serverless.start({"handler": handler})
