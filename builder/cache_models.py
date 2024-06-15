import os

import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


def fetch_pretrained_model(model_name, hf_token=None):
    """
    Fetches a pretrained model from the HuggingFace model hub.
    """
    max_retries = 3

    for attempt in range(max_retries):
        try:
            PaliGemmaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                revision="bfloat16",
                token=hf_token,
            ).eval()

            AutoProcessor.from_pretrained(model_name, token=hf_token)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise


def warm_up_pipeline():
    """
    Fetches the pipelines from the HuggingFace model hub.
    """

    hf_token = os.environ.get("HF_TOKEN", None)
    fetch_pretrained_model("google/paligemma-3b-mix-448", hf_token)


if __name__ == "__main__":
    warm_up_pipeline()
