from abc import ABC
from typing import Optional

import lightning as L


class BaseModel(L.LightningModule, ABC):
    def __init__(self, name_or_path: str, device: str):
        super().__init__()
        self.name_or_path = name_or_path
        self.device = device

    def generate(
        self,
        prompts: list[str],
        temperature: float = 0.6,
        top_p: float = 0.0,
        top_k: int = 0,
        max_tokens: Optional[int] = None,
    ) -> list[str]:
        """
        Generate text from prompts.

        Args:
            prompts (list[str]): List of prompts.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling ratio.
            max_tokens (Optional[int]): Maximum number of tokens to generate.

        Returns:
            list[str]: List of generated texts.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        prompt_tokens = tokenizer.batch_encode_plus(
            prompts, return_tensors="pt", padding=True
        )

        print(prompt_tokens["input_ids"].shape)
        return []
