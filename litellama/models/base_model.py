from abc import ABC

import lightning as L
import torch
from tqdm import tqdm

from litellama.samplers.top_k import TopK
from litellama.samplers.top_p import TopP


class BaseModel(L.LightningModule, ABC):
    def __init__(self, name_or_path: str, device: str, max_seq_len: int):
        super().__init__()
        self.name_or_path = name_or_path
        self._device = device
        self.max_seq_len = max_seq_len

    def generate(
        self,
        prompts: list[str],
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        max_tokens: int = 1,
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

        top_k_sampler = TopK()
        top_p_sampler = TopP()

        tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        prompt_tokens = tokenizer.batch_encode_plus(
            prompts, return_tensors="pt", padding=True
        )

        pad_id = tokenizer.all_special_ids[-1]
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens["input_ids"])
        total_len = min(self.max_seq_len, max_tokens + max_prompt_len)
        batch_size = prompt_tokens["input_ids"].shape[0]
        tokens = torch.full(
            (batch_size, total_len), pad_id, dtype=torch.long, device=self._device
        )
        for k, t in enumerate(prompt_tokens["input_ids"]):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self._device)

        eos_reached = torch.tensor([False] * batch_size, device=self._device)
        prompt_tokens_mask = (
            tokens != pad_id
        )  # True if the token is a prompt token, False otherwise
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")

        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = self.forward(tokens[:, cur_pos - 1 : cur_pos], cur_pos)
            if temperature > 0:
                # The temperature is applied before the softmax
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                if top_k > 0:
                    next_token = top_k_sampler.sample(probs, top_k)
                elif top_p > 0:
                    next_token = top_p_sampler.sample(probs, top_p)
            else:
                # Greedily select the token with the max probability
                next_token = torch.argmax(logits[:, -1], dim=-1)
            print(f"Next token: {next_token}")
            next_token = next_token.reshape(-1)
            # Only replace token if it is a padding token
            next_token = torch.where(
                prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (
                next_token == tokenizer.eos_token_id
            )
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []
        print(tokens)
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            if tokenizer.eos_token_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(tokenizer.eos_token_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            print(current_prompt_tokens)
            out_text.append(tokenizer.decode(current_prompt_tokens))
        print(out_text)
        return out_text
