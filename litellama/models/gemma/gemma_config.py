from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from transformers import GemmaConfig as HF_GemmaConfig


@dataclass
class GemmaConfig:
    """
    Configuration class for the Gemma model.

    Args:
        name_or_path (str): Model name or path.
        rope_theta (float): Rotary positional embedding theta value.
        vocab_size (int): Vocabulary size.
        type_vocab_size (int): Type vocabulary size.
        hidden_size (int): Hidden layer size.
        num_attention_heads (int): Number of attention heads.
        num_key_value_heads (Optional[int]): Number of key-value heads.
        feed_forward_hidden (int): Feed-forward layer size.
        intermediate_size (int): Intermediate layer size.
        classifier_dropout (Optional[float]): Dropout rate for classifier.
        pad_token_id (int): Padding token ID.
        max_position_embeddings (int): Maximum position embeddings.
        num_hidden_layers (int): Number of hidden layers.
        rms_norm_eps (float): Epsilon value for RMS normalization.
        device (str): Device to run the model on.
        batch_size (int): Batch size.
    """

    def __init__(
        self,
        name_or_path: str,
        rope_theta: float = 10000.0,
        vocab_size: int = -1,
        type_vocab_size: int = 3,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_key_value_heads: Optional[int] = None,
        feed_forward_hidden: int = 3072,
        intermediate_size: int = 11008,
        classifier_dropout: Optional[float] = None,
        pad_token_id: int = 0,
        max_position_embeddings: int = 4096,
        num_hidden_layers: int = 12,
        rms_norm_eps: float = 1e-12,
        device: str = "cpu",
        batch_size: int = 1,
    ):
        self.name_or_path = name_or_path
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_key_value_heads
            if num_key_value_heads is not None
            else num_attention_heads
        )
        self.intermediate_size = intermediate_size
        self.feed_forward_hidden = feed_forward_hidden
        self.classifier_dropout = classifier_dropout
        self.pad_token_id = pad_token_id
        self.num_hidden_layers = num_hidden_layers
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.batch_size = batch_size
        self.rope_theta = rope_theta

        self.device = device

    @classmethod
    def load_from_hf_config(
        cls,
        model_name: str,
    ) -> "GemmaConfig":
        """
        Load a Gemma config from the Hugging Face library.

        Args:
            model_name (str): The name of the Gemma model to load.

        Returns:
            GemmaConfig: The Gemma configuration of the custom model with values mapped
            from the Hugging Face model.
        """
        hf_gemma_config: HF_GemmaConfig = HF_GemmaConfig.from_pretrained(model_name)
        return GemmaConfig(
            name_or_path=model_name,
            rope_theta=hf_gemma_config.rope_theta,
            vocab_size=hf_gemma_config.vocab_size,
            pad_token_id=hf_gemma_config.pad_token_id,
            hidden_size=hf_gemma_config.hidden_size,
            max_position_embeddings=hf_gemma_config.max_position_embeddings,
            num_attention_heads=hf_gemma_config.num_attention_heads,
            num_key_value_heads=hf_gemma_config.num_key_value_heads,
            num_hidden_layers=hf_gemma_config.num_hidden_layers,
            rms_norm_eps=hf_gemma_config.rms_norm_eps,
            intermediate_size=hf_gemma_config.intermediate_size,
        )


class GemmaVariantConfig(Enum):
    """
    Enum class for different Gemma model variants.

    Each variant stores a tuple containing the model name and a function to load the configuration.
    """

    GEMMA_1_1_7B = ("google/gemma-1.1-7b-it", GemmaConfig.load_from_hf_config)
    GEMMA_1_1_2B = ("google/gemma-1.1-2b-it", GemmaConfig.load_from_hf_config)

    def __init__(self, model_name: str, load_func: Callable):
        """
        Initializes the GemmaVariantConfig instance.

        Args:
            model_name (str): The model name string.
            load_func (Callable): The function used to load the model configuration.
        """
        self._model_name = model_name
        self._load_func = load_func

    @property
    def value(self) -> GemmaConfig:
        """
        Lazy-loads the configuration when accessed.

        Returns:
            GemmaConfig: The loaded Gemma configuration object.
        """
        return self._load_func(self._model_name)
