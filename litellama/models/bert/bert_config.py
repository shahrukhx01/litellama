from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from transformers import BertConfig as HF_BertConfig


@dataclass
class BertConfig:
    """Configuration class for BERT model parameters.

    Args:
        name_or_path (str): Name or path of the pretrained model.
        vocab_size (int): Size of the vocabulary.
        type_vocab_size (int): Number of segment types.
        embed_size (int): Dimension of the embeddings.
        seq_len (int): Maximum sequence length for the input.
        heads (int): Number of attention heads.
        d_model (int): Dimension of the model.
        feed_forward_hidden (int): Hidden layer size of the feedforward network.
        hidden_dropout_prob (float): Dropout probability for hidden layers.
        attention_probs_dropout_prob (float): Dropout probability for attention layers.
        classifier_dropout (float): Dropout probability for the classifier layer.
        pad_token_id (int): Token ID for padding.
        n_layers (int): Number of layers in the encoder.
        layer_norm_eps (float): Epsilon value for layer normalization.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        warmup_steps (int): Number of steps for the learning rate warmup.
        init_weights (bool): Whether to initialize weights or not.
    """

    def __init__(
        self,
        name_or_path: str = "google-bert/bert-base-uncased",
        vocab_size: int = 30522,
        type_vocab_size: int = 3,
        embed_size: int = 768,
        seq_len: int = 512,
        heads: int = 12,
        d_model: int = 768,
        feed_forward_hidden: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        classifier_dropout: Optional[float] = None,
        pad_token_id: int = 0,
        n_layers: int = 12,
        layer_norm_eps: float = 1e-12,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 4000,
        init_weights: bool = False,
    ):
        self.name_or_path = name_or_path
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.heads = heads
        self.d_model = d_model
        self.feed_forward_hidden = feed_forward_hidden
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.classifier_dropout = classifier_dropout
        self.pad_token_id = pad_token_id
        self.n_layers = n_layers
        self.layer_norm_eps = layer_norm_eps
        self.init_weights = init_weights

        # Optimizer parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

    @classmethod
    def load_from_hf_config(cls, model_name_or_path: str) -> "BertConfig":
        """Load a BERT config from the Hugging Face library.

        Args:
            model_name_or_path (str): Name or path of the pretrained model.

        Returns:
            BertConfig: The BERT configuration of custom model with values mapped
            from the Hugging Face model.
        """
        hf_bert_config: HF_BertConfig = HF_BertConfig.from_pretrained(
            model_name_or_path
        )
        return BertConfig(
            name_or_path=model_name_or_path,
            vocab_size=hf_bert_config.vocab_size,
            type_vocab_size=hf_bert_config.type_vocab_size,
            pad_token_id=hf_bert_config.pad_token_id,
            embed_size=hf_bert_config.hidden_size,  # Standard for bert-base
            seq_len=hf_bert_config.max_position_embeddings,
            heads=hf_bert_config.num_attention_heads,
            d_model=hf_bert_config.hidden_size,
            feed_forward_hidden=hf_bert_config.intermediate_size,
            n_layers=hf_bert_config.num_hidden_layers,
            hidden_dropout_prob=hf_bert_config.hidden_dropout_prob,
            attention_probs_dropout_prob=hf_bert_config.attention_probs_dropout_prob,
            classifier_dropout=hf_bert_config.classifier_dropout,
            layer_norm_eps=hf_bert_config.layer_norm_eps,
        )


class BertVariantConfig(Enum):
    """
    Enum class for different Bert model variants.

    Each variant stores a tuple containing the model name and a function to load the configuration.
    """

    BASE_UNCASED = ("google-bert/bert-base-uncased", BertConfig.load_from_hf_config)
    LARGE_UNCASED = ("google-bert/bert-large-uncased", BertConfig.load_from_hf_config)
    TINY_UNCASED = ("prajjwal1/bert-tiny", BertConfig.load_from_hf_config)

    def __init__(self, model_name: str, load_func: Callable):
        """
        Initializes the BertVariantConfig instance.

        Args:
            model_name (str): The model name string.
            load_func (Callable): The function used to load the model configuration.
        """
        self._model_name = model_name
        self._load_func = load_func

    @property
    def value(self) -> BertConfig:
        """
        Lazy-loads the configuration when accessed.

        Returns:
            BertConfig: The loaded Bert configuration object.
        """
        return self._load_func(self._model_name)
