import math
from dataclasses import dataclass
from typing import Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from transformers import BertForMaskedLM

from crammers.optimizers.scheduled_optimizer import ScheduledOptim


@dataclass
class BertConfig:
    """Configuration class for BERT model parameters.

    Args:
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


class BertEmbedding(torch.nn.Module):
    """BERT embedding layer that generates token, positional, and segment embeddings.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.embed_size = config.embed_size
        self.token = nn.Embedding(
            config.vocab_size, config.embed_size, padding_idx=config.pad_token_id
        )
        self.segment = nn.Embedding(config.type_vocab_size, config.embed_size)
        self.position = nn.Embedding(config.seq_len, config.embed_size)
        self.layer_norm = nn.LayerNorm(config.embed_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids",
            torch.arange(config.seq_len).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self, input_ids: torch.Tensor, token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """Generates embeddings by adding token, positional, and segment embeddings.

        Args:
            input_ids (torch.Tensor): Tensor of input token indices.
            token_type_ids (torch.Tensor): Tensor of segment indices.

        Returns:
            torch.Tensor: Output embeddings after adding token, position, and segment embeddings.
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).expand(
            (input_ids.size(0), -1)
        )
        embeddings = (
            self.token(input_ids)
            + self.position(position_ids)
            + self.segment(token_type_ids)
        )
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadedAttentionOutput(nn.Module):
    """Output layer for multi-headed attention.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super(MultiHeadedAttentionOutput, self).__init__()
        self.dense = torch.nn.Linear(config.d_model, config.d_model)
        self.layer_norm = torch.nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.out_dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, embeddings: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(context)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + embeddings)
        return hidden_states


class MultiHeadedAttention(nn.Module):
    """Multi-head attention layer.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super(MultiHeadedAttention, self).__init__()
        assert config.d_model % config.heads == 0
        self.d_k = config.d_model // config.heads
        self.heads = config.heads
        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)

        self.query = torch.nn.Linear(config.d_model, config.d_model)
        self.key = torch.nn.Linear(config.d_model, config.d_model)
        self.value = torch.nn.Linear(config.d_model, config.d_model)

        self.output = MultiHeadedAttentionOutput(config)

    def forward(
        self,
        embeddings: torch.Tensor,
    ):
        """Forward pass for multi-headed attention layer.

        Args:
            embeddings (torch.Tensor): Embeddings tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        query = self.query(embeddings)
        key = self.key(embeddings)
        value = self.value(embeddings)

        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)

        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(
            query.size(-1)
        )

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        context = torch.matmul(weights, value)

        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(context.shape[0], -1, self.heads * self.d_k)
        )
        # pass the contextualized embeddings through the output layer
        hidden_states = self.output(embeddings, context)
        return hidden_states


class FFN(nn.Module):
    """Feed-forward network for the BERT encoder layer.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super(FFN, self).__init__()
        self.intermediate_dim = config.feed_forward_hidden
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act_fn = nn.GELU()
        self.fc1 = nn.Linear(config.d_model, self.intermediate_dim)
        self.fc2 = nn.Linear(self.intermediate_dim, config.d_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feed-forward network.

        Args:
            hidden_states (torch.Tensor): Input tensor representing activations from previous layers.

        Returns:
            torch.Tensor: Output tensor after feed-forward operations.
        """
        hidden_states = self.act_fn(self.fc1(hidden_states))
        hidden_states = self.dropout(self.fc2(hidden_states))
        return hidden_states


class BertEncoderLayer(nn.Module):
    """A single encoder layer in the BERT architecture.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super(BertEncoderLayer, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attention = MultiHeadedAttention(config)
        self.feed_forward = FFN(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single encoder layer.

        Args:
            embeddings (torch.Tensor): Input embeddings.

        Returns:
            torch.Tensor: Output embeddings after applying multi-head attention and feed-forward network.
        """
        interacted = self.attention(embeddings)
        feed_forward_out = self.feed_forward(interacted)
        return self.layer_norm(feed_forward_out + interacted)


class Bert(nn.Module):
    """Vanilla BERT model implementation with multiple transformer layers.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super(Bert, self).__init__()
        self.n_layers = config.n_layers
        self.embeddings = BertEmbedding(config)
        self.encoder_blocks = torch.nn.ModuleList(
            [BertEncoderLayer(config) for _ in range(config.n_layers)]
        )

    def forward(self, input_ids, token_type_ids):
        """Forward pass for the entire BERT model.

        Args:
            input_ids (torch.Tensor): Input token indices.
            token_type_ids (torch.Tensor): Segment labels (e.g., sentence A/B).

        Returns:
            torch.Tensor: Output embeddings after passing through all encoder layers.
        """
        embeddings = self.embeddings(input_ids, token_type_ids)

        for encoder in self.encoder_blocks:
            embeddings = encoder(embeddings)
        return embeddings


class BertPredictionHeadTransform(nn.Module):
    """Transforms hidden states to another latent space followed by non-linarity and layer normalization.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.transform_act_fn = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class MaskedLanguageModel(nn.Module):
    """Masked Language Model that predicts the original token from a masked input sequence.

    Args:
        hidden (int): BERT model output size.
        vocab_size (int): Size of the vocabulary.
    """

    def __init__(
        self, config: BertConfig, embedding_weights: torch.nn.Parameter = None
    ):
        super(MaskedLanguageModel, self).__init__()

        self.transform = BertPredictionHeadTransform(config)
        # the weights of the decoder layer are tied to the input embeddings
        # furthermore, as embeddings dont have bias, the decoder layer does not have bias
        self.decoder = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)

        if embedding_weights is not None:
            self.decoder.weight = embedding_weights

        # as the weights are tied, the bias is added separately
        self.bias = torch.nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for masked language model.

        Args:
            hidden_states (torch.Tensor): Input tensor with masked tokens.

        Returns:
            torch.Tensor: Output logits after applying linear transformation and softmax.
        """
        hidden_states = self.transform(hidden_states)
        logits = self.decoder(hidden_states)
        return logits


class BertMaskedLM(L.LightningModule):
    """BERT Language Model for pretraining: Masked Language Model

    Args:
        config (BertConfig): Configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.encoder = Bert(config)
        self.mask_lm = MaskedLanguageModel(
            config, embedding_weights=self.encoder.embeddings.token.weight
        )
        if config.init_weights:
            self._init_embedding_weights()
            self.apply(self.init_model_weights)

    def _init_embedding_weights(self) -> None:
        """Initialize the weights for the embedding layers"""

        # initialize position embeddings with a larger standard deviation
        nn.init.normal_(
            self.encoder.embeddings.position.weight,
            mean=0.0,
            std=self.config.embed_size**-0.5,
        )

        # initialize segment/token type embeddings
        nn.init.normal_(self.encoder.embeddings.segment.weight, mean=0.0, std=0.02)

        # Log model parameter count
        logger.info(
            f"Initialized BERT model with {sum(p.numel() for p in self.parameters())} parameters"
        )

    @staticmethod
    def init_model_weights(module: nn.Module) -> None:
        """Initialize the model weights according to the BERT paper.

        Linear layers are initialized with truncated normal distribution.
        Layer normalization is initialized with ones for weight and zeros for bias.
        Embedding layers are initialized with normal distribution.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Initialize Linear and Embedding weight
            module.weight.data.normal_(mean=0.0, std=0.02)

            # Initialize Linear bias
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self, input_ids: torch.Tensor, token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for the entire BERT model.

        Args:
            input_ids (torch.Tensor): Input token indices.
            token_type_ids (torch.Tensor): Segment labels (e.g., sentence A/B).

        Returns:
            torch.Tensor: Output embeddings after passing through all encoder layers.
        """
        hidden_states = self.encoder(input_ids, token_type_ids)
        return self.mask_lm(hidden_states)

    def training_step(
        self, input_ids: torch.Tensor, token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """Training step to compute the loss for and Masked LM task.

        Args:
            input_ids (torch.Tensor): Input token indices.
            token_type_ids (torch.Tensor): Segment indices.

        Returns:
            torch.Tensor: Loss value for the Masked LM task.
        """
        embeddings = self(input_ids, token_type_ids)
        return self.mask_lm(embeddings)

    def configure_optimizers(self):
        """Configure optimizers and learning rate scheduler for training.

        Returns:
            ScheduledOptim: Optimizer wrapped with learning rate scheduling.
        """
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        return ScheduledOptim(optimizer, self.config.d_model, self.config.warmup_steps)

    def load_pretrained_bert(self, model_name: str = "bert-base-uncased"):
        # sourcery skip: extract-method
        """
        Load pretrained weights from HuggingFace Transformers BERT model into custom BERT implementation.

        Args:
            model_name (str): Name of pretrained model to load from HuggingFace

        Returns:
            BERTLM: Custom model with loaded pretrained weights
            None: If transformers is not installed
        """
        try:
            # load pretrained model
            pretrained: BertForMaskedLM = BertForMaskedLM.from_pretrained(model_name)

            # map embeddings
            self.encoder.embeddings.token.weight.data = (
                pretrained.bert.embeddings.word_embeddings.weight.data
            )
            self.encoder.embeddings.position.weight.data = (
                pretrained.bert.embeddings.position_embeddings.weight.data
            )
            self.encoder.embeddings.segment.weight.data = (
                pretrained.bert.embeddings.token_type_embeddings.weight.data
            )
            self.encoder.embeddings.layer_norm.weight.data = (
                pretrained.bert.embeddings.LayerNorm.weight.data
            )
            self.encoder.embeddings.layer_norm.bias.data = (
                pretrained.bert.embeddings.LayerNorm.bias.data
            )
            # map encoder layers
            for custom_layer, pretrained_layer in zip(
                self.encoder.encoder_blocks, pretrained.bert.encoder.layer
            ):
                # self attention weights
                custom_layer.attention.query.weight.data = (
                    pretrained_layer.attention.self.query.weight.data
                )
                custom_layer.attention.query.bias.data = (
                    pretrained_layer.attention.self.query.bias.data
                )
                custom_layer.attention.key.weight.data = (
                    pretrained_layer.attention.self.key.weight.data
                )
                custom_layer.attention.key.bias.data = (
                    pretrained_layer.attention.self.key.bias.data
                )
                custom_layer.attention.value.weight.data = (
                    pretrained_layer.attention.self.value.weight.data
                )
                custom_layer.attention.value.bias.data = (
                    pretrained_layer.attention.self.value.bias.data
                )
                custom_layer.attention.output.dense.weight.data = (
                    pretrained_layer.attention.output.dense.weight.data
                )
                custom_layer.attention.output.dense.bias.data = (
                    pretrained_layer.attention.output.dense.bias.data
                )
                custom_layer.attention.output.layer_norm.weight.data = (
                    pretrained_layer.attention.output.LayerNorm.weight.data
                )
                custom_layer.attention.output.layer_norm.bias.data = (
                    pretrained_layer.attention.output.LayerNorm.bias.data
                )

                # layer norm weights
                custom_layer.layer_norm.weight.data = (
                    pretrained_layer.output.LayerNorm.weight.data
                )
                custom_layer.layer_norm.bias.data = (
                    pretrained_layer.output.LayerNorm.bias.data
                )

                # feed forward weights
                custom_layer.feed_forward.fc1.weight.data = (
                    pretrained_layer.intermediate.dense.weight.data
                )
                custom_layer.feed_forward.fc1.bias.data = (
                    pretrained_layer.intermediate.dense.bias.data
                )
                custom_layer.feed_forward.fc2.weight.data = (
                    pretrained_layer.output.dense.weight.data
                )
                custom_layer.feed_forward.fc2.bias.data = (
                    pretrained_layer.output.dense.bias.data
                )

            # load MLM transform weights
            self.mask_lm.transform.dense.weight.data = (
                pretrained.cls.predictions.transform.dense.weight.data
            )
            self.mask_lm.transform.dense.bias.data = (
                pretrained.cls.predictions.transform.dense.bias.data
            )
            self.mask_lm.transform.layer_norm.weight.data = (
                pretrained.cls.predictions.transform.LayerNorm.weight.data
            )
            self.mask_lm.transform.layer_norm.bias.data = (
                pretrained.cls.predictions.transform.LayerNorm.bias.data
            )

            # load MLM bias weights
            self.mask_lm.decoder.weight.data = (
                pretrained.cls.predictions.decoder.weight.data
            )
            self.mask_lm.bias.data = pretrained.cls.predictions.bias.data
            logger.info(
                f"Successfully loaded weights from pretrained model: {model_name}"
            )
            return self

        except Exception as e:
            logger.error(f"Error loading pretrained model: {str(e)}")
            return None
