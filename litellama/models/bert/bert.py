import math

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger

from litellama.models.bert.bert_config import BertConfig
from litellama.optimizers.scheduled_optimizer import ScheduledOptim


class BertEmbedding(torch.nn.Module):
    """BERT embedding layer that generates token, positional, and segment embeddings.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.embed_size = config.embed_size
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.embed_size, padding_idx=config.pad_token_id
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.embed_size
        )
        self.position_embeddings = nn.Embedding(config.seq_len, config.embed_size)
        self.LayerNorm = nn.LayerNorm(config.embed_size, eps=config.layer_norm_eps)
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
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        embeddings = self.LayerNorm(embeddings)
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
        self.LayerNorm = torch.nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.out_dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, embeddings: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(context)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + embeddings)
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

    def forward(
        self,
        embeddings: torch.Tensor,
    ):  # sourcery skip: inline-immediately-returned-variable
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
        return context


class BertAttention(nn.Module):
    """Bert attention layer that applies multi-headed attention and output layer.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super(BertAttention, self).__init__()
        self.self = MultiHeadedAttention(config)
        self.output = MultiHeadedAttentionOutput(config)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        hidden_states = self.self(embeddings)
        return self.output(embeddings, hidden_states)


class BertOutput(nn.Module):
    """Output layer for the feed-forward network.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.feed_forward_hidden, config.d_model)
        self.LayerNorm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for the output layer.

        Args:
            hidden_states (torch.Tensor): Hidden states from the feed-forward network.
            input_tensor (torch.Tensor): Input tensor to the feed-forward network.

        Returns:
            torch.Tensor: Output tensor after applying feed-forward network.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):
    """Intermediate layer for the feed-forward network.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.feed_forward_hidden)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for the intermediate layer.

        Args:
            hidden_states (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying feed-forward network.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertEncoderLayer(nn.Module):
    """A single encoder layer in the BERT architecture.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super(BertEncoderLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single encoder layer.

        Args:
            embeddings (torch.Tensor): Input embeddings.

        Returns:
            torch.Tensor: Output embeddings after applying multi-head attention and feed-forward network.
        """
        interacted = self.attention(embeddings)
        intermediate = self.intermediate(interacted)
        feed_forward_out = self.output(intermediate, interacted)
        return feed_forward_out


class BertEncoderBlock(nn.Module):
    def __init__(self, config: BertConfig):
        super(BertEncoderBlock, self).__init__()
        self.n_layers = config.n_layers
        self.layer = torch.nn.ModuleList(
            [BertEncoderLayer(config) for _ in range(config.n_layers)]
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass for the entire BERT model.

        Args:
            embeddings (torch.Tensor): Input embeddings.

        Returns:
            torch.Tensor: Output embeddings after passing through all encoder layers.
        """
        hidden_states = embeddings
        for encoder in self.layer:
            hidden_states = encoder(hidden_states)
        return hidden_states


class Bert(nn.Module):
    """Vanilla BERT model implementation with multiple transformer layers.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super(Bert, self).__init__()
        self.n_layers = config.n_layers
        self.embeddings = BertEmbedding(config)
        self.encoder = BertEncoderBlock(config)

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
        embeddings = self.embeddings(input_ids, token_type_ids)
        hidden_states = self.encoder(embeddings)
        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    """Transforms hidden states to another latent space followed by non-linarity and layer normalization.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    """BERT Language Model head that predicts masked tokens.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    """BERT Masked Language Model head that predicts masked tokens.

    Args:
        config (BertConfig): BERT configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Masked LM head.

        Args:
            sequence_output (torch.Tensor): Output embeddings from the BERT model.

        Returns:
            torch.Tensor: Predicted token logits.
        """
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertMaskedLM(L.LightningModule):
    """BERT Language Model for pretraining: Masked Language Model

    Args:
        config (BertConfig): Configuration object that contains model parameters.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self._config = config
        self.bert = Bert(config)
        self.cls = BertOnlyMLMHead(config)

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
        hidden_states = self.bert(input_ids, token_type_ids)
        return self.cls(hidden_states)

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
        return self.cls(embeddings)

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

    def load_pretrained_hf(self):
        # sourcery skip: extract-method
        """Load pretrained weights from a Hugging Face model checkpoint."""
        # load pretrained model
        import time

        from transformers import BertForMaskedLM

        load_start_ts = time.perf_counter()
        pretrained: BertForMaskedLM = BertForMaskedLM.from_pretrained(
            self._config.name_or_path
        )
        self.load_state_dict(pretrained.state_dict(), strict=True)
        load_duration = time.perf_counter() - load_start_ts
        logger.info(
            f"Successfully loaded weights from pretrained model: {self._config.name_or_path} in {load_duration:.2f}"
            " seconds."
        )
        return self


if __name__ == "__main__":
    from litellama.models.bert.bert_config import BertVariantConfig

    bert = BertMaskedLM(BertVariantConfig.TINY_UNCASED.value)
    bert.load_pretrained_hf()
