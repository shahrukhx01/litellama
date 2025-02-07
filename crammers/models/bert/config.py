from enum import Enum

from transformers import BertConfig as HF_BertConfig
from transformers import BertTokenizer

from crammers.models.bert.bert import BertConfig


def load_bert_config(model_name: str) -> BertConfig:
    """Load a BERT config from the Hugging Face library.

    Args:
        model_name (str): The name of the BERT model to load.

    Returns:
        BertConfig: The BERT configuration of custom model with values mapped
        from the Hugging Face model.
    """
    hf_bert_tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)
    hf_bert_config: HF_BertConfig = HF_BertConfig.from_pretrained(model_name)
    return BertConfig(
        vocab_size=hf_bert_tokenizer.vocab_size,
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
    BASE_UNCASED = load_bert_config("bert-base-uncased")
    LARGE_UNCASED = load_bert_config("bert-large-uncased")
    TINY_UNCASED = load_bert_config("prajjwal1/bert-tiny")
