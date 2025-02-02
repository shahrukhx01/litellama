import pytest
from transformers import BertTokenizer

from crammers.models import BertMaskedLM, BertVariantConfig


@pytest.fixture(scope="class")
def bert_model():
    bert_config = BertVariantConfig.BASE_UNCASED.value
    yield BertMaskedLM(config=bert_config)


@pytest.fixture(scope="class")
def bert_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")
