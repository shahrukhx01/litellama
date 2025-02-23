import pytest
from transformers import AutoTokenizer, BertTokenizer
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

from litellama.models import LLaMACausalLM  # isort:skip
from litellama.models import BertMaskedLM, LLaMAVariantConfig  # isort:skip

from litellama.models import BertVariantConfig  # isort:skip


@pytest.fixture(scope="session")
def bert_model():
    bert_config = BertVariantConfig.BASE_UNCASED.value
    yield BertMaskedLM(config=bert_config)


@pytest.fixture(scope="session")
def bert_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="session")
def llama_model():
    llama_config = LLaMAVariantConfig.LLAMA_3_2_1B.value
    yield LLaMACausalLM(config=llama_config)


@pytest.fixture(scope="session")
def llama_config():
    yield LLaMAVariantConfig.LLAMA_3_2_1B.value


@pytest.fixture(scope="session")
def hf_llama_model_and_tokenizer():
    """Fixture providing HuggingFace model and tokenizer."""
    model_name = LLaMAVariantConfig.LLAMA_3_2_1B.value.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = HF_LlamaForCausalLM.from_pretrained(model_name)
    return model, tokenizer
