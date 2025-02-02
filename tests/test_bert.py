import pytest
import torch
from transformers import BertForMaskedLM as HF_BertForMaskedLM
from transformers import BertTokenizer

from crammers.models import BertMaskedLM


class TestBertMaskedLM:
    """Test suite for BertMaskedLM model implementation.

    Tests various aspects of the BERT model including shape validation,
    weight loading, HuggingFace compatibility, and core functionalities.
    """

    def test_model_output_shape(
        self, bert_model: BertMaskedLM, bert_tokenizer: BertTokenizer
    ) -> None:
        """Test if model output shape matches expected dimensions.

        Args:
            bert_model: Custom BERT model implementation
            bert_tokenizer: BERT tokenizer instance

        Validates that the output tensor shape matches [batch_size, sequence_length, vocab_size].
        """
        text = "hello world"
        tokens = bert_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        filtered_tokens = {
            k: v for k, v in tokens.items() if k in ["input_ids", "token_type_ids"]
        }

        output = bert_model(**filtered_tokens)

        batch_size, seq_len = filtered_tokens["input_ids"].shape
        assert output.shape == (batch_size, seq_len, bert_model.config.vocab_size)

    def test_pretrained_weights_loading(self, bert_model: BertMaskedLM) -> None:
        """Test loading of pretrained weights.

        Args:
            bert_model: Custom BERT model implementation

        Verifies that pretrained weights are properly loaded and not zero-initialized.
        """
        model_name = "bert-base-uncased"
        loaded_model = bert_model.load_pretrained_bert(model_name)

        assert isinstance(loaded_model, type(bert_model))
        assert not torch.allclose(
            loaded_model.encoder.embeddings.token.weight,
            torch.zeros_like(loaded_model.encoder.embeddings.token.weight),
        )

    def test_huggingface_compatibility(
        self, bert_model: BertMaskedLM, bert_tokenizer: BertTokenizer
    ) -> None:
        """Test compatibility with HuggingFace implementation.

        Args:
            bert_model: Custom BERT model implementation
            bert_tokenizer: BERT tokenizer instance

        Compares outputs between custom implementation and HuggingFace's implementation.
        """
        model_name = "bert-base-uncased"
        text = "testing bert model"

        bert_model = bert_model.load_pretrained_bert(model_name).eval()
        hf_model = HF_BertForMaskedLM.from_pretrained(model_name).eval()

        tokens = bert_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        filtered_tokens = {
            k: v for k, v in tokens.items() if k in ["input_ids", "token_type_ids"]
        }

        with torch.no_grad():
            custom_output = bert_model(**filtered_tokens)
            hf_output = hf_model(**filtered_tokens).logits

        assert torch.allclose(custom_output, hf_output, atol=1e-4)

    def test_masking_prediction(
        self, bert_model: BertMaskedLM, bert_tokenizer: BertTokenizer
    ) -> None:
        """Test masked token prediction functionality.

        Args:
            bert_model: Custom BERT model implementation
            bert_tokenizer: BERT tokenizer instance

        Validates that the model can predict masked tokens with meaningful words.
        """
        model_name = "bert-base-uncased"
        bert_model = bert_model.load_pretrained_bert(model_name).eval()

        text = "The cat [MASK] on the mat."
        tokens = bert_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        filtered_tokens = {
            k: v for k, v in tokens.items() if k in ["input_ids", "token_type_ids"]
        }

        with torch.no_grad():
            output = bert_model(**filtered_tokens)

        mask_idx = (
            filtered_tokens["input_ids"] == bert_tokenizer.mask_token_id
        ).nonzero()
        predicted_token_id = output[0, mask_idx[0, 1]].argmax().item()
        predicted_word = bert_tokenizer.decode([predicted_token_id])

        assert isinstance(predicted_word, str)
        assert len(predicted_word.strip()) > 0

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batch_processing(
        self, bert_model: BertMaskedLM, bert_tokenizer: BertTokenizer, batch_size: int
    ) -> None:
        """Test model's ability to handle different batch sizes.

        Args:
            bert_model: Custom BERT model implementation
            bert_tokenizer: BERT tokenizer instance
            batch_size: Number of sequences to process in parallel

        Validates that the model can process different batch sizes correctly.
        """
        texts = ["hello world"] * batch_size
        tokens = bert_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        filtered_tokens = {
            k: v for k, v in tokens.items() if k in ["input_ids", "token_type_ids"]
        }

        output = bert_model(**filtered_tokens)
        assert output.shape[0] == batch_size

    def test_gradient_flow(
        self, bert_model: BertMaskedLM, bert_tokenizer: BertTokenizer
    ) -> None:
        """Test gradient computation and backpropagation.

        Args:
            bert_model: Custom BERT model implementation
            bert_tokenizer: BERT tokenizer instance

        Verifies that gradients are properly computed during backpropagation.
        """
        text = "testing gradient flow"
        tokens = bert_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        filtered_tokens = {
            k: v for k, v in tokens.items() if k in ["input_ids", "token_type_ids"]
        }

        bert_model.train()
        output = bert_model(**filtered_tokens)
        loss = output.mean()
        loss.backward()

        assert bert_model.encoder.embeddings.token.weight.grad is not None

    def test_input_validation(
        self, bert_model: BertMaskedLM, bert_tokenizer: BertTokenizer
    ) -> None:
        """Test model's input validation.

        Args:
            bert_model: Custom BERT model implementation
            bert_tokenizer: BERT tokenizer instance

        Validates that the model properly handles invalid inputs.
        """
        with pytest.raises(IndexError):
            bert_model(
                input_ids=torch.tensor([], dtype=torch.long),
                token_type_ids=torch.tensor([], dtype=torch.long),
            )

    def test_model_deterministic(
        self, bert_model: BertMaskedLM, bert_tokenizer: BertTokenizer
    ) -> None:
        """Test model's deterministic behavior.

        Args:
            bert_model: Custom BERT model implementation
            bert_tokenizer: BERT tokenizer instance

        Verifies that the model produces consistent outputs for the same input.
        """
        text = "testing deterministic behavior"
        tokens = bert_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        filtered_tokens = {
            k: v for k, v in tokens.items() if k in ["input_ids", "token_type_ids"]
        }

        bert_model.eval()
        with torch.no_grad():
            output1 = bert_model(**filtered_tokens)
            output2 = bert_model(**filtered_tokens)

        assert torch.allclose(output1, output2)
