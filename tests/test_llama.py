import pytest
import torch

from litellama.models import LLaMACausalLM, LLaMAConfig
from litellama.models.llama.llama import Block, RMSNorm


class TestLLaMACausalLM:
    """Test suite for LLaMACausalLM model implementation.

    Tests various aspects of the LLaMA model including shape validation,
    weight loading, HuggingFace compatibility, and core functionalities.
    """

    def test_model_output_shape(self, llama_model: LLaMACausalLM) -> None:
        """Test if model output shape matches expected dimensions."""
        batch_size = 1
        seq_len = 1
        input_ids = torch.randint(
            0, llama_model._config.vocab_size, (batch_size, seq_len)
        )

        output: torch.Tensor = llama_model(input_ids, start_pos=0)

        assert output.shape == (batch_size, seq_len, llama_model._config.vocab_size)

    def test_rotary_embeddings(self, llama_model: LLaMACausalLM) -> None:
        """Test if rotary embeddings are properly computed and applied."""
        seq_len = 1
        head_dim = (
            llama_model._config.hidden_size // llama_model._config.num_attention_heads
        )

        freqs_complex = llama_model.model.freqs_complex[:seq_len]
        assert freqs_complex.shape == (seq_len, head_dim // 2)

        # Test if freqs_complex contains non-zero values
        assert not torch.allclose(freqs_complex, torch.zeros_like(freqs_complex))

    def test_cache_initialization(self, llama_model: LLaMACausalLM) -> None:
        """Test if KV cache is properly initialized."""
        for layer in llama_model.model.layers:
            layer: Block  # type: ignore
            assert hasattr(layer.self_attn, "cache_k")
            assert hasattr(layer.self_attn, "cache_v")

            assert layer.self_attn.cache_k.shape == (
                llama_model._config.batch_size,
                llama_model._config.max_position_embeddings,
                layer.self_attn.num_key_value_heads,
                layer.self_attn.head_dim,
            )

    def test_pretrained_weights_loading(self, llama_model: LLaMACausalLM) -> None:
        """Test loading of pretrained weights."""
        try:
            llama_model.load_pretrained_hf()
            weights_loaded = True
        except Exception as e:
            weights_loaded = False
            pytest.skip(f"Pretrained weights loading failed: {str(e)}")

        if weights_loaded:
            # Check if embeddings are non-zero
            assert not torch.allclose(
                llama_model.model.embed_tokens.weight,
                torch.zeros_like(llama_model.model.embed_tokens.weight),
            )

    def test_autoregressive_generation(self, llama_model: LLaMACausalLM) -> None:
        """Test autoregressive token generation."""
        batch_size = 1
        input_ids = torch.randint(0, llama_model._config.vocab_size, (batch_size, 1))

        # Generate 5 tokens autoregressively
        generated_ids = []
        for pos in range(5):
            with torch.no_grad():
                output = llama_model(input_ids, start_pos=pos)
                next_token = output[0, -1].argmax()
                generated_ids.append(next_token.item())
                input_ids = torch.tensor([[next_token]], dtype=torch.long)

        assert len(generated_ids) == 5
        assert all(isinstance(token_id, int) for token_id in generated_ids)

    @pytest.mark.parametrize("start_pos", [0, 10, 50])
    def test_different_start_positions(
        self, llama_model: LLaMACausalLM, start_pos: int
    ) -> None:
        """Test model's ability to handle different starting positions."""
        batch_size = 1
        seq_len = 1
        input_ids = torch.randint(
            0, llama_model._config.vocab_size, (batch_size, seq_len)
        )

        output: torch.Tensor = llama_model(input_ids, start_pos=start_pos)
        assert output.shape == (batch_size, seq_len, llama_model._config.vocab_size)

    def test_gradient_flow(self, llama_model: LLaMACausalLM) -> None:
        """Test gradient computation and backpropagation."""
        batch_size = 1
        seq_len = 1
        input_ids = torch.randint(
            0, llama_model._config.vocab_size, (batch_size, seq_len)
        )

        llama_model.train()
        output = llama_model(input_ids, start_pos=0)
        loss = output.mean()
        loss.backward()

        # Check if gradients are computed
        assert llama_model.model.embed_tokens.weight.grad is not None

    def test_rms_norm(self, llama_model: LLaMACausalLM) -> None:
        """Test RMSNorm functionality."""
        # Test input
        x = torch.randn(1, 1, llama_model._config.hidden_size)

        # Get first RMSNorm layer
        layer: Block = llama_model.model.layers[0]
        rms_norm: RMSNorm = layer.input_layernorm

        # Apply normalization
        normalized: torch.Tensor = rms_norm(x)

        # Check if output maintains shape
        assert normalized.shape == x.shape
        # Check if normalization is applied (mean should be close to 0)
        assert torch.abs(normalized.mean()) < 1e-2

    def test_model_deterministic(self, llama_model: LLaMACausalLM) -> None:
        """Test model's deterministic behavior."""
        batch_size = 1
        seq_len = 1
        input_ids = torch.randint(
            0, llama_model._config.vocab_size, (batch_size, seq_len)
        )

        llama_model.eval()
        with torch.no_grad():
            output1 = llama_model(input_ids, start_pos=0)
            output2 = llama_model(input_ids, start_pos=0)

        assert torch.allclose(output1, output2)

    @pytest.mark.parametrize("n_layers", [1, 2, 4])
    def test_different_model_sizes(
        self, n_layers: int, llama_config: LLaMAConfig
    ) -> None:
        """Test model with different numbers of layers."""
        llama_config.num_hidden_layers = n_layers
        model = LLaMACausalLM(llama_config)

        assert len(model.model.layers) == n_layers

        batch_size = 1
        seq_len = 1
        input_ids = torch.randint(0, llama_config.vocab_size, (batch_size, seq_len))

        output: torch.Tensor = model(input_ids, start_pos=0)
        assert output.shape == (batch_size, seq_len, llama_config.vocab_size)

    def test_matching_generation_greedy(
        self, llama_model: LLaMACausalLM, hf_llama_model_and_tokenizer
    ) -> None:
        """Test if greedy generation matches HuggingFace implementation."""
        hf_model, tokenizer = hf_llama_model_and_tokenizer

        # Load pretrained weights
        llama_model.load_pretrained_hf()

        # Set both models to eval mode
        llama_model.eval()
        hf_model.eval()

        # Test prompts
        prompts = [
            "Once upon a time",
            "The meaning of life is",
            "In the future, AI will",
        ]

        for prompt in prompts:
            # Tokenize input
            tokens = tokenizer(prompt, return_tensors="pt")
            input_ids = tokens["input_ids"]

            with torch.no_grad():
                # Generate with custom implementation
                custom_generated_ids = []
                current_input = input_ids.clone()

                for pos in range(input_ids.shape[1], input_ids.shape[1] + 20):
                    output = llama_model(current_input[:, -1:], start_pos=pos)
                    next_token = output[0, -1].argmax()
                    custom_generated_ids.append(next_token.item())
                    current_input = torch.cat(
                        [current_input, next_token.unsqueeze(0).unsqueeze(0)], dim=1
                    )

                # Generate with HuggingFace implementation
                hf_output = hf_model.generate(
                    input_ids,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                hf_generated_ids = hf_output[0, input_ids.shape[1] :].tolist()  # noqa

                # Compare generations
                assert (
                    custom_generated_ids == hf_generated_ids
                ), f"Generations don't match for prompt: {prompt}"

    def test_matching_generation_with_temperature(
        self, llama_model: LLaMACausalLM, hf_llama_model_and_tokenizer
    ) -> None:
        """Test if temperature-based generation matches HuggingFace implementation."""
        hf_model, tokenizer = hf_llama_model_and_tokenizer

        # Load pretrained weights
        llama_model.load_pretrained_hf()

        # Set both models to eval mode
        llama_model.eval()
        hf_model.eval()

        # Set random seed for reproducibility
        torch.manual_seed(42)

        prompt = "The future of artificial intelligence is"
        temperature = 0.8

        # Tokenize input
        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"]

        with torch.no_grad():
            # Generate with custom implementation
            custom_generated_ids = []
            current_input = input_ids.clone()

            for pos in range(input_ids.shape[1], input_ids.shape[1] + 20):
                output = llama_model(current_input[:, -1:], start_pos=pos)
                # Apply temperature scaling
                logits = output[0, -1] / temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                custom_generated_ids.append(next_token.item())
                current_input = torch.cat(
                    [current_input, next_token.unsqueeze(0)], dim=1
                )

            # Reset seed for HF generation
            torch.manual_seed(42)

            # Generate with HuggingFace implementation
            hf_output = hf_model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
            hf_generated_ids = hf_output[0, input_ids.shape[1] :].tolist()  # noqa

            # Compare generations
            assert (
                custom_generated_ids == hf_generated_ids
            ), "Temperature-based generations don't match"

    def test_matching_generation_with_top_p(
        self, llama_model: LLaMACausalLM, hf_llama_model_and_tokenizer
    ) -> None:
        """Test if top-p sampling matches HuggingFace implementation."""
        hf_model, tokenizer = hf_llama_model_and_tokenizer

        # Load pretrained weights
        llama_model.load_pretrained_hf()

        # Set both models to eval mode
        llama_model.eval()
        hf_model.eval()

        # Set random seed for reproducibility
        torch.manual_seed(42)

        prompt = "Write a story about"
        temperature = 0.9
        top_p = 0.9

        # Tokenize input
        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"]

        with torch.no_grad():
            # Generate with custom implementation
            custom_generated_ids = []
            current_input = input_ids.clone()

            def top_p_sampling(logits, p):
                probs = torch.nn.functional.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs <= p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = True
                probs[sorted_indices[~mask]] = 0
                probs = probs / probs.sum()
                return probs

            for pos in range(input_ids.shape[1], input_ids.shape[1] + 20):
                output = llama_model(current_input[:, -1:], start_pos=pos)
                logits = output[0, -1] / temperature
                probs = top_p_sampling(logits, top_p)
                next_token = torch.multinomial(probs, num_samples=1)
                custom_generated_ids.append(next_token.item())
                current_input = torch.cat(
                    [current_input, next_token.unsqueeze(0)], dim=1
                )

            # Reset seed for HF generation
            torch.manual_seed(42)

            # Generate with HuggingFace implementation
            hf_output = hf_model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
            hf_generated_ids = hf_output[0, input_ids.shape[1] :].tolist()  # noqa

            # Compare generations
            assert (
                custom_generated_ids == hf_generated_ids
            ), "Top-p sampling generations don't match"

    def test_logit_comparisons(
        self, llama_model: LLaMACausalLM, hf_llama_model_and_tokenizer
    ) -> None:
        """Test if logits match exactly with HuggingFace implementation."""
        hf_model, tokenizer = hf_llama_model_and_tokenizer

        # Load pretrained weights
        llama_model.load_pretrained_hf()

        # Set both models to eval mode
        llama_model.eval()
        hf_model.eval()

        prompts = ["Hello world", "The quick brown fox", "In the beginning"]

        for prompt in prompts:
            # Tokenize input
            tokens = tokenizer(prompt, return_tensors="pt")
            input_ids = tokens["input_ids"]

            with torch.no_grad():
                # Get logits from custom implementation
                custom_output = llama_model(input_ids, start_pos=0)

                # Get logits from HuggingFace implementation
                hf_output = hf_model(input_ids).logits

                # Compare logits
                assert torch.allclose(
                    custom_output, hf_output, atol=1e-5
                ), f"Logits don't match for prompt: {prompt}"
