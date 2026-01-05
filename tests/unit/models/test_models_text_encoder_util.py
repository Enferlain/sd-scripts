import torch
from unittest.mock import MagicMock, patch
from transformers import CLIPTextModelWithProjection
from library.models.text_encoder_util import pool_workaround, get_hidden_states_sdxl


class TestPoolWorkaround:
    def test_finds_eos_token_correctly(self):
        # Mock text encoder with projection
        text_encoder = MagicMock(spec=CLIPTextModelWithProjection)
        text_encoder.text_projection = MagicMock()
        text_encoder.text_projection.weight.dtype = torch.float32
        # Identity projection for simplicity
        text_encoder.text_projection.side_effect = lambda x: x

        # Create input: batch_size=2, seq_len=77, hidden_dim=768
        last_hidden_state = torch.randn(2, 77, 768)
        # EOS at position 10 and 15
        input_ids = torch.zeros(2, 77, dtype=torch.long)
        input_ids[0, 10] = 49407  # EOS token ID
        input_ids[1, 15] = 49407

        result = pool_workaround(text_encoder, last_hidden_state, input_ids, eos_token_id=49407)

        # Verify result shape
        assert result.shape == (2, 768)

        # Verify values match hidden states at EOS positions
        torch.testing.assert_close(result[0], last_hidden_state[0, 10])
        torch.testing.assert_close(result[1], last_hidden_state[1, 15])

    def test_handles_no_eos_token(self):
        text_encoder = MagicMock(spec=CLIPTextModelWithProjection)
        text_encoder.text_projection = MagicMock()
        text_encoder.text_projection.weight.dtype = torch.float32
        text_encoder.text_projection.side_effect = lambda x: x

        last_hidden_state = torch.randn(1, 77, 768)
        # No EOS token
        input_ids = torch.zeros(1, 77, dtype=torch.long)

        result = pool_workaround(text_encoder, last_hidden_state, input_ids, eos_token_id=49407)

        # Should default to index 0
        torch.testing.assert_close(result[0], last_hidden_state[0, 0])

    def test_multiple_eos_tokens_uses_first(self):
        text_encoder = MagicMock(spec=CLIPTextModelWithProjection)
        text_encoder.text_projection = MagicMock()
        text_encoder.text_projection.weight.dtype = torch.float32
        text_encoder.text_projection.side_effect = lambda x: x

        last_hidden_state = torch.randn(1, 77, 768)
        input_ids = torch.zeros(1, 77, dtype=torch.long)
        # EOS at 5 and 10
        input_ids[0, 5] = 49407
        input_ids[0, 10] = 49407

        result = pool_workaround(text_encoder, last_hidden_state, input_ids, eos_token_id=49407)

        # Argmax finds first occurrence
        torch.testing.assert_close(result[0], last_hidden_state[0, 5])

    def test_applies_projection_with_dtype_conversion(self):
        text_encoder = MagicMock(spec=CLIPTextModelWithProjection)
        text_encoder.text_projection = MagicMock()
        text_encoder.text_projection.weight.dtype = torch.float16  # Model weights are FP16

        # Mock projection to return something distinct and check input dtype
        def projection_mock(x):
            assert x.dtype == torch.float16
            return x * 2

        text_encoder.text_projection.side_effect = projection_mock

        # Input is FP32
        last_hidden_state = torch.randn(1, 77, 768, dtype=torch.float32)
        input_ids = torch.zeros(1, 77, dtype=torch.long)
        input_ids[0, 10] = 49407

        result = pool_workaround(text_encoder, last_hidden_state, input_ids, eos_token_id=49407)

        # Result should be cast back to input dtype (FP32)
        assert result.dtype == torch.float32
        # Expected value must mimic the implementation's cast to fp16 and back to fp32
        # implementation: cast(input -> fp16) -> proj -> cast(output -> fp32)
        expected = (last_hidden_state[0, 10].to(torch.float16) * 2).to(torch.float32)
        torch.testing.assert_close(result[0], expected)

    def test_device_handling(self):
        text_encoder = MagicMock(spec=CLIPTextModelWithProjection)
        text_encoder.text_projection = MagicMock()
        text_encoder.text_projection.weight.dtype = torch.float32
        text_encoder.text_projection.side_effect = lambda x: x

        # Use CPU for test stability, but verify tensors stay on same device
        device = torch.device("cpu")
        last_hidden_state = torch.randn(1, 77, 768, device=device)
        input_ids = torch.zeros(1, 77, dtype=torch.long, device=device)
        input_ids[0, 10] = 49407

        result = pool_workaround(text_encoder, last_hidden_state, input_ids, eos_token_id=49407)
        assert result.device == device


class TestGetHiddenStatesSDXL:
    def setUp(self):
        pass

    def test_get_hidden_states_sdxl_basic_reshaping(self):
        # Mock tokenizers
        tokenizer1 = MagicMock()
        tokenizer1.model_max_length = 77
        tokenizer1.eos_token_id = 49407

        tokenizer2 = MagicMock()
        tokenizer2.model_max_length = 77
        tokenizer2.eos_token_id = 49407

        # Mock text encoders
        text_encoder1 = MagicMock()
        text_encoder2 = MagicMock()

        # Setup encoder outputs
        # n_size=1 (default when max_token_length is None)
        # batch=2, n=1 -> total=2
        enc1_out = {
            "hidden_states": [torch.randn(2, 77, 768) for _ in range(12)]  # 12 layers
        }
        text_encoder1.return_value = enc1_out

        enc2_out = {
            "hidden_states": [torch.randn(2, 77, 1280) for _ in range(33)],  # 32 layers + emb? needs index -2
            "last_hidden_state": torch.randn(2, 77, 1280),
        }
        text_encoder2.return_value = enc2_out

        # Input: batch=2, n=1, seq_len=77
        input_ids1 = torch.randint(0, 49408, (2, 1, 77))
        input_ids2 = torch.randint(0, 49408, (2, 1, 77))

        with patch("library.models.text_encoder_util.pool_workaround") as mock_pool:
            # Pool returns (batch*n, 1280)
            mock_pool.return_value = torch.randn(2, 1280)

            h1, h2, pool = get_hidden_states_sdxl(
                max_token_length=None,
                input_ids1=input_ids1,
                input_ids2=input_ids2,
                tokenizer1=tokenizer1,
                tokenizer2=tokenizer2,
                text_encoder1=text_encoder1,
                text_encoder2=text_encoder2,
            )

        # Verify shapes
        # (batch, n*77, dim)
        assert h1.shape == (2, 77, 768)
        assert h2.shape == (2, 77, 1280)
        assert pool.shape == (2, 1280)

    def test_max_token_length_chunking(self):
        # Test the "三連" logic with 225 tokens (75*3)
        # max_token_length = 225
        # n_size = 225 // 75 = 3
        # batch = 1

        max_token_length = 225
        n_size = 3
        batch_size = 1

        tokenizer1 = MagicMock()
        tokenizer1.model_max_length = 77
        tokenizer1.eos_token_id = 49407
        tokenizer2 = MagicMock()
        tokenizer2.model_max_length = 77
        tokenizer2.eos_token_id = 49407

        text_encoder1 = MagicMock()
        text_encoder2 = MagicMock()

        # Input: (batch, n, 77) -> (1, 3, 77)
        # Reshaped inside to (3, 77)
        input_ids1 = torch.zeros(batch_size, n_size, 77, dtype=torch.long)
        input_ids2 = torch.zeros(batch_size, n_size, 77, dtype=torch.long)

        enc1_out = {"hidden_states": [torch.randn(3, 77, 768) for _ in range(12)]}
        text_encoder1.return_value = enc1_out

        enc2_out = {"hidden_states": [torch.randn(3, 77, 1280) for _ in range(33)], "last_hidden_state": torch.randn(3, 77, 1280)}
        text_encoder2.return_value = enc2_out

        with patch("library.models.text_encoder_util.pool_workaround") as mock_pool:
            # Pool expects (batch*n, 1280) = (3, 1280)
            mock_pool.return_value = torch.randn(3, 1280)

            h1, h2, pool = get_hidden_states_sdxl(
                max_token_length=max_token_length,
                input_ids1=input_ids1,
                input_ids2=input_ids2,
                tokenizer1=tokenizer1,
                tokenizer2=tokenizer2,
                text_encoder1=text_encoder1,
                text_encoder2=text_encoder2,
            )

        # Expected concatenation:
        # Each chunk contributes 75 tokens except borders?
        # Logic:
        # i=1: [1:1+77-2] = [1:76] (75 tokens)
        # i=78: [78:78+77-2] = [78:153] (75 tokens)
        # And BOS (1) + EOS (1) = 227?

        # Verify shapes
        assert h1.shape == (batch_size, 227, 768)
        assert h2.shape == (batch_size, 227, 1280)

        # Verify pool selection
        # pool2 = pool2[::n_size] -> pool2[0] (since batch=1)
        # expected shape (batch, 1280)
        assert pool.shape == (batch_size, 1280)

    def test_max_token_length_150(self):
        """Test with max_token_length=150 (2 chunks, less common than 225)."""
        max_token_length = 150
        n_size = 2
        batch_size = 2

        tokenizer1 = MagicMock(model_max_length=77, eos_token_id=49407)
        tokenizer2 = MagicMock(model_max_length=77, eos_token_id=49407)
        text_encoder1 = MagicMock()
        text_encoder2 = MagicMock()

        # Input: (batch, n, 77) = (2, 2, 77)
        input_ids1 = torch.zeros(batch_size, n_size, 77, dtype=torch.long)
        input_ids2 = torch.zeros(batch_size, n_size, 77, dtype=torch.long)

        # Encoder outputs for batch*n = 4 sequences
        enc1_out = {"hidden_states": [torch.randn(4, 77, 768) for _ in range(12)]}
        text_encoder1.return_value = enc1_out

        enc2_out = {"hidden_states": [torch.randn(4, 77, 1280) for _ in range(33)], "last_hidden_state": torch.randn(4, 77, 1280)}
        text_encoder2.return_value = enc2_out

        with patch("library.models.text_encoder_util.pool_workaround") as mock_pool:
            mock_pool.return_value = torch.randn(4, 1280)

            h1, h2, pool = get_hidden_states_sdxl(
                max_token_length=max_token_length,
                input_ids1=input_ids1,
                input_ids2=input_ids2,
                tokenizer1=tokenizer1,
                tokenizer2=tokenizer2,
                text_encoder1=text_encoder1,
                text_encoder2=text_encoder2,
            )

        # Expected sequence length for 150:
        # [BOS] + chunk[1:76] (75) + chunk[1:76] (75) + [EOS] = 152
        assert h1.shape == (batch_size, 152, 768)
        assert h2.shape == (batch_size, 152, 1280)

        # Pool selection: pool2[::n_size] -> pool2[[0, 2]] for batch=2
        assert pool.shape == (batch_size, 1280)

    def test_accelerator_unwrapping(self):
        tokenizer1 = MagicMock(model_max_length=77)
        tokenizer2 = MagicMock(model_max_length=77)
        text_encoder1 = MagicMock()
        text_encoder1.return_value = {"hidden_states": [torch.randn(1, 77, 768) for _ in range(12)]}
        text_encoder2 = MagicMock()
        text_encoder2.return_value = {
            "hidden_states": [torch.randn(1, 77, 1280) for _ in range(33)],
            "last_hidden_state": torch.randn(1, 77, 1280),
        }

        accelerator = MagicMock()
        accelerator.unwrap_model.return_value = text_encoder2

        with patch("library.models.text_encoder_util.pool_workaround") as mock_pool:
            mock_pool.return_value = torch.randn(1, 1280)

            get_hidden_states_sdxl(
                max_token_length=None,
                input_ids1=torch.zeros(1, 1, 77, dtype=torch.long),
                input_ids2=torch.zeros(1, 1, 77, dtype=torch.long),
                tokenizer1=tokenizer1,
                tokenizer2=tokenizer2,
                text_encoder1=text_encoder1,
                text_encoder2=text_encoder2,
                accelerator=accelerator,
            )

        accelerator.unwrap_model.assert_called_with(text_encoder2)

    def test_dtype_conversion(self):
        tokenizer1 = MagicMock(model_max_length=77)
        tokenizer2 = MagicMock(model_max_length=77)
        text_encoder1 = MagicMock()
        text_encoder1.return_value = {"hidden_states": [torch.randn(1, 77, 768) for _ in range(12)]}
        text_encoder2 = MagicMock()
        text_encoder2.return_value = {
            "hidden_states": [torch.randn(1, 77, 1280) for _ in range(33)],
            "last_hidden_state": torch.randn(1, 77, 1280),
        }

        with patch("library.models.text_encoder_util.pool_workaround", return_value=torch.randn(1, 1280)) as mock_pool:
            h1, h2, pool = get_hidden_states_sdxl(
                max_token_length=None,
                input_ids1=torch.zeros(1, 1, 77, dtype=torch.long),
                input_ids2=torch.zeros(1, 1, 77, dtype=torch.long),
                tokenizer1=tokenizer1,
                tokenizer2=tokenizer2,
                text_encoder1=text_encoder1,
                text_encoder2=text_encoder2,
                weight_dtype=torch.float16,
            )

            # Pool should NOT be converted inside get_hidden_states_sdxl (it returns pool2 directly from pool_workaround)
            # pool_workaround returns whatever dtype it computes (usually based on input/projection).
            # If we mocked it to return float32 (default), it stays float32.
            assert pool.dtype == torch.float32

        assert h1.dtype == torch.float16
        assert h2.dtype == torch.float16
