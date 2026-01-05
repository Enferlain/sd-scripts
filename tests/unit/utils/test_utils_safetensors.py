import torch
import pytest
import os
import struct
import json
import numpy as np
import tempfile
from unittest.mock import MagicMock, patch
from library.utils.safetensors_utils import mem_eff_save_file, load_safetensors, MemoryEfficientSafeOpen, find_key


@pytest.fixture
def temp_safetensors_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "model.safetensors")
        tensors = {"key1": torch.tensor([1, 2, 3], dtype=torch.int64), "key2": torch.tensor([0.1, 0.2], dtype=torch.float32)}
        metadata = {"format": "pt", "version": "1.0"}
        mem_eff_save_file(tensors, file_path, metadata)
        yield file_path, tensors, metadata


class TestSafetensorsUtils:
    def test_mem_eff_save_and_load(self, temp_safetensors_file):
        file_path, original_tensors, _ = temp_safetensors_file
        loaded_tensors = load_safetensors(file_path, device="cpu")

        assert "key1" in loaded_tensors
        assert torch.equal(loaded_tensors["key1"], original_tensors["key1"])
        assert torch.allclose(loaded_tensors["key2"], original_tensors["key2"])

    def test_metadata_reading(self, temp_safetensors_file):
        file_path, _, original_metadata = temp_safetensors_file
        with MemoryEfficientSafeOpen(file_path) as f:
            metadata = f.metadata()
            assert metadata["format"] == original_metadata["format"]

    def test_find_key(self, temp_safetensors_file):
        file_path, _, _ = temp_safetensors_file
        assert find_key(file_path, starts_with="key", ends_with="1") == "key1"
        assert find_key(file_path, starts_with="nonexistent") is None

    def test_find_key_missing_file(self):
        with pytest.raises(FileNotFoundError):
            find_key("nonexistent_file.safetensors", starts_with="key")

    def test_manual_header_parsing_truncated(self, tmp_path):
        # Create a file smaller than 8 bytes
        bad_file = tmp_path / "bad.safetensors"
        bad_file.write_bytes(b"1234")

        with pytest.raises(struct.error):
            MemoryEfficientSafeOpen(str(bad_file))

    def test_manual_header_parsing_invalid_json(self, tmp_path):
        bad_file = tmp_path / "bad_json.safetensors"
        with open(bad_file, "wb") as f:
            # Header size 10
            f.write(struct.pack("<Q", 10))
            # Invalid JSON content
            f.write(b"NOT_JSON!!" * 1)

        with pytest.raises(json.JSONDecodeError):
            MemoryEfficientSafeOpen(str(bad_file))

    def test_mmap_logic_threshold(self, tmp_path):
        # We want to verify that for large tensors on GPU, mmap is attempted.
        # We'll mock the device and tensor size logic.

        file_path = tmp_path / "large.safetensors"
        # Create a dummy file with a header claiming a large tensor
        header = {
            "large_tensor": {
                "dtype": "F32",
                "shape": [100],
                "data_offsets": [0, 11 * 1024 * 1024],  # > 10MB
            }
        }
        header_json = json.dumps(header).encode("utf-8")
        # Align to 8 bytes isn't strictly enforced by our reader logic but good practice

        with open(file_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_json)))
            f.write(header_json)
            # Write dummy data
            f.truncate(f.tell() + 11 * 1024 * 1024)

        # Mock torch.cuda.device and numpy.memmap
        # logic: if num_bytes > 10MB and device is not None and device.type != "cpu"

        # Use a string for device if possible, or a MagicMock that we don't pass to real tensor
        mock_device = MagicMock()
        mock_device.type = "cuda"

        with (
            patch("numpy.memmap") as mock_memmap,
            patch("numpy.fromfile") as mock_fromfile,
            patch("torch.from_numpy") as mock_torch_from_numpy,
        ):
            # Setup torch.from_numpy to return a Mock tensor, not a real one
            # because we will call .to() on it with a Mock device
            mock_tensor = MagicMock()
            mock_torch_from_numpy.return_value = mock_tensor

            with MemoryEfficientSafeOpen(str(file_path)) as reader:
                # We need to bypass the _deserialize_tensor call crashing on mocked data or provide return value
                # If we mock _deserialize_tensor to return mock_tensor
                reader._deserialize_tensor = MagicMock(return_value=mock_tensor)

                reader.get_tensor("large_tensor", device=mock_device)

            # Verify memmap was called because size > 10MB and device is cuda
            mock_memmap.assert_called_once()
            mock_fromfile.assert_not_called()

    def test_disable_mmap_behavior_cpu(self, temp_safetensors_file):
        # Verify that disable_mmap=True uses MemoryEfficientSafeOpen which uses fromfile on CPU
        file_path, original_tensors, _ = temp_safetensors_file

        with patch("numpy.fromfile") as mock_fromfile:
            # We must provide a side_effect or return value that looks like a valid array
            # so the subsequent torch.from_numpy doesn't fail.

            # Needs to return valid data matching size
            # Calculate size of key1 (int64, 3 elements) = 24 bytes
            mock_fromfile.side_effect = lambda f, dtype, count: np.zeros(count, dtype=dtype)

            with patch("torch.from_numpy") as mock_tfn:
                # Mock return to avoid interaction issues
                mock_tfn.return_value = MagicMock()

                load_safetensors(file_path, device="cpu", disable_mmap=True)

                # Should have been called multiple times (once per tensor)
                assert mock_fromfile.call_count >= 1

    def test_float8_parsing_mock(self, tmp_path):
        # Create file with F8_E4M3 metadata
        file_path = tmp_path / "float8.safetensors"
        header = {"f8_tensor": {"dtype": "F8_E4M3", "shape": [10], "data_offsets": [0, 10]}}
        header_json = json.dumps(header).encode("utf-8")
        with open(file_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_json)))
            f.write(header_json)
            f.write(b"\x00" * 10)

        # Mock torch to ensure attributes exist or checking behavior
        pass

    def test_metadata_non_string_values(self, tmp_path):
        # Verify robust handling or failure if metadata values aren't strings
        file_path = tmp_path / "bad_meta.safetensors"

        # Test Save: Implementation prints warning and converts to string, does NOT raise
        with patch("builtins.print") as mock_print:
            mem_eff_save_file({"t": torch.tensor([1])}, str(file_path), metadata={"epoch": 10})

            # Verify warning was printed
            # Note: The exact string might vary depending on implementation (e.g. key vs value)
            # The code says: "Warning: Metadata value for key 'epoch' is not a string. Converting to string."
            mock_print.assert_called_with("Warning: Metadata value for key 'epoch' is not a string. Converting to string.")

        # Verify it reads back as string "10"
        with MemoryEfficientSafeOpen(str(file_path)) as f:
            assert f.metadata()["epoch"] == "10"
