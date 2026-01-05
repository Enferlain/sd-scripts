"""
Unit tests for library/utils/huggingface_util.py.

Tests HuggingFace Hub API interactions with mocked peft calls.
"""

from unittest.mock import patch, MagicMock

from library.utils.huggingface_util import exists_repo, list_dir


class TestExistsRepo:
    """Tests for exists_repo function."""

    def test_returns_true_when_repo_exists(self):
        """Should return True when repo_info succeeds."""
        with patch("library.utils.huggingface_util.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api_class.return_value = mock_api
            mock_api.repo_info.return_value = {"id": "test/repo"}

            result = exists_repo("test/repo", "model")

            assert result is True
            mock_api.repo_info.assert_called_once_with(repo_id="test/repo", revision="main", repo_type="model")

    def test_returns_false_when_repo_not_found(self):
        """Should return False when repo_info raises exception."""
        with patch("library.utils.huggingface_util.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api_class.return_value = mock_api
            mock_api.repo_info.side_effect = Exception("Not found")

            result = exists_repo("nonexistent/repo", "model")

            assert result is False

    def test_uses_provided_revision(self):
        """Should use the provided revision parameter."""
        with patch("library.utils.huggingface_util.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api_class.return_value = mock_api
            mock_api.repo_info.return_value = {}

            exists_repo("test/repo", "model", revision="dev")

            mock_api.repo_info.assert_called_once_with(repo_id="test/repo", revision="dev", repo_type="model")

    def test_uses_provided_token(self):
        """Should pass token to HfApi constructor."""
        with patch("library.utils.huggingface_util.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api_class.return_value = mock_api
            mock_api.repo_info.return_value = {}

            exists_repo("test/repo", "model", token="hf_token123")

            mock_api_class.assert_called_once_with(token="hf_token123")

    def test_supports_dataset_repo_type(self):
        """Should work with dataset repo type."""
        with patch("library.utils.huggingface_util.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api_class.return_value = mock_api
            mock_api.repo_info.return_value = {}

            result = exists_repo("test/dataset", "dataset")

            assert result is True
            mock_api.repo_info.assert_called_with(repo_id="test/dataset", revision="main", repo_type="dataset")


class TestListDir:
    """Tests for list_dir function."""

    def test_returns_files_in_subfolder(self):
        """Should return files that start with the subfolder prefix."""
        with patch("library.utils.huggingface_util.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api_class.return_value = mock_api

            # Mock file siblings
            file1 = MagicMock()
            file1.rfilename = "models/model.safetensors"
            file2 = MagicMock()
            file2.rfilename = "models/config.json"
            file3 = MagicMock()
            file3.rfilename = "README.md"

            mock_repo_info = MagicMock()
            mock_repo_info.siblings = [file1, file2, file3]
            mock_api.repo_info.return_value = mock_repo_info

            result = list_dir("test/repo", "models", "model")

            assert len(result) == 2
            assert file1 in result
            assert file2 in result
            assert file3 not in result

    def test_returns_empty_for_nonexistent_subfolder(self):
        """Should return empty list when no files match subfolder."""
        with patch("library.utils.huggingface_util.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api_class.return_value = mock_api

            file1 = MagicMock()
            file1.rfilename = "other/file.txt"

            mock_repo_info = MagicMock()
            mock_repo_info.siblings = [file1]
            mock_api.repo_info.return_value = mock_repo_info

            result = list_dir("test/repo", "models", "model")

            assert len(result) == 0

    def test_uses_provided_revision(self):
        """Should use the provided revision parameter."""
        with patch("library.utils.huggingface_util.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api_class.return_value = mock_api

            mock_repo_info = MagicMock()
            mock_repo_info.siblings = []
            mock_api.repo_info.return_value = mock_repo_info

            list_dir("test/repo", "subfolder", "model", revision="v1.0")

            mock_api.repo_info.assert_called_once_with(repo_id="test/repo", revision="v1.0", repo_type="model")

    def test_uses_provided_token(self):
        """Should pass token to HfApi constructor."""
        with patch("library.utils.huggingface_util.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api_class.return_value = mock_api

            mock_repo_info = MagicMock()
            mock_repo_info.siblings = []
            mock_api.repo_info.return_value = mock_repo_info

            list_dir("test/repo", "subfolder", "model", token="hf_abc")

            mock_api_class.assert_called_once_with(token="hf_abc")

    def test_handles_empty_siblings(self):
        """Should handle repos with no files."""
        with patch("library.utils.huggingface_util.HfApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api_class.return_value = mock_api

            mock_repo_info = MagicMock()
            mock_repo_info.siblings = []
            mock_api.repo_info.return_value = mock_repo_info

            result = list_dir("test/repo", "subfolder", "model")

            assert result == []
