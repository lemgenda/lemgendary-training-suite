"""Unit tests for notebook filename standardization (zero hyphens, strict underscores)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from training.notebooks.builders.colab import (
    generate_colab_inference_notebook,
    generate_colab_usage_notebook,
)
from training.notebooks.builders.kaggle import (
    generate_inference_notebook,
    generate_usage_notebook,
)


class TestNotebookNaming(unittest.TestCase):
    """Verifies that notebook builders strictly generate {model_key}_{platform}_{type}.ipynb."""

    def test_kaggle_notebook_filenames_zero_hyphens(self) -> None:
        """Verify Kaggle notebook generator produces zero-hyphen filenames."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nb_dir = Path(tmp_dir) / "notebooks"
            nb_dir.mkdir(parents=True)

            train_path = generate_inference_notebook("mirnet_exposure", export_dir=nb_dir)
            self.assertIsNotNone(train_path)
            train_file = Path(train_path).name
            self.assertEqual(train_file, "mirnet_exposure_kaggle_training.ipynb")
            self.assertNotIn("-", train_file)

            usage_path = generate_usage_notebook("mirnet_exposure", export_dir=nb_dir)
            self.assertIsNotNone(usage_path)
            usage_file = Path(usage_path).name
            self.assertEqual(usage_file, "mirnet_exposure_kaggle_usage.ipynb")
            self.assertNotIn("-", usage_file)

    def test_colab_notebook_filenames_zero_hyphens(self) -> None:
        """Verify Colab notebook generator produces zero-hyphen filenames."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nb_dir = Path(tmp_dir) / "notebooks"
            nb_dir.mkdir(parents=True)

            train_path = generate_colab_inference_notebook("nafnet_debluring", export_dir=nb_dir)
            self.assertIsNotNone(train_path)
            train_file = Path(train_path).name
            self.assertEqual(train_file, "nafnet_debluring_colab_training.ipynb")
            self.assertNotIn("-", train_file)

            usage_path = generate_colab_usage_notebook("nafnet_debluring", export_dir=nb_dir)
            self.assertIsNotNone(usage_path)
            usage_file = Path(usage_path).name
            self.assertEqual(usage_file, "nafnet_debluring_colab_usage.ipynb")
            self.assertNotIn("-", usage_file)

    def test_ai_helper_notebook_naming(self) -> None:
        """Verify notebook naming for ai_helper_lemgend."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nb_dir = Path(tmp_dir) / "notebooks"
            nb_dir.mkdir(parents=True)

            train_path = generate_inference_notebook("ai_helper_lemgend", export_dir=nb_dir)
            self.assertIsNotNone(train_path)
            train_file = Path(train_path).name
            self.assertEqual(train_file, "ai_helper_lemgend_kaggle_training.ipynb")
            self.assertNotIn("-", train_file)


if __name__ == "__main__":
    unittest.main()
