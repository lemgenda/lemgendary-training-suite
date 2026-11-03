import tempfile
import unittest
from pathlib import Path

import torch

from training.export.soup import blend_state_dicts, create_pareto_model_soup


class TestParetoModelSoup(unittest.TestCase):
    def test_uniform_blending(self) -> None:
        sd1 = {"weight": torch.tensor([1.0, 3.0]), "bias": torch.tensor([2.0])}
        sd2 = {"weight": torch.tensor([3.0, 5.0]), "bias": torch.tensor([4.0])}

        blended = blend_state_dicts([sd1, sd2])
        self.assertTrue(torch.allclose(blended["weight"], torch.tensor([2.0, 4.0])))
        self.assertTrue(torch.allclose(blended["bias"], torch.tensor([3.0])))

    def test_weighted_blending(self) -> None:
        sd1 = {"weight": torch.tensor([0.0, 0.0])}
        sd2 = {"weight": torch.tensor([10.0, 10.0])}

        blended = blend_state_dicts([sd1, sd2], weights=[0.25, 0.75])
        self.assertTrue(torch.allclose(blended["weight"], torch.tensor([7.5, 7.5])))

    def test_checkpoint_soup_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            p1 = tmp / "vault_psnr.pth"
            p2 = tmp / "vault_lpips.pth"
            out = tmp / "blended_sota.pth"

            torch.save({"model_state": {"w": torch.tensor([1.0, 1.0])}, "epoch": 5}, p1)
            torch.save({"model_state": {"w": torch.tensor([3.0, 3.0])}, "epoch": 8}, p2)

            res = create_pareto_model_soup([p1, p2], weights=[0.5, 0.5], output_path=out)
            self.assertTrue(out.exists())
            self.assertTrue(res["is_pareto_soup"])
            self.assertTrue(torch.allclose(res["model_state"]["w"], torch.tensor([2.0, 2.0])))


if __name__ == "__main__":
    unittest.main()

