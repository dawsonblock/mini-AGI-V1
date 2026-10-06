from pathlib import Path

import pytest

from kvcontinual.continual.integrations.mlx_lora import MlxLoraPlan


def test_mlx_lora_command_and_validation(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text('{"messages":[]}\n')
    plan = MlxLoraPlan(
        model="mlx-community/test-model",
        data_dir=str(data),
        adapter_path=str(tmp_path / "adapter"),
        iters=12,
        num_layers=2,
    )
    plan.validate()
    cmd = plan.command()
    assert "mlx_lm.lora" in cmd
    assert "--train" in cmd
    assert cmd[cmd.index("--iters") + 1] == "12"
    assert cmd[cmd.index("--num-layers") + 1] == "2"


def test_mlx_lora_requires_training_data(tmp_path: Path):
    plan = MlxLoraPlan("model", str(tmp_path / "missing"), str(tmp_path / "adapter"))
    with pytest.raises(FileNotFoundError):
        plan.validate()
