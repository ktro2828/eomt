from __future__ import annotations

import argparse
import importlib
import os

import yaml
from huggingface_hub import hf_hub_download
from huggingface_hub.hub_mixin import torch
from torch import nn

from datasets.lightning_data_module import LightningDataModule
from inference.base import InferenceBase, Task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the configuration file")
    return parser.parse_args()


def load_data_module(config: dict) -> LightningDataModule:
    data_module_name, class_name = config["data"]["class_path"].rsplit(".", 1)
    data_module_cls = getattr(importlib.import_module(data_module_name), class_name)
    data_module_kwargs = config["data"].get("init_args", {})

    return data_module_cls(
        path="dummy_path",
        batch_size=1,
        num_workers=0,
        check_empty_targets=False,
        **data_module_kwargs,
    )


def build_network(
    config: dict,
    img_size: tuple[int, int],
    num_classes: int,
) -> nn.Module:
    # Load encoder
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(
        importlib.import_module(encoder_module_name), encoder_class_name
    )
    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get("init_args", {}))

    # Load network
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(
        importlib.import_module(network_module_name), network_class_name
    )
    network_kwargs = {
        k: v for k, v in network_cfg["init_args"].items() if k != "encoder"
    }

    return network_cls(
        masked_attn_enabled=False,
        num_classes=num_classes,
        encoder=encoder,
        **network_kwargs,
    )


def build_model(
    config: dict,
    img_size: tuple[int, int],
    num_classes: int,
) -> InferenceBase:
    network = build_network(config, img_size, num_classes)

    model_module_name, model_class_name = (
        config["model"]["class_path"].replace("training", "inference").rsplit(".", 1)
    )
    model_cls = getattr(importlib.import_module(model_module_name), model_class_name)
    model_kwargs = {
        k: v for k, v in config["model"]["init_args"].items() if k != "network"
    }
    if "stuff_classes" in config["data"].get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

    return model_cls(
        network=network,
        img_size=img_size,
        num_classes=num_classes,
        **model_kwargs,
    ).eval()


def load_checkpoint(model: InferenceBase, config: dict) -> tuple[InferenceBase, str]:
    name = config.get("trainer", {}).get("logger", {}).get("init_args", {}).get("name")

    if name is None:
        raise ValueError("No logger name found in the config. Please specify it.")

    state_dict_path = hf_hub_download(
        repo_id=f"tue-mps/{name}",
        filename="pytorch_model.bin",
    )
    state_dict: dict = torch.load(
        state_dict_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    state_dict.pop("criterion.empty_weight")
    model.load_state_dict(state_dict)
    return model, name


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_module = load_data_module(config)

    model = build_model(
        config,
        img_size=data_module.img_size,
        num_classes=data_module.num_classes,
    )
    model, model_name = load_checkpoint(model, config)

    match model.task():
        case Task.SEMANTIC:
            output_names = ["mask"]
        case Task.INSTANCE:
            output_names = ["mask"]
        case Task.PANOPTIC:
            output_names = ["semantic", "instance"]
        case _:
            raise ValueError(f"Unknown task: {model.task()}")

    os.makedirs("onnx", exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            torch.randn(1, 3, *data_module.img_size),
            os.path.join("onnx", f"{model_name}.onnx"),
            input_names=["input"],
            output_names=output_names,
            opset_version=17,
        )


if __name__ == "__main__":
    main()
