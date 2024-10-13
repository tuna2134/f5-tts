from argparse import ArgumentParser

from vocos import Vocos
from torch import nn
import torch


def main() -> None:
    parser = ArgumentParser(
        description="VocosをONNXに変換します。",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="charactr/vocos-mel-24khz",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    vocos = Vocos.from_pretrained(args.model_name)

    def forward(x):
        return vocos.decode(x)

    vocos.forward = forward

    dummy_input = torch.randn(512, 100, 7)
    exporter = torch.onnx.dynamo_export(vocos, dummy_input)
    exporter.export(args.output_path)


if __name__ == "__main__":
    main()