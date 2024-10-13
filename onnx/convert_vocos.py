from argparse import ArgumentParser

from vocos import Vocos
from torch import nn


def main() -> None:
    parser = ArgumentParser(
        name="Convert vocos to onnx",
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

    def forward(self, x):
        return self.decode(x)

    vocos = Vocos.from_pretrained(args.model_name)
    vocos.forward = forward

    dummy_input = torch.randn(1, 80, 100)
    torch.onnx.export(vocos, dummy_input, args.output_path, verbose=True)


if __name__ == "__main__":
    main()