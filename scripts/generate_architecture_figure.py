#!/usr/bin/env python3
"""Generate the publication-ready AE/AAE architecture comparison.

The representative dimensions match the article: input dimension 256,
latent dimension 16, and depth index 3. PlotNeuralNet is supplied as an
argument so the upstream library does not need to be vendored here.

Example
-------
python scripts/generate_architecture_figure.py \
    --plotneuralnet /path/to/PlotNeuralNet
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

INPUT_DIM = 256
LATENT_DIM = 16
N_LAYERS = 3


def build_hidden_layers(start: int, end: int, n_hidden: int) -> list[int]:
    """Equivalent to main.py's np.geomspace-based layer construction."""
    if n_hidden <= 0:
        return []
    ratio = (end / start) ** (1 / (n_hidden + 1))
    return [round(start * ratio**i) for i in range(1, n_hidden + 1)]


def box(
    name: str,
    dimension: int,
    x: float,
    y: float,
    fill: str,
    caption: str,
) -> str:
    heights = {
        256: 20,
        128: 17,
        102: 15,
        64: 13,
        40: 11,
        32: 10,
        16: 7,
    }
    height = heights.get(dimension, max(7, min(20, 4 + 2.5 * math.log2(dimension))))
    width = 2.2 if dimension == 16 else 1.35
    return rf"""
\pic[shift={{({x},{y},0)}}] at (0,0,0)
  {{Box={{
    name={name},
    caption={{{caption}}},
    xlabel={{{{{dimension},}}}},
    fill={fill},
    opacity=0.70,
    height={height},
    width={width},
    depth={height}
  }}}};
"""


def connection(left: str, right: str, label: str = "") -> str:
    node = rf" node[midway,above,font=\scriptsize] {{{label}}}" if label else ""
    return (
        rf"\draw[draw=black,line width=0.35mm,opacity=0.65,"
        rf"postaction={{decorate}},decoration={{markings,"
        rf"mark=at position 0.5 with "
        rf"{{\arrow{{Stealth[length=2.2mm,width=1.5mm]}}}}}}] "
        rf"({left}-east) --{node} ({right}-west);" + "\n"
    )


def title(text: str, x: float, y: float) -> str:
    return rf"\node[font=\sffamily\bfseries\large] at ({x},{y},0) {{{text}}};" + "\n"


def note(text: str, x: float, y: float, color: str = "black") -> str:
    return (
        rf"\node[font=\sffamily\small,text={color},align=center] "
        rf"at ({x},{y},0) {{{text}}};" + "\n"
    )


def add_network(
    *,
    prefix: str,
    dimensions: list[int],
    split_index: int,
    y: float,
    x0: float = 0.0,
    gap: float = 2.1,
) -> str:
    """Draw one AE; split_index is the index of the latent layer."""
    chunks: list[str] = []
    names: list[str] = []
    for i, dimension in enumerate(dimensions):
        name = f"{prefix}{i}"
        names.append(name)
        if i == 0 or i == len(dimensions) - 1:
            fill = r"\IOColor"
            caption = "Input $\\mathbf{x}$" if i == 0 else "Output $\\hat{\\mathbf{x}}$"
        elif i == split_index:
            fill = r"\LatentColor"
            caption = "Latent space $\\mathbf{z}$"
        elif i < split_index:
            fill = r"\EncoderColor"
            caption = ""
        else:
            fill = r"\DecoderColor"
            caption = ""
        chunks.append(box(name, dimension, x0 + gap * i, y, fill, caption))
    for left, right in pairwise(names):
        chunks.append(connection(left, right))
    return "".join(chunks)


def document(tikzeng, body: str) -> str:
    colors = r"""
\definecolor{edgeblue}{RGB}{47,106,167}
\definecolor{cloudorange}{RGB}{230,126,34}
\definecolor{latentgreen}{RGB}{39,174,96}
\definecolor{iogray}{RGB}{155,164,176}
\usetikzlibrary{decorations.markings}
\def\EncoderColor{edgeblue}
\def\DecoderColor{cloudorange}
\def\LatentColor{latentgreen}
\def\IOColor{iogray}
\def\edgecolor{black}
"""
    return "".join(
        [
            tikzeng.to_head(str(Path(tikzeng.__file__).resolve().parents[1])),
            tikzeng.to_cor(),
            colors,
            tikzeng.to_begin(),
            body,
            tikzeng.to_end(),
        ]
    )


def compile_tex(tex_path: Path) -> None:
    subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            tex_path.name,
        ],
        cwd=tex_path.parent,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            "300",
            "-singlefile",
            tex_path.with_suffix(".pdf").name,
            tex_path.stem,
        ],
        cwd=tex_path.parent,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plotneuralnet",
        type=Path,
        required=True,
        help="Path to a checkout of HarisIqbal88/PlotNeuralNet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()

    library_root = args.plotneuralnet.resolve()
    if not (library_root / "pycore" / "tikzeng.py").exists():
        raise SystemExit(f"PlotNeuralNet not found at {library_root}")
    if shutil.which("pdflatex") is None or shutil.which("pdftoppm") is None:
        raise SystemExit("Both pdflatex and pdftoppm are required.")

    sys.path.insert(0, str(library_root))
    from pycore import tikzeng

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    symmetric_encoder = build_hidden_layers(INPUT_DIM, LATENT_DIM, N_LAYERS)
    symmetric_decoder = build_hidden_layers(LATENT_DIM, INPUT_DIM, N_LAYERS)
    symmetric_dims = [
        INPUT_DIM,
        *symmetric_encoder,
        LATENT_DIM,
        *symmetric_decoder,
        INPUT_DIM,
    ]

    asymmetric_encoder_layers = (N_LAYERS - 1) // 2
    asymmetric_decoder_layers = N_LAYERS - asymmetric_encoder_layers
    asymmetric_encoder = build_hidden_layers(
        INPUT_DIM, LATENT_DIM, asymmetric_encoder_layers
    )
    asymmetric_decoder = build_hidden_layers(
        LATENT_DIM, INPUT_DIM, asymmetric_decoder_layers
    )
    asymmetric_dims = [
        INPUT_DIM,
        *asymmetric_encoder,
        LATENT_DIM,
        *asymmetric_decoder,
        INPUT_DIM,
    ]

    symmetric_split = 1 + len(symmetric_encoder)
    asymmetric_split = 1 + len(asymmetric_encoder)
    gap = 2.1
    symmetric_center = gap * (len(symmetric_dims) - 1) / 2
    asymmetric_center = gap * (len(asymmetric_dims) - 1) / 2

    comparison_asymmetric_x0 = 20.5
    comparison_asymmetric_center = comparison_asymmetric_x0 + asymmetric_center
    comparison_body = (
        title(
            r"(a) Symmetric AE \quad $h_E=h_D=3$",
            symmetric_center,
            4.0,
        )
        + add_network(
            prefix="csym",
            dimensions=symmetric_dims,
            split_index=symmetric_split,
            y=0.0,
        )
        + note("Encoder", 4.2, 2.5, "edgeblue")
        + note("Decoder", 12.6, 2.5, "cloudorange")
        + title(
            r"(b) Asymmetric AE \quad $h_E=1,\ h_D=2$",
            comparison_asymmetric_center,
            4.0,
        )
        + add_network(
            prefix="casym",
            dimensions=asymmetric_dims,
            split_index=asymmetric_split,
            y=0.0,
            x0=comparison_asymmetric_x0,
        )
        + note("Encoder", comparison_asymmetric_x0 + 2.1, 2.5, "edgeblue")
        + note("Decoder", comparison_asymmetric_x0 + 7.35, 2.5, "cloudorange")
        + note(
            r"\textcolor{edgeblue}{\rule{0.42cm}{0.22cm}} Encoder\qquad"
            r"\textcolor{latentgreen}{\rule{0.42cm}{0.22cm}} Latent space\qquad"
            r"\textcolor{cloudorange}{\rule{0.42cm}{0.22cm}} Decoder",
            (symmetric_center + comparison_asymmetric_center) / 2,
            -4.5,
        )
    )

    figures = {"architectures": comparison_body}
    for stem, body in figures.items():
        tex_path = output_dir / f"{stem}.tex"
        tex_path.write_text(document(tikzeng, body), encoding="utf-8")
        compile_tex(tex_path)

    for extension in ("aux", "log"):
        for path in output_dir.glob(f"architectures.{extension}"):
            path.unlink()

    print("Generated:")
    for stem in figures:
        print(f"  {output_dir / (stem + '.pdf')}")
        print(f"  {output_dir / (stem + '.png')}")


if __name__ == "__main__":
    main()
