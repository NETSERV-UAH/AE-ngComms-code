# AE-ngComms-code

Reproducible research code for the experimental study in:

> **The Role of Autoencoders in Next-Generation Communication Systems:
> Architectures and Domain-Specific Paradigms**
>
> David Carrascal, Javier Diaz-Fuentes, Elisa Rojas,
> Joaquín Álvarez-Horcajo, and José M. Arco.

The repository evaluates how block length, latent dimension, network depth,
and encoder-decoder asymmetry affect absolute-temperature reconstruction,
nominal compression, and a compute-derived energy proxy. The study compares
fully connected symmetric autoencoders (AEs), asymmetric autoencoders (AAEs),
Rice-Golomb coding, and Gorilla XOR coding.

## Requirements

- Python 3.10 or newer.
- A CPU is sufficient. CUDA is used automatically when available.
- The full 50-seed experiment is computationally expensive; use `--quick`
  first to verify an installation.

Create an isolated environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[plot,test]"
```

Run the test suite and validate the archived tables:

```bash
python -m pytest
python scripts/validate_published_results.py
```

## Reproducing the experiments

A smoke test runs one small configuration from each selected sweep:

```bash
ae-ngcomms-run \
  --sweeps b c d rice gorilla \
  --quick
```

The complete article protocol uses 50 seeds, at most 300 epochs, and early
stopping after 15 validation epochs:

```bash
ae-ngcomms-run \
  --sweeps b c d rice gorilla \
  --seeds 50 \
  --epochs 300 \
  --patience 15
```

New tables are written to `results/reproduced/`. The exact tables used in
the manuscript remain immutable in `results/published/`.

The sweep identifiers correspond to:

| Sweep | Variables | Fixed settings |
|---|---|---|
| `b` | Depth index 1–5; AE versus AAE | `n=100`, `m=25` |
| `c` | `n={128,256,512,1024}`, `R={2,4,8,16,32}` | Depth index 2 |
| `d` | `m={2,4,8,16,32,64,128}` | `n=256`, depth index 2 |
| `rice` | Rice-Golomb over first differences | Non-overlapping test windows |
| `gorilla` | Gorilla XOR over absolute float32 values | Non-overlapping test windows |

Regenerate the result figures from either the published or reproduced
tables:

```bash
ae-ngcomms-plot \
  --results-dir results/published \
  --output-dir figures/reproduced
```

## Experimental protocol

The included trace has 115,235 Caples Lake N7 air-temperature observations.
The deterministic chronological split contains 79,710 training, 9,220
validation, and 26,305 test observations, with 2017 reserved for testing.
Training windows use strides 10, 27, and 33; validation and test windows do
not overlap.

Each AE window is converted to first differences and independently min-max
normalized. The decoder output is denormalized and cumulatively summed using
the stored reference value. Hidden widths are geometrically spaced, hidden
layers use ELU, the latent projection uses SELU, and the output uses a
sigmoid. Training minimizes normalized-domain MSE with Adam at a learning
rate of `2e-3`; reported MAE is measured after reconstructing absolute
temperature.

For the asymmetric topology at depth index `h`, the encoder receives
`floor((h-1)/2)` hidden layers and the decoder receives the remainder. For
the symmetric topology, both branches receive `h` hidden layers. Linear-layer
MACs are counted analytically, and the energy proxy uses `4.6 pJ/MAC`.

Nominal AE compression is `R=n/m`. A transmitted latent vector also needs
three floating-point values per window (minimum difference, maximum
difference, and reference), so the corresponding value-count ratio is
`n/(m+3)`. The archived sweep-B table retains its historical `ratio` column
for compatibility and also records dimensions from which either definition
can be recovered.

## Repository layout

- `src/ae_ngcomms/`: maintained models, data pipeline, codecs, training,
  experiments, and plotting code.
- `scripts/`: experiment/plot wrappers, result validation, and architecture
  figure generation.
- `data/`: the exact input trace and its provenance.
- `results/published/`: exact CSV tables used in the article.
- `figures/published/`: final vector result figures.
- `tests/`: model, data, codec, and result-integrity tests.

Exploratory scripts, superseded experiments, generated model weights,
platform-specific dependency lists, and the vendored PlotNeuralNet checkout
from the development repository are intentionally excluded.

## Data

The included trace is a subset of the American River Hydrologic Observatory
dataset:

> Bales, R.; Cui, G.; Rice, R.; Meng, X.; Zhang, Z.; Hartsough, P.;
> Glaser, S.; Conklin, M. (2020). *Snow depth, air temperature, humidity,
> soil moisture and temperature, and solar radiation data from the
> basin-scale wireless-sensor network in American River Hydrologic
> Observatory (ARHO)* [Dataset]. Dryad.
> https://doi.org/10.6071/M39Q2V

The dataset is available under CC0. See `data/README.md` and `NOTICE`.

## Research-code notice

This code supports source-compression experiments over a noiseless latent
path. The AE ratio is not a bit-rate measurement, the lossless baselines are
not rate-matched to the learned representation, and the energy values are
MAC-derived proxies rather than hardware measurements. Validate
quantization, serialization, memory, radio, and channel costs before drawing
deployment conclusions.

## Citation

Please cite the release-specific Zenodo record and the associated article
when available. Machine-readable metadata are provided in `CITATION.cff`.

## License

The software is licensed under the Apache License 2.0. See `LICENSE`.
The data subset is distributed under its original CC0 terms and is not
relicensed under Apache-2.0.
