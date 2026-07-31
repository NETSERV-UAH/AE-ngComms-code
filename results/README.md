# Results

`published/` contains the exact CSV tables used for the article figures:

- `sweep_b_results.csv`: depth/topology and encoder/decoder complexity.
- `sweep_c_results.csv`: block length and nominal compression ratio.
- `sweep_d_results.csv`: latent dimension at fixed block length.
- `sweep_rice_results.csv`: per-window Rice-Golomb baseline.
- `sweep_gorilla_results.csv`: per-window Gorilla baseline.

The AE tables contain one row per seed and architecture. The baseline tables
contain one row per non-overlapping test window. Run
`python scripts/validate_published_results.py` to check row counts and the
summary values quoted in the article.

New runs are written to `results/reproduced/` by default so the archived
tables cannot be overwritten accidentally.

SHA-256:

```text
e2ecf0e29a0d18c688909d6e6d7cfa451b7b449e561b8d654407aeac20e235ee  sweep_b_results.csv
a4be63de0cdc6c3f57edbfe403fa1c32695d769ac2bffa09689bcb998d3ff3bf  sweep_c_results.csv
532157654796639d57c56bf3e59babdcfa866fd6829ad0a51b83871233af68d2  sweep_d_results.csv
5a566c3c480df77b471adf47a3f40a112fdea59ea67b955a05778377d5a9cf14  sweep_rice_results.csv
bf9f0a83a720af52c902459bc8d4c08f5215ff2a91570c21f3061f6624555765  sweep_gorilla_results.csv
```
