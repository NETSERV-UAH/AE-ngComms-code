# Figures

`published/` contains the final vector figures prepared from the archived
result tables. To regenerate equivalent plots with the maintained Python
plotting code, run:

```bash
ae-ngcomms-plot \
  --results-dir results/published \
  --output-dir figures/reproduced
```

The generated PDFs are intentionally placed in an ignored directory.

The architecture comparison uses
[PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet), LaTeX, and
Poppler. It can be regenerated without vendoring the upstream project:

```bash
git clone https://github.com/HarisIqbal88/PlotNeuralNet.git /tmp/PlotNeuralNet
git -C /tmp/PlotNeuralNet checkout e96bc852189c2089dd500527a0a01a5a36e8977e
python scripts/generate_architecture_figure.py \
  --plotneuralnet /tmp/PlotNeuralNet \
  --output-dir figures/reproduced
```
