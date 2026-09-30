<h1 align="center">Frozen Scenes, Shifting Winners: Configuration Fragility in Text-to-3D Evaluation</h1>

<p align="center">
  <strong>
    <a href="https://yflam1.github.io/">Anson Y. Lam</a><sup>*</sup>,
    <a href="https://shuqing-li.github.io/">Shuqing Li</a><sup>*,&dagger;</sup>,
    <a href="https://www.cse.cuhk.edu.hk/lyu/home">Michael R. Lyu</a>
  </strong>
</p>

<p align="center">
  Department of Computer Science and Engineering, The Chinese University of Hong Kong<br>
  <sup>*</sup>Equal contribution. <sup>&dagger;</sup>Corresponding author.
</p>

<!-- Add an arXiv link here once the preprint is up. -->
<p align="center">
  <a href="https://yflam1.github.io/frozen-scenes-shifting-winners/"><strong>Project Page</strong></a> |
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><strong>arXiv</strong></a>
</p>

## 📝 Abstract

Can a text-to-3D leaderboard change when every generated scene stays fixed?
We audit this question for rendered-image evaluation, where camera settings and caption wording become part of the measurement protocol.
Across 300 frozen scenes from six generators, we vary eight render and caption factors for 19 alignment evaluators plus one perceptual-quality control, then test four targeted scene degradations.
Peak configuration variance exceeds between-generator variance for 17/19 alignment evaluators, with prompt-bootstrap lower bounds above 1 for 11/19.
Rankings are more stable than scores, yet 18/19 evaluators change their point-estimate winner under some configuration.
Pairwise protocol margin envelopes show which comparisons keep their direction across the tested settings, and no reversal survives simultaneous inference over the full search.
Thus the observed winner changes are descriptive, not confirmed changes in generator superiority.
Sensitivity remains separate: no evaluator, even the prompt-free control, exceeds 67% tie-adjusted directional discrimination on layout scrambling.
The audit separates score stability, decision uncertainty, and targeted sensitivity, and recommends reporting _(generator, score, card ID)_ with protocol-dependent comparisons and selection-aware uncertainty.

## 📦 Installation

```bash
conda create -n vlmunr python=3.13 -y
conda activate vlmunr
python -m pip install "setuptools<70.0.0" wheel
python -m pip install -e ".[blender,vqa]" --no-build-isolation
python -m pip install \
  "salesforce-lavis@git+https://github.com/anson416/LAVIS.git@single-env" \
  "transformers==4.49.0" --no-build-isolation
```

## 🎬 Quickstart

Score rendered scene images against prompt texts with the two installed command-line tools.

```bash
# CLIPScore
clipscore -i renders/scene_0/view_0.png -i renders/scene_0/view_1.png \
  -pt "a cozy living room with a fireplace"

# BLIPScore
blipscore -i renders/scene_0/view_0.png -i renders/scene_0/view_1.png \
  -pt "a cozy living room with a fireplace"
```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## 📚 Citation

If you find this code useful, please cite our paper.

```bibtex
@article{lam2026frozen,
  title={Frozen Scenes, Shifting Winners: Configuration Fragility in Text-to-3D Evaluation},
  author={Lam, Anson Y. and Li, Shuqing and Lyu, Michael R.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## 📄 License

This project is released under the [MIT License](LICENSE).
