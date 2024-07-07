# Prior-driven initialization of Gaussian Splatting

This is the implementation of prior-driven initialization described in "3.2.1 Single frame initialization" in ["GFlow: Recovering 4D World from Monocular Video"](https://littlepure2333.github.io/GFlow/).

This initialization method, which takes an image and its corresponding depth as input, can achieve faster 3DGS reconstruction compared to random initialization.

<table>
  <tr>
    <th>Method</th>
    <th>RGB</th>
    <th>Depth</th>
    <th>Centers</th>
  </tr>
  <tr>
    <td>Prior-init</td>
    <td><video src="data/training_rgb_prior.mp4" autoplay loop muted controls></td>
    <td><video src="data/training_depth_prior.mp4" autoplay loop muted controls></td>
    <td><video src="data/training_center_prior.mp4" autoplay loop muted controls></td>
  </tr>
  <tr>
    <td>Random-init</td>
    <td><video src="data/training_rgb_random.mp4" autoplay loop muted controls></td>
    <td><video src="data/training_depth_random.mp4" autoplay loop muted controls></td>
    <td><video src="data/training_center_random.mp4" autoplay loop muted controls></td>
  </tr>
</table>




# Usage
1. Install following [msplat's instructions](https://github.com/pointrix-project/msplat?tab=readme-ov-file#how-to-install).
2. run `python tutorials/run.py` and check the results in the `logs` folder.

## Acknowledgment
We thank the [msplat](https://github.com/pointrix-project/msplat) team for re-implementing 3DGS in a more developer-friendly way.
We also salute the original [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) team for their seminal work.

If you find this initialization method helpful, please consider to cite:
```
@article{wang2024gflow,
  title={GFlow: Recovering 4D World from Monocular Video},
  author={Wang, Shizun and Yang, Xingyi and Shen, Qiuhong and Jiang, Zhenxiang and Wang, Xinchao},
  journal={arXiv preprint arXiv:2405.18426},
  year={2024}
}
```
