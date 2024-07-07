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
    <td>Prior-init (500steps)</td>
    <td><video src="https://github.com/littlepure2333/prior-driven-init-gs/assets/36270711/5897ddec-1564-44d5-9e2e-ca29e48aec00" autoplay loop muted controls></td>
    <td><video src="https://github.com/littlepure2333/prior-driven-init-gs/assets/36270711/3541911d-2c2a-40ab-9812-8b17c476cc5e" autoplay loop muted controls></td>
    <td><video src="https://github.com/littlepure2333/prior-driven-init-gs/assets/36270711/20e02542-460e-4264-990c-096df3c73e76" autoplay loop muted controls></td>
  </tr>
  <tr>
    <td>Random-init (5000 steps)</td>
    <td><video src="https://github.com/littlepure2333/prior-driven-init-gs/assets/36270711/1cc309b4-e724-416d-9693-06649ae2741f" autoplay loop muted controls></td>
    <td><video src="https://github.com/littlepure2333/prior-driven-init-gs/assets/36270711/1aa09f23-f6e0-473a-a938-b874fc12e8d5" autoplay loop muted controls></td>
    <td><video src="https://github.com/littlepure2333/prior-driven-init-gs/assets/36270711/103bd1bb-9dc0-4e7c-a2e2-25ad9023c180" autoplay loop muted controls></td>
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
