## Readme

Overview of h/i-MADRL:

![archi1_6_9改](https://cdn.jsdelivr.net/gh/1candoallthings/figure-bed@main/img/202206091648024.png)

h/i-MADRL consists of one base module and two plug-in modules:

- the base module can be almost any multi-agent actor-critic algorithms, we use IPPO as the exemplar base module.

- the first plug-in module i-EOI encourages a better spatial division of work between UAVs and UGVs.


- the second plug-in module h-CoPO accurately modeling cooperation preferences for both UAVs and UGVs.

## Installation
Here we give an example installation on CUDA == 11.4.

Create conda environment

```
conda create --name hi-MADRL python=3.7
conda activate hi-MADRL
```

Install torch

```
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

In order to use osmnx to generate roadmap, some dependencies are required：

- [GDAL](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal) (note the version of GDAL should compatible with your Python version)

- [fiona](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal) 
- osmnx
- geopandas
- movingpandas

For other required packages, please run the code and find which required package hasn't installed yet. Most of them can be installed by `pip install`.

## How to train hi-MADRL

To train hi-MADRL, use:

```
python main_PPO_vecenv.py --dataset <DATASET_STR> --use_eoi --use_hcopo
```

where `<DATASET_STR>` can be "purdue" or "NCSU". Default hyperparameters of hi-MADRL are used and the default simulation settings are summarized in Table 2.

add `--output_dir <OUTPUT_DIR>` to specify the place to save outputs（by default outputs are saved in  `../runs/debug`).

*For ablation study, simply remove `--use_eoi` or `--use_hcopo` or both of them.*

## Outputs

- tensorboard
- `model` saved best model
- `train_saved_trajs` saved best trajectories for UAVs and UGVs

- `train_output.txt` records the performance in terms of 5 metrics:

  ```
  best trajs have been changed in ts=200. best_train_reward: 0.238 efficiency: 2.029 collect_data_ratio: 0.550 loss_ratio: 0.011 fairness: 0.577 energy_consumption_ratio: 0.155
  ```

## Visualized trajectories

Use:

```
python tools/post/vis_gif.py --output_dir <OUTPUT_DIR>
--group_save_dir <OUTPUT_DIR>
```

You can use our pretrained output:

```
python tools/post/vis_gif.py --output_dir runs\pretrained_output_purdue
--group_save_dir runs\pretrained_output_purdue
```

then a .html file showing visualized trajectories is generated:

![image-20220609183019344](https://cdn.jsdelivr.net/gh/1candoallthings/figure-bed@main/img/202206091830013.png)

you can drag the control panel at lower left corner to see how UAVs and UGVs move.
