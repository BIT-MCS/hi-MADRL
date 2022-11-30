## h/i-MADRL

This is the code accompanying the paper:  "Exploring both Individuality and Cooperation for Air-Ground Spatial Crowdsourcing by Multi-Agent Deep Reinforcement Learning", to be appear in ICDE 2023.

## :page_facing_up: Description

Spatial crowdsourcing (SC) has proven as a promising paradigm to employ human workers to collect data from diverse Point-of-Interests (PoIs) in a given area. Different from using human participants, we propose a novel air-ground SC scenario to fully take advantage of benefits brought by unmanned vehicles (UVs), including unmanned aerial vehicles (UAVs)
with controllable high mobility and unmanned ground vehicles (UGVs) with abundant sensing resources. The objective is to maximize the amount of collected data, geographical fairness among all PoIs, and minimize the data loss and energy consumption, integrated as one single metric called “efficiency”. We explicitly explore both individuality and cooperation natures of UAVs and UGVs by proposing a multi-agent deep reinforcement learning (MADRL) framework called “h/i-MADRL”. Compatible with all multi-agent actor-critic methods, h/i-MADRL adds two novel plug-in modules: (a) h-CoPO, which models the cooperation preference among heterogenous UAVs and UGVs; and (b) i-EOI, which extracts the UV’s individuality and encourages better spatial division of work by adding intrinsic reward. Extensive experimental results on two real-world datasets on Purdue and NCSU campuses confirm that h/i-MADRL achieves a better exploration of both individuality and cooperation simultaneously, resulting in a better performance in terms of efficiency compared with five baselines.

Overview of h/i-MADRL:

![archi1_6_9改](https://cdn.jsdelivr.net/gh/1candoallthings/figure-bed@main/img/202206091648024.png)

## :wrench: ​Installation
Here we give an example installation on CUDA == 11.4.

1. Clone repo

   ```
   git clone https://github.com/BIT-MCS/hi-MADRL.git
   cd hi-MADRL	
   ```

2. Create conda environment

   ```
   conda create --name hi-MADRL python=3.7
   conda activate hi-MADRL
   ```

3. Install dependent packages

   1. Install torch

      ```
      pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
      ```

   2. In order to use osmnx to generate roadmap, some dependencies are required

      - [GDAL](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal) (note the version of GDAL should compatible with your Python version)
      - [fiona](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal) 
      - osmnx
      - geopandas
      - movingpandas

   3. For other required packages, please run the code and find which required package hasn't installed yet. Most of them can be installed by `pip install`.

## :computer: Training

To train hi-MADRL, use:

```
python main_PPO_vecenv.py --dataset <DATASET_STR> --use_eoi --use_hcopo
```

where `<DATASET_STR>` can be "purdue" or "NCSU". Default hyperparameters of hi-MADRL are used and the default simulation settings are summarized in Table 2.

add `--output_dir <OUTPUT_DIR>` to specify the place to save outputs（by default outputs are saved in  `../runs/debug`).

*For ablation study, simply remove `--use_eoi` or `--use_hcopo` or both of them.*

## :checkered_flag: Visualization

The output of training includes

- tensorboard

- `model` saved best model

- `train_saved_trajs` saved best trajectories for UAVs and UGVs

- `train_output.txt` records the performance in terms of 5 metrics:

  ```
  best trajs have been changed in ts=200. best_train_reward: 0.238 efficiency: 2.029 collect_data_ratio: 0.550 loss_ratio: 0.011 fairness: 0.577 energy_consumption_ratio: 0.155
  ```

To generate visualized trajectories, Use:

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

## :clap: Reference

- https://github.com/XinJingHao/PPO-Continuous-Pytorch
- https://github.com/decisionforce/CoPO
- https://github.com/jiechuanjiang/EOI_on_SMAC


## :scroll: Acknowledgement

This work was sponsored by the National Natural Science Foundation of China (No. U21A20519 and 62022017). 

Corresponding Author: Jianxin Zhao.

## :e-mail: Contact

If you have any question, please email `3120220985@bit.edu.cn`.

## Paper

If you are interested in our work, please cite our paper as

```
coming soon~~
```

