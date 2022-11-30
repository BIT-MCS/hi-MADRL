# metrics
OLD_METRICS = ['collect_data_ratio', 'loss_ratio', 'energy_consumption_ratio', 'fairness', 'best_train_reward']
METRICS = ['collect_data_ratio', 'loss_ratio', 'energy_consumption_ratio', 'fairness', 'uav_util_factor', 'best_train_reward']
METRICS_WITH_EFFICIENCY = ['collect_data_ratio', 'loss_ratio', 'energy_consumption_ratio', 'fairness', 'uav_util_factor', 'efficiency 1 (use UUF)', 'efficiency 2 (use fairness)']
METRICS_FINAL = ["Data Collection Ratio", "Data Loss Ratio", "Energy Consumption Ratio", "Geographical Fairness", "Efficiency"]  #

# hyper tuning
HT_INDEX1 = ['EoiCoef=0.001', 'EoiCoef=0.003', 'EoiCoef=0.01', 'EoiCoef=0.03']
HT_INDEX2 = ['w/o SL, w/o CC', 'w/ SL, w/o CC', 'w/o SL, w/ CC', 'w/ SL, w/ CC']

# FIVE
## walk_summary.py，five_{algo}.csv
FIVE_NU_INDEX = ['NU=1', 'NU=2', 'NU=3', 'NU=4', 'NU=5', 'NU=7', 'NU=10']
FIVE_SD_INDEX = ['SD=0.2', 'SD=0.6', 'SD=1.0', 'SD=2.0', 'SD=5.0']
FIVE_NS_INDEX = ['NS=1', 'NS=2', 'NS=3', 'NS=4', 'NS=5', 'NS=7', 'NS=10']
FIVE_UH_INDEX = ['UH=50', 'UH=70', 'UH=90', 'UH=120', 'UH=150']

## compare，walkfive_{algo}.csv，df
ALGOS = ['OurSolution', 'Our(CoPO)', 'MAPPO', 'maddpg', 'TSP', 'random']

## compare
# yrange_for_metrics = {
#         "Data Collection Ratio": [0.0, 1.2],
#         "Data Loss Ratio": [0, 0.3],
#         "Energy Consumption Ratio": [0.0, 0.5],
#         "Geographical Fairness": [0.0, 1.2],
#         "Efficiency": [0.0, 12.0],
#     }

# 4_25
yrange_for_metrics = {
        "Data Collection Ratio": [0.0, 1.3],
        "Data Loss Ratio": [0, 0.35],
        "Energy Consumption Ratio": [0.0, 0.5],
        "Geographical Fairness": [0.0, 1.3],
        "Efficiency": [0.0, 13.0],
    }

xlabel_for_xs = {
    'NU': "No. of UAVs/UGVs",
    'SD': "SINR threshold (dB)",
    'NS': "No. of Subchannels",
    'UH': "UAV height (m)",
}
xtick_for_xs = {
    'NU': [1, 2, 3, 4, 5, 7, 10],
    'SD': [-7.0, -2.2, 0.0, 3.0, 7.0],
    'NS': [1, 2, 3, 4, 5, 7, 10],
    'UH': [60, 70, 90, 120, 150],
}
