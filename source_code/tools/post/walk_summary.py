'''
summarize a group of exp, gen summary.txt
can be called by post_process.py to summarize multiple groups of exp
'''

import numpy as np
import os
import argparse
import pandas as pd
import sys
print(os.path.abspath(__file__))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from tools.macro.macro import *

parser = argparse.ArgumentParser()
parser.add_argument('--group_dir', type=str)
parser.add_argument('--tag', type=str, default='train')
parser.add_argument('--gen_hyper_tune_csv', default=False, action='store_true')
parser.add_argument('--gen_five_csv', default=False, action='store_true')
args = parser.parse_args()

postfix = '\summary.txt' if args.tag == 'train' else '\eval_summary.txt'
sum_dir = args.group_dir + postfix

def write_summary():
    with open(sum_dir, 'w') as f:
        for root, dirs, files in os.walk(args.group_dir):
            for file in files:
                if file == f'{args.tag}_output.txt':
                    abs_path = os.path.join(root, file)
                    f.write(abs_path + '\n')
                    with open(abs_path, 'r') as re_f:
                        text = re_f.read()
                        if '\nsvo' in text:  # copo
                            metrics = text[text.rindex(f'best_{args.tag}_reward'):text.rindex('\nsvo')]
                        elif '\nphi' in text:  # hcopo
                            metrics = text[text.rindex(f'best_{args.tag}_reward'):text.rindex('\nphi')]
                        else:  # copo
                            assert '\n' in text
                            metrics = text[text.rindex(f'best_{args.tag}_reward'):text.rindex('\n')]
                        f.write(metrics + '\n\n')
                    print(1)


def two_dimensional_spline(data):
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib as mpl
    from scipy import interpolate
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    def func(x, y):
        return (x + y) * np.exp(-5.0 * (x ** 2 + y ** 2))

    # X-Y20*20
    x = np.array(range(4))
    y = np.array(range(3))
    x, y = np.meshgrid(x, y)  #
    fvals = func(x, y)

    # Draw sub-graph1
    ax = plt.subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(x, y, fvals, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=True)  # ，，，
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_zlim3d(0.6, 1.0)
    plt.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

'''hyper tuning'''
def gen_swapped_hyper_tune_csv():
    def parse_multi_index(line):
        multi_index = [None, None]
        # index1
        if 'EoiCoef=0.001' in line:
            multi_index[0] = HT_INDEX1[0]
        elif 'EoiCoef=0.003' in line:
            multi_index[0] = HT_INDEX1[1]
        elif 'EoiCoef=0.03' in line:
            multi_index[0] = HT_INDEX1[3]
        else:  # 0.01
            multi_index[0] = HT_INDEX1[2]
        # index2
        if 'ShareLayer_CCobs' in line:
            multi_index[1] = HT_INDEX2[3]
        elif 'ShareLayer' in line:
            multi_index[1] = HT_INDEX2[1]
        elif 'CCobs' in line:
            multi_index[1] = HT_INDEX2[2]
        else:
            multi_index[1] = HT_INDEX2[0]
        return tuple(multi_index)

    metrics = METRICS  # metric

    df = pd.DataFrame(np.zeros((len(HT_INDEX1)*len(metrics), len(HT_INDEX2))), columns=HT_INDEX2)
    with open(sum_dir, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            if line == '\n': continue
            # if-else
            if line.endswith('output.txt\n'):  #
                multi_index = parse_multi_index(line)
                start_row = len(metrics) * HT_INDEX1.index(multi_index[0])
                col = HT_INDEX2.index(multi_index[1])
            else:  #
                item = []
                for metric in metrics:
                    print(metric)
                    if metric in line:
                        sub = line.index(metric) + len(metric) + 2
                        item.append(line[sub:sub+5])
                    else:
                        item.append('0.0')
                df.iloc[start_row:start_row+len(metrics), col] = item  # 4

    df.index = pd.MultiIndex.from_product([HT_INDEX1, metrics])
    df.to_csv(args.group_dir + '/hyper_tune.csv')

    # data ratio
    a = [i*len(metrics) for i in range(len(HT_INDEX1))]  # loss ratio, i*len(METRICS)+1
    b = range(len(HT_INDEX2))
    data_ratio_array = df.iloc[a, b].values.astype(np.float)
    two_dimensional_spline(data_ratio_array)

'''five'''
def gen_five_csv():
    if args.group_dir.endswith('NU'):
        index = FIVE_NU_INDEX
        x = 'NU'
    elif args.group_dir.endswith('SD'):
        index = FIVE_SD_INDEX
        x = 'SD'
    elif args.group_dir.endswith('NS'):
        index = FIVE_NS_INDEX
        x = 'NS'
    elif args.group_dir.endswith('UH'):
        index = FIVE_UH_INDEX
        x = 'UH'
    else:
        raise NotImplementedError('')

    def parse_index(line):
        for ind in sorted(index, key=lambda x: len(x), reverse=True):  # , NU=10NU=1
            if ind in line:
                return ind

    def compute_efficiency(item):
        item['efficiency 1 (use UUF)'] = float(item['collect_data_ratio']) * (1 - float(item['loss_ratio'])) * float(item['uav_util_factor']) / float(item['energy_consumption_ratio'])
        item['efficiency 1 (use UUF)'] = str(np.round(item['efficiency 1 (use UUF)'], 3))
        item['efficiency 2 (use fairness)'] = float(item['collect_data_ratio']) * (1 - float(item['loss_ratio'])) * float(item['fairness']) / float(item['energy_consumption_ratio'])
        item['efficiency 2 (use fairness)'] = str(np.round(item['efficiency 2 (use fairness)'], 3))
        return item

    metrics = METRICS_WITH_EFFICIENCY
    df = pd.DataFrame(np.zeros((len(index), len(metrics))), columns=metrics)

    with open(sum_dir, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            if line == '\n': continue
            # if-else
            if line.endswith('output.txt\n'):  #
                ind = parse_index(line)
                row = index.index(ind)
            else:  #
                item = dict()
                for col in metrics:
                    if col in line:
                        start = line.index(col) + len(col) + 2  # +2 output.txt
                        end = start + line[start:].index('.') + 4  # scalar
                        item[col] = line[start:end]
                    else:
                        item[col] = '0.0'
                item = compute_efficiency(item)  # train_output.txtefficiency，
                df.loc[row] = item
    df.index = index
    df.to_csv(args.group_dir + f'/five_{x}.csv')


write_summary()
if args.gen_hyper_tune_csv:
    gen_swapped_hyper_tune_csv()
if args.gen_five_csv:
    gen_five_csv()
