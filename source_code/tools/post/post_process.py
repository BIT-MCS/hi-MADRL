'''
conduct bat_rm.py and walk_summary.py for multiple groups of exp
'''

import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--group_dirs', type=str, nargs='+', help='multiple groups of experiments')
parser.add_argument('--only_walk_summary', default=False, action='store_true')

parser.add_argument('--tag', type=str, default='train', help='for walk_summary.py')
parser.add_argument('--gen_five_csv', default=False, action='store_true', help='for walk_summary.py')
parser.add_argument('--gen_hyper_tune_csv', default=False, action='store_true', help='for walk_summary.py')
parser.add_argument('--only_best', default=True, action='store_false', help='for bat_rm.py')

args = parser.parse_args()

for group_dir in args.group_dirs:
    postfix1 = ' --only_best ' if not args.only_best else ''
    postfix2 = ' --gen_five_csv ' if args.gen_five_csv else ''
    postfix3 = ' --gen_hyper_tune_csv ' if args.gen_hyper_tune_csv else ''

    os.system(f'python walk_summary.py --group_dir {group_dir} --tag {args.tag}' + postfix2 + postfix3)
    if args.only_walk_summary: continue
    os.system(f'python bat_rm.py --group_dir {group_dir}' + postfix1)