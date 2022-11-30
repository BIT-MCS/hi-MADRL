'''
draw visualized trajectories for a group of exp, generate .html for each exp
can be called by post_process.py to draw groups of exp
'''

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--group_dir', type=str)
parser.add_argument('--only_best', default=True, action='store_false')
parser.add_argument('--gif', default=True, action='store_false')

parser.add_argument("--draw_car_lines", default=True, action='store_false', help='for vis_gif.py')
parser.add_argument("--draw_uav_lines", default=True, action='store_false', help='for vis_gif.py')
parser.add_argument('--diff_color', default=True, action='store_false', help='for vis_gif.py')
args = parser.parse_args()

exps = []
for root, dirs, files in os.walk(args.group_dir):
    if root == args.group_dir:
        for dir in dirs:
            exps.append(os.path.join(root, dir))

for output_dir in exps:
    print(output_dir)
    if args.gif:
        postfix1 = '' if args.draw_car_lines else ' --draw_car_lines '
        postfix2 = '' if args.draw_uav_lines else ' --draw_uav_lines '
        postfix3 = '' if args.diff_color else ' --diff_color '
        os.system(f"python vis_gif.py --output_dir {output_dir} "
                  f"--group_save_dir {os.path.join(args.group_dir, 'group_gif_final')}"
                  + postfix1 + postfix2 + postfix3)
    else:
        os.system(f'python vis_roadmap.py --output_dir {output_dir}')



