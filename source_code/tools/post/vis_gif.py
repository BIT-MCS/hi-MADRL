# -*-coding:utf-8-*-
# -*-coding:utf-8-*-

import sys
import time
from selenium import webdriver
import math
import os
import os.path as osp
import matplotlib.pyplot as plt
from src.envs.noma_env.utils import *
import json
import osmnx as ox
import networkx as nx
import folium
import pandas as pd
import geopandas as gpd
import movingpandas as mpd
from folium.plugins import TimestampedGeoJson
from src.envs.roadmap_env.roadmap_utils import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir")
parser.add_argument("--group_save_dir")
parser.add_argument("--tag", type=str, default='train', choices=['train', 'eval'], help='load trajs from train or eval')
parser.add_argument("--traj_filename", type=str, default='eps_best.npz', help='load which .npz file')
parser.add_argument("--draw_car_lines", default=True, action='store_false')
parser.add_argument("--draw_uav_lines", default=True, action='store_false')
parser.add_argument("--diff_color", default=True, action='store_false')
parser.add_argument("--ps", default=False, action='store_true')  # strongly specific to a certain case
parser.add_argument("--set_range", default=False, action='store_true')  # strongly specific to a certain case
parser.add_argument("--check_roadmap", default=False, action='store_true')  # don't render any trajs

args = parser.parse_args()


def get_arg_postfix(args):
    arg_postfix = ''
    if args.draw_car_lines:
        arg_postfix += '_drawUgvLines'
    if args.draw_uav_lines:
        arg_postfix += '_drawUavLines'
    if not args.diff_color:
        arg_postfix += '_NotDiffColor'
    if args.set_range:
        arg_postfix += '_SetRange'
    return arg_postfix

def get_save_postfix_by_copo_tune(exp_args):
    postfix = ''
    postfix += '_' + str(exp_args['eoi3_coef'])
    postfix += '_SL' if exp_args['share_layer'] else ''
    postfix += '_CC' if exp_args['use_ccobs'] else ''
    return postfix

def get_save_postfix_by_sinr_demand(exp_args):
    postfix = ''
    postfix += '_' + str(exp_args['sinr_demand'])
    return postfix

def main(args):
    os.chdir('../../')
    traj_file = osp.join(args.output_dir, f'{args.tag}_saved_trajs/{args.traj_filename}')
    trajs = np.load(traj_file)
    uav_trajs, car_trajs = list(trajs['arr_1']), list(trajs['arr_2'])

    if args.set_range:
        if 'purdue' in args.output_dir:
            # purdue Ours
            uav_trajs[0][4:] = [uav_trajs[0][3] for _ in range(len(uav_trajs[0][4:]))]  # purdue, uav1, 0~4
            uav_trajs[1][68:] = [uav_trajs[1][67] for _ in range(len(uav_trajs[1][68:]))]  # purdue, uav2, 68
            car_trajs[0][2:] = [car_trajs[0][1] for _ in range(len(car_trajs[0][2:]))]  # purdue, car1, 0~1
            car_trajs[1][2:] = [car_trajs[1][1] for _ in range(len(car_trajs[1][2:]))]  # purdue, car2, 0~53
        elif 'NCSU' in args.output_dir:
            # NCSU Ours
            uav_trajs[0][79:] = [uav_trajs[0][78] for _ in range(len(uav_trajs[0][79:]))]  # NCSU, uav1, 0~79
            uav_trajs[1][16:] = [uav_trajs[1][15] for _ in range(len(uav_trajs[1][16:]))]  # NCSU, uav2, 0~16
            # car_trajs[0][16:] = [car_trajs[0][15] for _ in range(len(car_trajs[0][16:]))]  # NCSU, car1, 0~16
            car_trajs[1][2:] = [car_trajs[1][1] for _ in range(len(car_trajs[1][2:]))]  # NCSU, car2, 0~1
        else:
            raise ValueError
    # ==

    if args.check_roadmap:
        uav_trajs[0][1:] = [uav_trajs[0][0] for _ in range(len(uav_trajs[0][1:]))]  # purdue, uav1, 0~4
        uav_trajs[1][1:] = [uav_trajs[1][0] for _ in range(len(uav_trajs[1][1:]))]  # purdue, uav2, 68
        car_trajs[0][1:] = [car_trajs[0][0] for _ in range(len(car_trajs[0][1:]))]  # purdue, car1, 0~1
        car_trajs[1][1:] = [car_trajs[1][0] for _ in range(len(car_trajs[1][1:]))]  # purdue, car2, 0~53


    json_file = osp.join(args.output_dir, 'params.json')
    with open(json_file, 'r') as f:
        result = json.load(f)
        args.setting_dir = result['setting_dir']
        args.roadmap_dir = result['args']['roadmap_dir']
        args.dataset = result['args']['dataset']
        env_config = result['my_env_config']
    num_uav = env_config['num_uav']
    num_car = env_config['num_car']
    num_agent = num_uav + num_car
    num_human = env_config['num_human']
    num_timestep = env_config['num_timestep']


    try:
        G = ox.load_graphml(args.roadmap_dir)
    except PermissionError:
        G = ox.load_graphml(args.roadmap_dir + '/map.graphml')

    rm = Roadmap(args.dataset)
    map = folium.Map(location=[(rm.lower_left[1] + rm.upper_right[1]) / 2, (rm.lower_left[0] + rm.upper_right[0]) / 2],
                     tiles="cartodbpositron", zoom_start=14, max_zoom=24)

    folium.TileLayer('Stamen Terrain').add_to(map)
    folium.TileLayer('Stamen Toner').add_to(map)
    folium.TileLayer('cartodbpositron').add_to(map)
    folium.TileLayer('OpenStreetMap').add_to(map)

    grid_geo_json = get_border(rm.upper_right, rm.lower_left)
    color = "red"
    weight = 4 if 'NCSU' in args.output_dir else 2  # 2 by default
    dashArray = '10,10' if 'NCSU' in args.output_dir else '5,5'  # '5,5' by default
    border = folium.GeoJson(grid_geo_json,
                            style_function=lambda feature, color=color: {
                                'fillColor': color,
                                'color': "black",
                                'weight': weight,
                                'dashArray': dashArray,
                                'fillOpacity': 0,
                            })
    map.add_child(border)

    for id in range(num_uav):
        for ts in range(len(uav_trajs[0])):
            uav_trajs[id][ts][:2] = rm.pygamexy2lonlat(*uav_trajs[id][ts][:2])
    for id in range(num_car):
        for ts in range(len(car_trajs[0])):
            car_trajs[id][ts][:2] = rm.pygamexy2lonlat(*car_trajs[id][ts][:2])

    # assemble route for cars
    routes = []
    assert num_timestep == len(car_trajs[0]) - 1
    for id in range(num_car):
        all_route = []
        for ts in range(num_timestep):
            origin_point = car_trajs[id][ts][:2]  # in osmnx, pos is (lon, lat)
            destination_point = car_trajs[id][ts+1][:2]
            origin_node = ox.distance.nearest_nodes(G, *origin_point)
            destination_node = ox.distance.nearest_nodes(G, *destination_point)
            route = nx.shortest_path(G, origin_node, destination_node, weight='length')
            if all_route == []:
                all_route += route
            else:
                all_route += route[1:]
        routes.append(all_route)

    # uv_color_dict = {
    #     'uav1': '#%02X%02X%02X' % (255, 0, 0),  # red, UAV1
    #     'uav2': '#%02X%02X%02X' % (255, 128, 0),  # orange, UAV2
    #     'car1': '#%02X%02X%02X' % (0, 0, 255),  # blue, UGV1
    #     'car2': '#%02X%02X%02X' % (50, 205, 50),  # lime green, UGV2
    # }

    uv_color_dict = {
        'uav1': '#%02X%02X%02X' % (255, 0, 0),  # red1, UAV1
        'uav2': '#%02X%02X%02X' % (255, 114, 86),  # red2, UAV2
        'car1': '#%02X%02X%02X' % (0, 0, 255),  # blue1, UGV1
        'car2': '#%02X%02X%02X' % (0, 191, 255),  # blue2, UGV2
    }

    ps = 1
    # draw static car traj
    if args.draw_car_lines and not args.check_roadmap:
        if args.ps:
            routes[1] = routes[1][:-10]  # delete last ten
        for i, route in enumerate(routes):
            # map = ox.folium.plot_route_folium(G, route, map, color='#%02X%02X%02X' % (0, 255, 0))
            color = uv_color_dict[list(uv_color_dict.keys())[i+num_uav]]
            map = ox.folium.plot_route_folium(G, route, map, color=color)

    # fillin positions for uav, car
    mixed_df = pd.DataFrame()
    for id in range(num_agent):
        df = pd.DataFrame(
            {'id': id, 't': pd.date_range(start='20230315090000', end=None, periods=num_timestep+1, freq='15s')})
        if id < num_uav:
            df['longitude'], df['latitude'] = uav_trajs[id][:, 0], uav_trajs[id][:, 1]  # x=lat, y=lon
        else:
            df['longitude'], df['latitude'] = car_trajs[id-num_uav][:, 0], car_trajs[id-num_uav][:, 1]  # x=lat, y=lon
        if args.ps and id == 3:
            df[-3:] = [df.iloc[-4] for _ in range(3)]  # last three point dont move
        mixed_df = mixed_df.append(df)

    # positions to traj
    mixed_gdf = gpd.GeoDataFrame(mixed_df, geometry=gpd.points_from_xy(mixed_df.longitude, mixed_df.latitude), crs=4326)
    mixed_gdf = mixed_gdf.set_index('t').tz_localize(None)
    trajs = mpd.TrajectoryCollection(mixed_gdf, 'id')

    def get_name_color_by_index(index):

        if index < num_uav:
            name = f"UAV {index}"
            if args.diff_color:
                color = uv_color_dict[list(uv_color_dict.keys())[index]]
            else:
                color = '#%02X%02X%02X' % (255, 0, 0)
        elif num_uav <= index < num_agent:
            name = f"CAR {index - num_uav}"
            if args.diff_color:
                color = uv_color_dict[list(uv_color_dict.keys())[index]]
            else:
                color = '#%02X%02X%02X' % (0, 0, 255)
        elif num_agent <= index:
            name = f"Human {index - num_agent}"
            color = '#%02X%02X%02X' % (0, 0, 0)
        else:
            raise ValueError
        return name, color

    for index, traj in enumerate(trajs.trajectories):
        name, color = get_name_color_by_index(index)

        features = traj_to_timestamped_geojson(index, traj, num_uav, num_agent, color, only_UVs=True)
        TimestampedGeoJson(
            {
                "type": "FeatureCollection",
                "features": features,
            },
            period="PT15S",
            add_last_point=True,
            transition_time=200,  # The duration in ms of a transition from between timestamps.
            max_speed=0.2,
            loop=True,
        ).add_to(map)

        # line for uav
        if args.draw_uav_lines and index < num_uav:
            geo_col = traj.to_point_gdf().geometry
            for s in range(geo_col.shape[0] - 2):
                xy = [[y, x] for x, y in zip(geo_col.x[s:s + 2], geo_col.y[s:s + 2])]
                f1 = folium.FeatureGroup(name)
                folium.PolyLine(locations=xy, color=color, weight=4, opacity=0.7).add_to(f1)  # opacity=1 might more beautiful
                f1.add_to(map)

    # dot for humans
    human_df = pd.read_csv(osp.join(args.setting_dir, 'human.csv'))
    for i in range(num_human):
        pos = rm.pygamexy2lonlat(human_df.iloc[i].px, human_df.iloc[i].py)
        color = '#%02X%02X%02X' % (0, 0, 0)
        folium_draw_CircleMarker(map, (pos[1], pos[0]), color, radius=3)  # in folium, (lat, lon)

    folium.LayerControl().add_to(map)

    # save
    arg_postfix = get_arg_postfix(args)
    if args.group_save_dir is None:
        vsave_dir = args.output_dir + f'/gif/{args.tag}_{args.traj_filename}'
        if not osp.exists(vsave_dir): os.makedirs(vsave_dir)
        map.get_root().save(vsave_dir + f'/{args.traj_filename[:-4]}{arg_postfix}.html')
    else:
        if not osp.exists(args.group_save_dir): os.makedirs(args.group_save_dir)
        # postfix = get_save_postfix_by_copo_tune(result['args'])  # note: only use this line when group=hypertune, otherwise use next line
        postfix = args.output_dir.split('\\')[-1]
        save_file = os.path.join(args.group_save_dir, f'{postfix}{arg_postfix}.html')
        map.get_root().save(save_file)

    print(1)

    # 4_25 try save by png
    # delay = 5
    # fn = 'testmap.html'
    # tmpurl = 'file://{path}/{mapfile}'.format(path=os.getcwd(), mapfile=fn)
    # map.save(fn)
    #
    # # browser = webdriver.Firefox()
    # browser = webdriver.Chrome()
    # browser.get(tmpurl)
    # # Give the map tiles some time to load
    # time.sleep(delay)
    # browser.save_screenshot('map.png')
    # browser.quit()

if __name__ == '__main__':
    main(args)
