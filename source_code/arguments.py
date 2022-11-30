def add_args(parser):
    parser.add_argument('--write', action='store_false', default=True, help='Use SummaryWriter to record the training')
    parser.add_argument('--test', action='store_true', default=False, help='True means test phase, otherwise train phase')
    parser.add_argument('--debug', action='store_true', default=False, )
    # parser.add_argument('--load_dir', type=str, help='If load model, specify the location')
    # parser.add_argument('--load_timestep', type=int, help='If load model, specify the timestep')
    parser.add_argument('--ModelIdex', type=int, default=400, help='which model to load')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--cuda', action='store_false', default=True)
    parser.add_argument('--gpu_id', type=str, default='0', help='')
    parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
    parser.add_argument('--Max_train_steps', type=int, default=1e6, help='Max training steps, rllib iter=1000 等价于1.2M个ts')
    parser.add_argument('--save_interval', type=int, default=2e5, help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=10e10, help='Model evaluating interval, in steps.')
    parser.add_argument('--num_test_episode', type=int, default=1, )

    parser.add_argument('--dist', type=str, default='Beta')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
    parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
    parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
    parser.add_argument('--W_epochs', type=int, default=5, help='SVO update times, same with pzh')
    parser.add_argument('--net_width', type=int, default=150, help='Hidden net width')
    parser.add_argument('--a_lr', type=float, default=2e-4, help='Learning rate of actor')
    parser.add_argument('--c_lr', type=float, default=2e-4, help='Learning rate of critic')
    parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')

    parser.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory of actor and critic')
    parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
    parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
    parser.add_argument('--vf_coef', type=float, default=1.0, )
    parser.add_argument('--share_parameter', action='store_true', default=False, help='Share parameter between agent or not')
    parser.add_argument('--use_ccobs', action='store_true', default=False, help='Centralized input of critics')

    # model
    parser.add_argument('--use_eoi', action='store_true', default=False)
    parser.add_argument('--use_copo', action='store_true', default=False)
    parser.add_argument('--use_hcopo', action='store_true', default=False)
    parser.add_argument('--eoi_kind', type=int, default=3, )
    parser.add_argument('--eoi3_coef', type=float, default=0.003, )
    parser.add_argument('--eoi_coef_decay', type=float, default=1.0, )

    # svo
    parser.add_argument('--copo_kind', type=int, default=1, help='1: copo 2: hetero-copo')
    parser.add_argument('--hcopo_shift', default=False, action='store_true')
    parser.add_argument('--hcopo_shift_513', default=False, action='store_true')
    parser.add_argument('--share_layer', action='store_true', default=False, help="if True, share layers of 1 actor and 3 critics in copo. only work when share_parameter=False")
    parser.add_argument('--HID_phi', default=[0, 0], nargs='+', type=int)
    parser.add_argument('--HID_theta', default=[45, 45], nargs='+', type=int)
    parser.add_argument('--svo_lr', default=1e-4, type=float)
    parser.add_argument('--nei_dis_scale', default=0.25, type=float)
    parser.add_argument('--hcopo_sqrt2_scale', default=True, action='store_false')
    parser.add_argument('--svo_frozen', default=False, action='store_true')

    # env
    parser.add_argument('--type2_act_dim', type=int, default=20)

    # dir
    parser.add_argument("--dataset", type=str, default='NCSU', choices=['NCSU', 'purdue', 'KAIST'])
    parser.add_argument("--config_dir", type=str, help='import which config file')
    parser.add_argument("--setting_dir", type=str, help='Will be used in noma env')
    parser.add_argument("--output_dir", type=str, default='../runs/debug', help="which fold to save under 'runs/'")

    # roadmap
    parser.add_argument('--roadmap_dir', type=str)
    parser.add_argument('--gr', type=int, default=200, choices=[0, 50, 100, 200],)
    # multi-thread
    parser.add_argument('--n_rollout_threads', type=int, default=32)

    # debug
    parser.add_argument('--reward_scale', type=float, default=1)

    parser.add_argument('--num_uv', type=int, default=None)
    parser.add_argument('--sinr_demand', type=float, default=None)
    parser.add_argument('--num_subchannel', type=int, default=None)
    parser.add_argument('--uav_height', type=int, default=None)

    return parser