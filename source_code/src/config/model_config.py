model_config = {

    'gnn_config': {
        'similarity_function': 'embedded_gaussian',
        'layerwise_attn': False,
        'skip_connection': True,
        'embed_dim': 32,
        'u_embed_block_dims': [64, 32],  # last should equal to 'embed_dim'
        'c_embed_block_dims': [64, 32],
        'h_embed_block_dims': [64, 32],
        'b_embed_block_dims': [64, 32],
        'num_layer': 2,
        'uav_state_dim': 4,
        'car_state_dim': 4,
        'human_state_dim': 3,
        'building_state_dim': 6,
    },

    'mcts_config': {
        'do_action_clip': True,
        'planning_width': 1,
        'planning_depth': 1,
    },

    'v_network_config': {
        'v_network_dims': [32, 256, 256, 1],
    },


}
