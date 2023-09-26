## fast_gym_n.py

Each n env, inherits from FastGymBase, and extends it with:

- New action and observation space:         super().set_spaces(low_action, high_action, low_obs, high_obs)
- map_inputs
- map_outputs
- Reward