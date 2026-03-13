# read observations from robot
joint_states = piper.GetArmJointMsgs().joint_state
joint_states = np.array((
    joint_states.joint_1 / 1000.0,
    joint_states.joint_2 / 1000.0,
    joint_states.joint_3 / 1000.0,
    joint_states.joint_4 / 1000.0,
    joint_states.joint_5 / 1000.0,
    joint_states.joint_6 / 1000.0,
))
gripper_states = np.array(piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0).reshape(1)
# print("Joint states:", joint_states.shape, "Gripper state:", gripper_states.shape)
if t ==0: 
    origin_observation_state = np.concatenate((joint_states, gripper_states), axis=0)
# state = np.concatenate((joint_states, gripper_states), axis=0)
# print("Obs state:", state)

current_observation_state = np.concatenate((joint_states, gripper_states), axis=0)
print("Current observation state:", current_observation_state)
if t == 0:
    # history_action = np.zeros((1,32))
    pad = np.zeros((len_prev_actions, 32), dtype=np.float32)
    obs_state_padded = np.zeros(32, dtype=np.float32)
    obs_state_padded[:origin_observation_state.shape[0]] = origin_observation_state
    pad[:] = obs_state_padded
    history_action = pad
elif t < len_prev_actions:
    pad_len = len_prev_actions - t
    # Pad observation_state to shape (pad_len, 32)
    pad = np.zeros((pad_len, 32), dtype=np.float32)
    obs_state_padded = np.zeros(32, dtype=np.float32)
    obs_state_padded[:origin_observation_state.shape[0]] = origin_observation_state
    pad[:] = obs_state_padded
    print(f"pad shape {pad.shape}")
    history_action = np.concatenate(
        (pad, np.array(history_actions).reshape(-1, 32)),
        axis=0
)
else:   
    history_action = np.array(history_actions[-len_prev_actions:]).reshape(-1, 32)

obs = {
    'observation/wrist_image': cv2.resize(all_device_images[0], (224, 224)),
    'observation/image': cv2.resize(all_device_images[1], (224, 224)),
    'observation/state': current_observation_state,
    # 'prev_actions': history_action,
    'prompt': "sequentially touch the yellow, blue and red blocks",
}
