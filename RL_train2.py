env = gym.make("Pong-v0")
observation = env.reset() # This gets us the image

# hyperparameters
episode_number = 0
batch_size = 10
gamma = 0.99 # discount factor for reward
decay_rate = 0.99
num_hidden_layer_neurons = 200
input_dimensions = 80 * 80
learning_rate = 1e-4

episode_number = 0
reward_sum = 0
running_reward = None
prev_processed_observations = None

weights = {
    '1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
    '2': np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
}

# To be used with rmsprop algorithm (http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop)
expectation_g_squared = {}
g_dict = {}
for layer_name in weights.keys():
    expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
    g_dict[layer_name] = np.zeros_like(weights[layer_name])

episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []


def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = input_observation[35:195] # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1 # everything else (paddles, ball) just set to 1
    # Convert from 80 x 80 matrix to 1600 x 1 matrix
    processed_observation = processed_observation.astype(np.float).ravel()

    # subtract the previous frame from the current one so we are only processing on changes in the game
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    # store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation
    return input_observation, prev_processed_observations



def apply_neural_nets(observation_matrix, weights):
    """ Based on the observation_matrix and weights, compute the new hidden layer values and the new output layer values"""
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values

while True:
    env.render()
    processed_observations, prev_processed_observations = preprocess_observations(observation, prev_processed_observations, input_dimensions)
    hidden_layer_values, up_probability = apply_neural_nets(processed_observations, weights)

    episode_observations.append(processed_observations)
    episode_hidden_layer_values.append(hidden_layer_values)

    action = choose_action(up_probability)

    # carry out the chosen action
    observation, reward, done, info = env.step(action)

    reward_sum += reward
    episode_rewards.append(reward)