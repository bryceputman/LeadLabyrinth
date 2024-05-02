import os
import pickle
import random
import time
import gymnasium as gym

import numpy as np
import optuna
from matplotlib import pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from game import SCREEN_WIDTH, SCREEN_HEIGHT, MAX_ENEMY_SPEED, MAX_BULLET_SPEED, Game, AIPlayer


class SingleAgentEnvironment(gym.Env):
    PLAYER_ATTRIBUTES = 2
    ENEMY_ATTRIBUTES = 8
    BULLET_ATTRIBUTES = 5+9
    NUM_ENEMIES_LISTED = 2
    NUM_BULLETS_LISTED = 12
    TOTAL_ATTRIBUTES = PLAYER_ATTRIBUTES + (ENEMY_ATTRIBUTES * NUM_ENEMIES_LISTED) + (
            BULLET_ATTRIBUTES * NUM_BULLETS_LISTED)

    def __init__(self, game, agent_id, control_clock=False, should_render=True, training_ai=True, playing_ai=False, delta_time=0.01666, frame_repeats=1):
        super(SingleAgentEnvironment, self).__init__()
        self.agent_id = agent_id
        self.game = game
        self.game.wave_manager.current_wave = 0
        self.game.wave_manager.generate_wave_patterns()
        self.game.wave_manager.generate_wave()
        self.game.wave_manager.wave_cooldown = 0  # Seconds in between each wave
        self.game.wave_ended = False  # Reset the flag as the new wave has started
        self.prev_health = None
        self.prev_enemy_count = None
        self.control_clock = control_clock
        self.player = self.game.players[agent_id]
        self.frame_repeats = frame_repeats
        self.action_space = gym.spaces.MultiDiscrete([2, 2, 2, 2])

        self.state_size = self.TOTAL_ATTRIBUTES
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.state_size,))
        self.prev_game_time = 0.0
        self.prev_enemy_healths = {}
        self.should_render = should_render
        if isinstance(delta_time, (tuple, list)) and len(delta_time) == 2:
            self.delta_time_range = delta_time
            self.delta_time = random.uniform(*delta_time)
            self.avg_delta_time = np.mean(self.delta_time_range)
        else:
            self.delta_time_range = None
            self.delta_time = delta_time
            self.avg_delta_time = delta_time

    def reset(self, seed=None):
        self.game.reset()
        self.player = next((player for player in self.game.players if isinstance(player, AIPlayer)), None)
        initial_state = self.get_state()
        reset_info = {}  # Populate this as needed, can remain empty if no additional info
        # Return both the initial state and the reset info
        return initial_state, reset_info

    def step(self, action):
        total_reward = 0
        terminated = False  # Indicates natural end of an episode
        truncated = False   # Indicates an external condition cutting the episode short
        info = {}

        for i in range(self.frame_repeats):
            if self.delta_time_range:
                self.delta_time = random.uniform(*self.delta_time_range)
            self.prev_health = self.player.health
            self.prev_enemy_count = len(self.game.wave_manager.enemies)
            if self.control_clock:
                self.game.game_logic(self.delta_time)
                if self.should_render:
                    self.game.draw_game_state()

            self.apply_action_to_agent(action)

            new_state = self.get_state()
            reward = self._get_reward()
            if self.player not in self.game.players:  # Condition for termination
                terminated = True
            total_reward += reward

            if terminated:
                break  # exit loop if game is naturally over

        if terminated:
            self.reset()

        # Ensure that the environment returns observation, reward, terminated, truncated, and info
        return new_state, total_reward / self.frame_repeats, terminated, truncated, info

    def apply_action_to_agent(self, action):
        #auto aim
        if self.game.wave_manager.enemies:
            enemies_with_distances = [(enemy, self.distance_to_player_squared(enemy))
                                      for enemy in self.game.wave_manager.enemies]
            sorted_enemies = sorted(enemies_with_distances, key=lambda x: x[1])[:self.NUM_ENEMIES_LISTED]
            closest_enemy = sorted_enemies[0][0]  # Extracting the enemy object of the closest enemy
            mouse_x = closest_enemy.rect.centerx  # Normalizing the x-coordinate
            mouse_y = closest_enemy.rect.centery # Normalizing the y-coordinate
        else:
            mouse_x, mouse_y = 0,0
        keys = self.player.action_to_keys(action)
        if mouse_x or mouse_y != 0:
            mouse_buttons = (True, False, False)
        else:
            mouse_buttons = (False, False, False)

        self.game.all_bullets.extend(
            self.player.move((keys, mouse_buttons, mouse_x, mouse_y), self.delta_time))

        self.game.all_bullets.extend(
            self.player.fire((keys, mouse_buttons, mouse_x, mouse_y), self.delta_time))

    #TODO add reward for hitting enemy if agent aims
    def _get_reward(self):

        # ex: if 60 fps, lose constant * 60 hp, so in graph, the avg when dying once per sec, is 1
        health_scaling_factor = 1.0 * (1 / self.avg_delta_time) / self.player.max_health

        # Calculate reward for agent
        reward = 0

        # Penalize the agent for losing health
        if self.player.health < self.prev_health:
            reward -= min(self.player.max_health, (self.prev_health - self.player.health)) * health_scaling_factor

        return reward

    def get_state(self):
        state = np.full((self.TOTAL_ATTRIBUTES,), 0, dtype=np.float32)

        # player information
        player_pos_screen = self.player.camera.apply(self.player.rect).center
        state[:self.PLAYER_ATTRIBUTES] = [player_pos_screen[0] / SCREEN_WIDTH, player_pos_screen[1] / SCREEN_HEIGHT]

        # Calculate distances for enemies
        enemies_with_distances = [(enemy, self.distance_to_player_squared(enemy))
                                  for enemy in self.game.wave_manager.enemies]

        # Sort enemies by distance and take the 10 closest
        sorted_enemies = sorted(enemies_with_distances, key=lambda x: x[1])[:self.NUM_ENEMIES_LISTED]
        for i, (enemy, _) in enumerate(sorted_enemies):
            # Before adding enemies to the state
            if self.player.camera.in_view(enemy):
                enemy_pos_screen = self.player.camera.apply(enemy.rect).center
                enemy_id_one_hot = [0] * 6
                enemy_id_one_hot[enemy.id - 1] = 1
                state[self.PLAYER_ATTRIBUTES + self.ENEMY_ATTRIBUTES * i:
                      self.PLAYER_ATTRIBUTES + self.ENEMY_ATTRIBUTES * (i + 1)] = [
                    enemy_pos_screen[0] / SCREEN_WIDTH,
                    enemy_pos_screen[1] / SCREEN_HEIGHT,
                    # enemy.speed / MAX_ENEMY_SPEED,
                    *enemy_id_one_hot,
                    ]
        # enemy.speed / MAX_ENEMY_SPEED
        # Calculate distances for bullets, excluding those with bullet_path == 'swirl_function'
        bullets_with_distances = [(bullet, self.distance_to_player_squared(bullet))
                                  for bullet in self.game.all_bullets if
                                  bullet.creator != 'player' and bullet.bullet_path != 'spiral_function']

        # Sort bullets by distance and take the NUM_BULLETS_LISTED closest
        sorted_bullets = sorted(bullets_with_distances, key=lambda x: x[1])[:self.NUM_BULLETS_LISTED]
        # print(sorted_bullets)
        for i, (bullet, _) in enumerate(sorted_bullets):
            if self.player.camera.in_view(bullet):
                bullet_pos_screen = self.player.camera.apply(bullet.rect).center
                bullet_id_one_hot = [0] * 9
                bullet_id_one_hot[bullet.id - 1] = 1
                state[
                self.PLAYER_ATTRIBUTES + self.ENEMY_ATTRIBUTES * self.NUM_ENEMIES_LISTED + self.BULLET_ATTRIBUTES * i:
                self.PLAYER_ATTRIBUTES + self.ENEMY_ATTRIBUTES * self.NUM_ENEMIES_LISTED + self.BULLET_ATTRIBUTES * (
                        i + 1)] = [
                    bullet_pos_screen[0] / SCREEN_WIDTH,
                    bullet_pos_screen[1] / SCREEN_HEIGHT,
                    *bullet_id_one_hot,
                    bullet.dx * bullet.speed / MAX_BULLET_SPEED,
                    bullet.dy * bullet.speed / MAX_BULLET_SPEED,
                    bullet.speed / MAX_BULLET_SPEED,]

        # clip state values to the range [-1, 1], excluding -2
        state = np.where(state == -2, state, np.clip(state, -1, 1))

        # check that the state values are all in the expected range
        for i, value in enumerate(state):
            if i < self.PLAYER_ATTRIBUTES:
                attribute_type = "Player"
            elif i < self.PLAYER_ATTRIBUTES + self.ENEMY_ATTRIBUTES * self.NUM_ENEMIES_LISTED:
                attribute_type = "Enemy"
            elif i < self.TOTAL_ATTRIBUTES:
                attribute_type = "Bullet"
            else:
                attribute_type = "Unknown"

            if value < -1 or value > 1:
                if value != -2:
                    print(f"Warning: {attribute_type} state value at index {i} is out of expected range: {value}")
        # print("New state:", state)
        return state

    def distance_to_player_squared(self, obj):
        dx = self.player.rect.centerx - obj.rect.centerx
        dy = self.player.rect.centery - obj.rect.centery
        return dx ** 2 + dy ** 2


class RewardLoggerCallback(BaseCallback):
    def __init__(self, print_freq: int, save_freq: int, trial: optuna.Trial=None, checkpoints=[1_000_000], agent: PPO = None,
                 model_file: str = None, steps_file: str = None, avg_reward_file: str = None, should_plot_rewards=False, patience=50000):
        super(RewardLoggerCallback, self).__init__()
        #TODO add default values and option to not print/save, just use high freq in meantime
        self.print_freq = print_freq
        self.save_freq = save_freq
        self.agent = agent
        self.should_plot_rewards = should_plot_rewards

        # Load or initialize the necessary attributes
        self.total_steps = self._load_attribute(steps_file, 0)
        self.avg_rewards = self._load_attribute(avg_reward_file, [])


        self.model_file = model_file
        self.steps_file = steps_file
        self.avg_reward_file = avg_reward_file
        self.rewards = []
        self.highest_reward = -np.inf
        self.last_improvement_step = 0
        # if not improved in patience time steps, end learning
        self.patience = patience
        self.trial = trial
        self.checkpoints = checkpoints

    def _load_attribute(self, file_name, default_value):
        if file_name is None:
            return default_value

        try:
            with open(file_name, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return default_value

    def _save_attribute(self, file_name, value):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'wb') as f:
            pickle.dump(value, f)

    def _on_step(self):
        self.total_steps += 1
        reward = self.locals.get('rewards')
        if reward is not None:
            self.rewards.extend(reward)  # Extend if 'reward' is a list
        if len(self.rewards) > 0 and self.num_timesteps % self.print_freq == 0:
            avg_reward = np.mean(self.rewards) if self.rewards else 0  # Handle empty rewards
            print(f"t={self.total_steps}: avg_reward={avg_reward:.4f}")
            self.rewards.clear()
            self.avg_rewards.append(avg_reward)

            # Use a rolling average of the past N episodes
            N = min(len(self.avg_rewards), 50)
            rolling_avg_reward = float(np.mean(self.avg_rewards[-N:]))

            if rolling_avg_reward > self.highest_reward:
                self.highest_reward = rolling_avg_reward
                self.last_improvement_step = self.total_steps
                if self.num_timesteps % self.save_freq == 0:
                    self._save_progress()
            if self.total_steps in self.checkpoints and self.trial:
                self.trial.report(rolling_avg_reward, step=self.total_steps)
                # if self.trial.should_prune():
                #     raise optuna.TrialPruned()



        # Check if it has improved in the patience time steps
        if self.total_steps - self.last_improvement_step > self.patience:
            print(f"No improvement in the last {self.patience} steps. Stopping early.")
            return False

        return True

    def _save_progress(self):
        self.agent.save(self.model_file)
        self._save_attribute(self.steps_file, self.total_steps)
        self._save_attribute(self.avg_reward_file, self.avg_rewards)
        print('saved')

    def plot_rewards(self):
        if not self.should_plot_rewards:
            return

        if len(self.avg_rewards) == 0:
            print("Warning: No rewards data available to plot.")
            return
        # Generate a list of time steps corresponding to the average rewards
        time_steps = list(range(self.print_freq, self.print_freq * len(self.avg_rewards) + 1, self.print_freq))

        # print(f"time_steps: {time_steps}, self.avg_rewards: {self.avg_rewards}")

        # Calculate the moving average of the rewards
        window_size = 20
        moving_avg_rewards = [np.mean(self.avg_rewards[i - window_size + 1:i + 1]) for i in
                              range(window_size - 1, len(self.avg_rewards))]

        # Generate a new list of time steps corresponding to the moving average
        moving_avg_time_steps = time_steps[window_size - 1:]

        # Fit a linear regression model to the moving average rewards
        coeffs = np.polyfit(moving_avg_time_steps, moving_avg_rewards, 1)
        slope, intercept = coeffs[0], coeffs[1]
        print(f"slope {slope} intercept {intercept}")
        linear_fit_rewards = [slope * t + intercept for t in moving_avg_time_steps]
        plt.plot(time_steps, self.avg_rewards, label='Average Reward')
        plt.plot(moving_avg_time_steps, moving_avg_rewards, label='Moving Average Reward')
        plt.plot(moving_avg_time_steps, linear_fit_rewards, label='Linear Fit')
        plt.xlabel('Total Steps')
        plt.ylabel('Reward')
        plt.annotate(f'slope: {slope:.2f}\nintercept: {intercept:.2f}',
                     xy=(0.05, 0.95), xycoords='axes fraction', verticalalignment='top')
        plt.legend()
        plt.show()


class AgentTrainer:
    def __init__(self, agent_type, iteration, study_name=None, plotting=0, rendering=0):
        self.agent_type = agent_type
        self.iteration = iteration
        study_str = f"{study_name}_" if study_name else ""
        iteration_str = f"{iteration}"

        self.model_file = f"agents/{agent_type}_{study_str}{iteration_str}.zip"
        self.steps_file = f"steps/steps_{(study_str[:-1] + '_') if study_name else ''}{iteration_str}.pkl"
        self.avg_reward_file = f"avg_rewards/avg_rewards_{(study_str[:-1] + '_') if study_name else ''}{iteration_str}.pkl"
        self.plotting = plotting
        self.rendering = rendering

        # Create necessary directories
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.steps_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.avg_reward_file), exist_ok=True)



    def load_pickle(self, file_name):
        try:
            with open(file_name, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}

    def create_envs(self, game, delta_time=.01666, frame_repeats=4):
        return [DummyVecEnv([lambda: SingleAgentEnvironment(game, i + bool(
            hasattr(game, 'human_player') and game.human_player in game.players), control_clock=(i == 0), should_render=self.rendering, delta_time=delta_time, frame_repeats=frame_repeats)])
                for i in range(game.num_of_ai_players_at_start)]

    def load_agent(self, env):
        if os.path.isfile(self.model_file):
            print("Using existing agent")
            model_type = self.model_file.split('/')[-1].split('_')[0].upper()
            if model_type == "PPO":
                return PPO.load(self.model_file, env=env)
            elif model_type == "SAC":
                return SAC.load(self.model_file, env=env)
            else:
                print("Unknown model type:", model_type)
        else:
            print("Creating new agent")
            policy_kwargs = dict(
                net_arch=dict(
                    pi=[64, 64],
                    vf=[64, 64]
                )
            )
            # return PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, learning_rate=0.0003)
            return None


class StudyManager:
    def __init__(self, study_name, min_resources_value=200_000):
        self.study_name = study_name
        self.study = None
        # time steps before pruner can prune trial
        self.min_resources_value = min_resources_value

    def objective(self, trial, dry_run=False):
        # todo change back to 5 or some other number, just want to try 1 to try to get full study done
        num_attempts = 1
        all_rewards = []
        if dry_run:
            # Dry-run code (simulate the setup and return a random value)
            game = Game(rendering=False)
            start_time = time.time()
            agent_trainer = AgentTrainer(agent_type="ppo_agent", study_name=self.study_name,
                                         iteration="trial_" + str(trial.number))
            env = agent_trainer.create_envs(game, delta_time=0.01666, frame_repeats=4)[0]

            # Create the agent using the define_model function
            agent = self.define_model(trial, env)
            end_time = time.time()
            print(f"Dry-run Trial {trial.number} completed in {end_time - start_time} seconds")
            return random.random()
        else:
            for attempt in range(num_attempts):
                # create the game and environment
                game = Game(rendering=False)
                agent_trainer = AgentTrainer(agent_type="ppo_agent", study_name=self.study_name,
                                             iteration=f"trial_{trial.number}_attempt_{attempt}")
                env = agent_trainer.create_envs(game, delta_time=0.01666,frame_repeats=4)[0]

                # Create the agent using the define_model function
                agent = self.define_model(trial, env)

                total_timesteps = 5_000_000
                evaluation_interval = 10000

                optuna_callback = RewardLoggerCallback(print_freq=1000,
                                                       save_freq=10000,
                                                       agent=agent,
                                                       model_file=agent_trainer.model_file,
                                                       steps_file=agent_trainer.steps_file,
                                                       avg_reward_file=agent_trainer.avg_reward_file,
                                                       should_plot_rewards=False,
                                                       patience=500_000, trial=trial)

                # Training loop
                agent.learn(total_timesteps=total_timesteps, callback=optuna_callback)

                # Collect the rewards
                print(f"Attempt {attempt + 1} - all rewards in logger.avg_rewards:", optuna_callback.avg_rewards)
                # Calculate the best average group within the collected rewards
                N = 50  # Window size for the moving average
                moving_avgs = [np.mean(optuna_callback.avg_rewards[i:i + N])
                               for i in range(len(optuna_callback.avg_rewards) - N + 1)]
                best_avg_group = max(moving_avgs)
                print(f"Attempt {attempt + 1} - best average group in logger.avg_rewards: {best_avg_group:.4f}")
                all_rewards.append(best_avg_group)

            print(f"Reward for all attempts: {np.mean(all_rewards)}")
            # Return the average reward over all attempts
            return np.mean(all_rewards)

    def define_model(self, trial, env):
        gamma = trial.suggest_float('gamma', 0.98, 0.999)
        gae_lambda = trial.suggest_float('gae_lambda', 0.91, 0.97)
        initial_layer_size = trial.suggest_int('initial_layer_size', 64, 512)
        num_layers = trial.suggest_int('num_layers', 1, 5)
        total_percent_decrease = trial.suggest_float('total_percent_decrease', 0, 0.8)
        # divide the total percent decrease by the number of layers to get the decrease per layer
        percent_decrease_per_layer = total_percent_decrease / num_layers
        clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
        # suggest batch_size from factors of 2048
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512, 1024, 2048])
        n_epochs = trial.suggest_int('n_epochs', 1, 20)
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.0009, log=True)

        # Compute the sizes of all layers based on the initial size and percent decrease
        layer_sizes = [initial_layer_size]
        for _ in range(1, num_layers):
            next_layer_size = int(layer_sizes[-1] * (1 - percent_decrease_per_layer))
            # Ensure the layer size is never 0
            next_layer_size = max(next_layer_size, 1)
            layer_sizes.append(next_layer_size)

        policy_kwargs = dict(
            net_arch=dict(
                pi=layer_sizes,
                vf=layer_sizes
            )
        )
        agent = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate
        )
        return agent


    def load_or_create_study(self):
        storage_path = f"sqlite:///studys/{self.study_name}.db"
        
        pruner = optuna.pruners.NopPruner
        self.study = optuna.create_study(study_name=self.study_name, storage=storage_path, direction='maximize',
                                         pruner=pruner, load_if_exists=True)
        print("Study loaded or created in local SQLite database.")

    def save_study(self):
        with open(f"{self.study_name}.pkl", "wb") as f:
            pickle.dump(self.study, f)

    def get_study(self):
        return self.study

    def evaluate_agent(self, agent, env, num_episodes=10, num_steps=200000, print_interval=10000):
        all_episode_rewards = []  # A list to hold the rewards for each episode
        for episode in range(num_episodes):
            print('Starting episode:', episode)
            obs = env.reset()
            obs = obs[0]  # Extract the array from the tuple

            episode_rewards = []  # A list to hold the rewards for this episode
            for step in range(num_steps):
                action, _ = agent.predict(obs)
                obs, reward, done, _, _ = env.step(action)
                episode_rewards.append(reward)  # Store the reward for this step

                if step % print_interval == 0 and step > 0:  # Print every print_interval time steps
                    avg_reward_so_far = np.mean(episode_rewards)
                    print(f'Episode {episode} average reward is {avg_reward_so_far} at {step} time_steps')

            all_episode_rewards.append(np.mean(episode_rewards))
            print(f'Ending episode {episode} with average reward: {np.mean(episode_rewards)}')

        mean_reward = np.mean(all_episode_rewards)
        std_dev_reward = np.std(all_episode_rewards)
        return mean_reward, std_dev_reward

    def evaluate_hyperparameters(self, trial, env, num_train_steps=200000, eval_interval=100000,
                                 num_training_trials=3, eval_episodes=0, eval_steps=20000, print_interval=10000):

        all_training_rewards = []

        for eval_trial in range(num_training_trials):
            print(f'Starting training trial {eval_trial}')
            agent = self.define_model(trial, env)
            reward_logger = RewardLoggerCallback(print_freq=1000, save_freq=99999999999999, )

            for step in range(0, num_train_steps, eval_interval):
                # Train the agent for eval_interval steps
                agent.learn(total_timesteps=eval_interval, callback=reward_logger)

                # TODO if using model eval, make sure to copy the agent or change the callback, so the logger doesn't carry over
                if eval_episodes > 0:
                    # Evaluate the agent
                    mean_reward, std_dev_reward = self.evaluate_agent(agent, env, num_episodes=eval_episodes,
                                                                      num_steps=eval_steps)

                    # Print and store the evaluation results
                    print(
                        f'Evaluation at step {step + eval_interval}: Mean reward over {eval_episodes} episodes: {mean_reward}')

                # Print progress within each episode
                if step % print_interval == 0:
                    print(
                        f'Training trial {eval_trial}, step {step}: Mean reward so far: {np.mean(reward_logger.avg_rewards)}')

            print("all rewards in logger.avg_rewards:", reward_logger.avg_rewards)
            print("last 5 rewards in logger.avg_rewards:", reward_logger.avg_rewards[-5:])
            avg_reward_last_5 = np.mean(reward_logger.avg_rewards[-5:])
            all_training_rewards.append(avg_reward_last_5)

        return all_training_rewards
