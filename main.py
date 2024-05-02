import cProfile
import os
import pygame
import pstats
import torch
import numpy as np
import pickle
import optuna
from stable_baselines3 import PPO
from ai import RewardLoggerCallback, AgentTrainer, StudyManager
from game import Game

def main():
    print(torch.cuda.is_available())

    try:
        # TODO create UI or some better method of control
        manually_training = 1
        optuna_tuning = 0
        model_testing = 0
        playing = 0

        if manually_training:
            rendering = 0
            plotting = 1
            # makes agent trainer
            agent_trainer = AgentTrainer(agent_type="ppo_agent", iteration="1", plotting=plotting, rendering=rendering)
            game = Game(rendering=rendering)  # Create a game for the env
            env = agent_trainer.create_envs(game, delta_time=(0.01666), frame_repeats=4)[0]  # first env, just one agent
            agent = agent_trainer.load_agent(env.envs[0])

            if agent is None:
                print("Creating new agent")
                policy_kwargs = dict(net_arch=dict(pi=[64,64], vf=[64,64]),)
                agent = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, learning_rate=0.0003)

            num_parameters = sum(p.numel() for p in agent.policy.parameters())
            print(f"param count : {num_parameters}")

            reward_logger = RewardLoggerCallback(print_freq=1000, save_freq=10000, patience=1000000000, agent=agent,
                                                 model_file=agent_trainer.model_file,
                                                 steps_file=agent_trainer.steps_file,
                                                 avg_reward_file=agent_trainer.avg_reward_file,
                                                 should_plot_rewards=agent_trainer.plotting)

            reward_logger.plot_rewards()
            agent.learn(total_timesteps=100_000_000, callback=reward_logger)

        if optuna_tuning:
            study_manager = StudyManager(study_name="study_1")
            study_manager.load_or_create_study()
            study = study_manager.get_study()
            study.optimize(study_manager.objective, n_trials=500)

        if model_testing:
            study_name = "study_1"
            trial_number = 1

            rewards_file = f"avg_rewards/avg_rewards{study_name}_trial_{trial_number}.pkl"

            # Load the rewards
            with open(rewards_file, "rb") as f:
                avg_rewards = pickle.load(f)

            # Create a RewardLoggerCallback instance with the loaded avg_rewards
            reward_logger = RewardLoggerCallback(print_freq=1000, save_freq=10000, should_plot_rewards=True)
            reward_logger.avg_rewards = avg_rewards

            reward_logger.plot_rewards()

            study_name = "study_1"
            study_manager = StudyManager(study_name=study_name)
            study_manager.load_or_create_study()
            study = study_manager.get_study()
            parameters_to_assess = ['learning_rate', 'n_layers', 'n_neurons_layer_0', 'n_neurons_layer_1',
                                    'n_neurons_layer_2']
            # fig = optuna.visualization.plot_param_importances(study)
            # fig.show()
            # Load the best agent
            game = Game(rendering=False)  # Create a game for the env
            agent_trainer = AgentTrainer(agent_type="ppo_agent", study_name=study_name, iteration="_best_agent")
            env = agent_trainer.create_envs(game)[0]

            # Evaluate the agent
            num_training_trials = 10
            all_rewards = study_manager.evaluate_hyperparameters(study.best_trial, env.envs[0], num_training_trials=num_training_trials)
            print(f'all rewards: ', all_rewards)
            print(f"Mean reward over {num_training_trials} episodes: {np.mean(all_rewards)}")
            print(f"Standard deviation of rewards: {np.std(all_rewards)}")

        # play game
        if playing:
            global running
            running = True
            game = Game(num_of_ai_players_at_start=0, has_human_player=True)
            while running:
                game.clock.tick()
    finally:
        pass
        if profiling:
            pr.disable()
            stats = pstats.Stats(pr)
            stats.sort_stats('tottime')  # Sort by total time
            stats.print_stats()

if __name__ == "__main__":
    profiling = False  # Set this to True to enable profiling
    if profiling:
        pr = cProfile.Profile()
        pr.enable()
        main()
        pr.disable()
        pr.dump_stats("profile_data.prof")  # Saves the profile data to a file
    else:
        main()
