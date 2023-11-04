import cProfile
import os
import pstats
import pygame
import torch
from stable_baselines3 import PPO
import numpy as np
import pickle
import optuna
from LeadLabyrinth.ai import RewardLoggerCallback, AgentTrainer, StudyManager
from LeadLabyrinth.game import Game

def main():

    try:
        # TODO create UI or some better method of control
        manually_training = 0
        optuna_tuning = 1
        model_testing = 0
        playing = 0

        if manually_training:
            rendering = 1
            plotting = 1
            # makes agent trainer
            agent_trainer = AgentTrainer(agent_type="ppo_agent", iteration="130", plotting=plotting, rendering=rendering)
            # agent_trainer = AgentTrainer(agent_type="ppo_agent",study_name="study_2",iteration="18", plotting=1, rendering=1)
            # agent_trainer = AgentTrainer(agent_type="ppo_agent", study_name="study_27", iteration ="trial_0_attempt_0", plotting=1, rendering=rendering)
            print(agent_trainer.model_file)
            print(agent_trainer.steps_file)
            print(agent_trainer.avg_reward_file)
            game = Game(rendering=rendering)  # Create a game for the env
            # env = agent_trainer.create_envs(game, delta_time=(0.01666))[0]  # first env, just one agent
            # env = agent_trainer.create_envs(game, delta_time=(0.014285714,0.025))[0]  # first env, just one agent
            env = agent_trainer.create_envs(game, delta_time=(0.01666), frame_repeats=4)[0]  # first env, just one agent
            # env = agent_trainer.create_envs(game, delta_time=(0.05))[0]  # first env, just one agent
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
            # agent.save(agent_trainer.model_file)
        if optuna_tuning:
            study_manager = StudyManager(study_name="study_33")
            study_manager.load_or_create_study()
            study = study_manager.get_study()
            # fig = optuna.visualization.plot_optimization_history(study)
            # fig.show()
            # fig = optuna.visualization.plot_param_importances(study)
            # fig.show()
            # fig = optuna.visualization.plot_slice(study)
            # fig.show()
            study.optimize(study_manager.objective, n_trials=500)
    finally:
        if optuna_tuning:
            print("Study Statistics:")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ",
                  sum(1 for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED))
            print("  Number of complete trials: ",
                  sum(1 for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE))
            print("Best params: ", study.best_params)
            print("Best value (negative reward): ", study.best_value)
            print("Best Trial: ", study.best_trial)

            print("All Trials:")
            for trial in study.trials:
                avg_reward = trial.value
                params = trial.params
                print(f"  Trial {trial.number}: Params {params} Average Reward {avg_reward}")
            fig = optuna.visualization.plot_param_importances(study)
            fig.show()
            study_manager.save_study()  # Save the study after optimization
        if model_testing:
            study_name = "study_2"
            trial_number =18

            rewards_file = f"avg_rewards/avg_rewards{study_name}_trial_{trial_number}.pkl"

            # Load the rewards
            with open(rewards_file, "rb") as f:
                avg_rewards = pickle.load(f)

            # Create a RewardLoggerCallback instance with the loaded avg_rewards
            reward_logger = RewardLoggerCallback(print_freq=1000, save_freq=10000, should_plot_rewards=True)
            reward_logger.avg_rewards = avg_rewards

            # Plot the rewards using the existing method
            reward_logger.plot_rewards()

            study_name = "study_2"
            study_manager = StudyManager(study_name=study_name)
            study_manager.load_or_create_study()
            study = study_manager.get_study()
            parameters_to_assess = ['learning_rate', 'n_layers', 'n_neurons_layer_0', 'n_neurons_layer_1',
                                    'n_neurons_layer_2']
            fig = optuna.visualization.plot_param_importances(study)
            fig.show()
            # Load the best agent
            game = Game(rendering=False)  # Create a game for the env
            agent_trainer = AgentTrainer(agent_type="ppo_agent", study_name=study_name, iteration="_best_agent")
            env = agent_trainer.create_envs(game)[0]
            print('best trial:', study.best_trial)
            print('best params:', study.best_params)

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
    pygame.quit()

if __name__ == "__main__":
    pr = cProfile.Profile()
    try:
        pr.enable()
        main()
    finally:
        pr.disable()
        stats = pstats.Stats(pr)
        stats.sort_stats('tottime')  # Sort by total time
        stats.print_stats()
