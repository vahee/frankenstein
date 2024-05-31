from train import configure, evaluate_policy, model, eval_env

if __name__ == "__main__":

    model.load("dqn_policy")
    new_logger = configure(None, ["stdout", "log", "csv"])
    model.set_logger(new_logger)

    print(evaluate_policy(model, eval_env, n_eval_episodes=1, return_episode_rewards=True, render=True))
