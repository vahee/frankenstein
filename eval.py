from train import configure, evaluate_policy, model, eval_env, env

stats = {}

def callback(locals, globals) -> bool:
    global stats
    reward = locals["reward"]
    observation = locals["observations"]
    env = locals["env"]
    action = locals["actions"][0]
    stats[action] = stats.get(action, 0) + 1
    env_stats = env.env_method("get_stats")
    print(f"Action: {action}, Reward: {round(reward, 2)}, Stats: {env_stats}")
    
if __name__ == "__main__":

    model.load("dqn_policy")
    new_logger = configure(None, ["stdout", "log", "csv"])
    model.set_logger(new_logger)

    print(evaluate_policy(model, eval_env, n_eval_episodes=1, return_episode_rewards=True, callback=callback, render=True, deterministic=False))
    print(stats)
