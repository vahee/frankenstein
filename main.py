import argparse
from pathlib import Path
import asyncio as aio
import sys
from frankenstein.components.skills.management import Management
from agentopy import State

def load_config_from_yaml() -> str:
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-c", "--config", default=".config.yaml", help="Path to the config file")
    args, _ = argparser.parse_known_args()
    
    config_path = Path(args.config)
    
    if not config_path.exists():
        raise Exception("Config file does not exist")

    with open(config_path, "r", encoding="utf-8") as stream:
        return stream.read()


async def run(yml_str: str):
    management = Management()
    try:
        tasks = await management.start_tasks(config_str=yml_str)
        await aio.wait(tasks, return_when=aio.FIRST_EXCEPTION)
    except Exception as e:
        print(e)
        sys.exit()

if __name__ == "__main__":
    aio.run(run(load_config_from_yaml()))
