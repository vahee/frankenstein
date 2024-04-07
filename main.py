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
        await management.launch(config_str=yml_str, caller_context=State())
    except Exception as e:
        print(e)
        sys.exit()
    while True:
        await aio.sleep(1)
    

if __name__ == "__main__":
    aio.run(run(load_config_from_yaml()))
