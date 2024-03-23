import argparse
from pathlib import Path

from frankenstein.components.skills.management import Management


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


if __name__ == "__main__":
    management = Management()
    yml_str = load_config_from_yaml()
    management.launch(yml_str)
