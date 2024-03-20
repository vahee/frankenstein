import argparse
from pathlib import Path
import yaml

from frankenstein.components.skills.management import Management


def load_config_from_yaml():
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-c", "--config", default=".config.yaml", help="Path to the config file")
    args, _ = argparser.parse_known_args()
    
    config_path = Path(args.config)
    print(config_path)
    if not config_path.exists():
        raise Exception("Config file does not exist")

    with open(config_path, "r", encoding="utf-8") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise Exception("Failed to load config file") from exc


if __name__ == "__main__":
    management = Management()
    cfg = load_config_from_yaml()
    management.launch(cfg)
