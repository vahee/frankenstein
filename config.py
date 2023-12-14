from dataclasses import dataclass
import argparse
from pathlib import Path
from typing import Optional, List
import yaml
from dacite import from_dict


@dataclass
class AgentConfig:
    """Defines the configuration for an agent"""
    system_prompt: str
    response_template: str
    language_model: str
    embeddings_model: str


@dataclass
class MyConfig:
    """Defines the configuration for the user"""
    email: str
    first_name: str
    last_name: str
    location: str


@dataclass
class GlobalConfig:
    """Defines the global configuration"""
    openai_api_key: Optional[str]
    hf_access_token: Optional[str]
    email_imap_address: Optional[str]
    email_imap_port: Optional[int]
    email_smtp_address: Optional[str]
    email_smtp_port: Optional[int]
    email_login: Optional[str]
    email_password: Optional[str]
    email_from: Optional[str]
    remote_control_websocket_host: Optional[str]
    remote_control_websocket_port: Optional[int]
    remote_control_websocket_ping_interval: Optional[int]
    remote_control_websocket_ping_timeout: Optional[int]
    messenger_websocket_host: Optional[str]
    messenger_websocket_port: Optional[int]
    messenger_websocket_ping_interval: Optional[int]
    messenger_websocket_ping_timeout: Optional[int]
    me: MyConfig
    agent: AgentConfig
    serper_api_key: Optional[str]
    outbound_emails_whitelist: Optional[List[str]]
    testing_email_outbound: Optional[str]
    testing_email_inbound: Optional[str]


agent = AgentConfig(
    language_model="openai:gpt-4-1106-preview",  # "google:chat-bison@001"
    # "google:textembedding-gecko@001",
    # "openai:text-embedding-ada-002"
    # "st:all-MiniLM-L6-v2",
    embeddings_model="st:all-MiniLM-L6-v2",
    system_prompt="""You are an autonomous AI assistant, your name is {name}.
You operate in an environment where you have access to various components of it. For example Email, Web Browser, Todo List, etc.
You can make observations about the environment, and you can take actions to interact with the environment.
Your goal is to continuously monitor the environment, and take actions that benefit me most.

I am your user, my name is {my_first_name} {my_last_name}, I live in {my_location}, and my email is {my_email}.

You must always follow these principles:
- Your actions should be only from the list of available actions, do not try to take any other action, all responses should be in JSON format!
- Be proactive and creative. Do not wait for me to tell you what to do, try to figure out what to do next by yourself.
- Carefully check the environment for anything that deserves attention.
- Set tasks for yourself based on your observations.
- Operate autonomously, do not ask me for permission to do anything or for any inputs.
- Make sure to check the todo list for any tasks that I have set for you, and complete them.
- Stay idle if there is no reasonable thing to do, do not do anything just for the sake of doing something.
- Make sure to remove tasks from the todo list once you have completed them.
- Do not make assumptions, you don't know anything unless you have observed it.
- Work each task out step by step.

The following is the list of available actions, you can action ONLY from this list, nothing else! 
""",
    response_template="""{query}

Current time: {current_time}

Respond in JSON with the following format:
{{
    "thoughts": "text describing the thought process on what is the next most reasonable action to take and why",
    "criticism": "text describing constructive criticism of the thought process",
    "task": "text describing the task that you are currently working on",
    "plan": "text describing the sequence of steps that you plan to take next, including the conclusion of the task completion",
    "action": "text describing the name of the next action to take, it can be only an action from the defined list of actions above!",
    "args": {{
        "argument": "value"
    }}
}}"""
)


# TODO: UGLY, FIX THIS
argparser = argparse.ArgumentParser()
argparser.add_argument(
    "-config", default=Path(".config.yaml"), help="Path to the config file")
args, _ = argparser.parse_known_args()
config_path = Path(args.config)
if not config_path.exists():
    raise Exception("Config file does not exist")

with open(config_path, "r", encoding="utf-8") as stream:
    try:
        config_dict = yaml.safe_load(stream)
        config_dict['agent'] = agent
        CONFIG = from_dict(data_class=GlobalConfig, data=config_dict)
    except yaml.YAMLError as exc:
        raise Exception("Failed to load config file") from exc
