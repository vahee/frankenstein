import traceback
import asyncio as aio
import logging
import sys

from typing import Tuple, Dict
    
from agentopy import IEnvironmentComponent, WithActionSpaceMixin, WithStateMixin, Action, EntityInfo, Agent, Environment, IAgent, IEnvironment, ActionResult

from frankenstein.lib.language.openai_language_models import OpenAIChatModel
from frankenstein.lib.language.embedding_models import OpenAIEmbeddingModel, SentenceTransformerEmbeddingModel
from frankenstein.policies.llm_policy import LLMPolicy
from frankenstein.lib.db.in_memory_vector_db import InMemoryVectorDB
from frankenstein.components import TodoList, Creativity, Email, Memory, WebBrowser, Messenger, RemoteControl
from frankenstein.lib.networking.communication import WebsocketMessagingServer
from frankenstein.lib.language.protocols import ILanguageModel

logging.basicConfig(handlers=[logging.FileHandler('frank.log', mode='w', encoding='utf-8'),
                    logging.StreamHandler(sys.stdout)],
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logging.info("Running Dr. Frankenstein")

logger = logging.getLogger('main')

class Management(WithStateMixin, WithActionSpaceMixin, IEnvironmentComponent):
    """Implements a creativity environment component"""

    def __init__(self):
        """Initializes the creativity environment component"""
        super().__init__()
        
        self.action_space.register_actions(
            [
                Action(
                    "agents", "get the list of running agents.", self.agents),
                Action(
                    "launch_agent", "launch a new agent", self.launch_agent),
                Action("kill_agent", "kill an agent", self.kill_agent)
            ])
        
    async def agents(self) -> ActionResult:
        """Creates text content"""
        return ActionResult(value="List of agents", success=True)

    async def launch_agent(self, agent_config: Dict) -> ActionResult:
        """Creates text content"""
        return ActionResult(value="Agent launched", success=True)
    
    async def kill_agent(self, agent_id: str) -> ActionResult:
        """Creates text content"""
        return ActionResult(value="Agent killed", success=True)

    def create_env_and_agent(self, config: Dict) -> Tuple[IEnvironment, IAgent]:
    
        environment_components = []
        agent_components = []
        
        for component_name, component_config in config.get("agent_components", {}).items():
            agent_components.append(self.create_component(component_name, component_config))
        
        for component_name, component_config in config.get("environment_components", {}).items():
            environment_components.append(self.create_component(component_name, component_config))
        
        
        policy = self.create_policy(config)

        env = Environment(environment_components)

        agent = Agent(policy, env, agent_components)

        return env, agent

    def create_component(self, component_name: str, component_config: Dict):
        """
        Creates a component based on the configuration
        """
        if component_name == "todo_list":
            return TodoList()
        if component_name == "creativity":
            assert component_config.get("language_model") is not None, "Language model is not set"
            language_model = self.create_language_model(component_config["language_model"])
            return Creativity(language_model)
        if component_name == "email":
            try:
                return Email(**component_config)
            except Exception as e:
                raise Exception(f"Failed to create email component: {e}")
        if component_name == "memory":
            assert component_config.get("embedding_model") is not None, "Embedding model is not set"
            embedding_model = self.create_embegging_model(component_config["embedding_model"])
            assert component_config.get("db", {}).get("implementation") in ["in_memory_vector_db"], "DB is not set or not supported"
            if component_config["db"]["implementation"] == "in_memory_vector_db":
                db = InMemoryVectorDB(embedding_model.dim(), component_config["db"]["params"]["affinity"])
            
            assert component_config.get("memory_size"), "Memory size is not set or is 0"
            
            return Memory(db, embedding_model, component_config["memory_size"])
        if component_name == "web_browser":
            assert component_config.get("language_model") is not None, "Language model is not set"
            language_model = self.create_language_model(component_config["language_model"])
            search_api = component_config.get("search_api")
            assert search_api in ["ddg", "serper"], "Search API is not set"
            if search_api == "serper":
                serper_api_key = component_config.get("serper_api_key")
                assert serper_api_key is not None, "Serper API key is not set"
            return WebBrowser(language_model, search_api, serper_api_key)
        if component_name == "messenger":
            assert component_config.get("messaging", {}).get("implementation") in ["websocket"], "Messaging is not set or not supported"
            if component_config["messaging"]["implementation"] == "websocket":
                params = component_config["messaging"].get("params", {})
                try:
                    messaging = WebsocketMessagingServer(**params)
                except Exception as e:
                    raise Exception(f"Failed to create messaging server: {e}")
            return Messenger(messaging)
        if component_name == "remote_control":
            assert component_config.get("messaging", {}).get("implementation") in ["websocket"], "Messaging is not set or not supported"
            if component_config["messaging"]["implementation"] == "websocket":
                params = component_config["messaging"].get("params", {})
                try:
                    messaging = WebsocketMessagingServer(**params)
                except Exception as e:
                    raise Exception(f"Failed to create messaging server: {e}")
            return RemoteControl(messaging)
        raise Exception(f"Component {component_name} is not supported")

    def create_policy(self, config: Dict):
        policy_name = config.get("policy", {}).get("implementation")
        assert policy_name in ["llm_policy", "trading_policy"], "Policy is not set or not supported"
        if policy_name == "llm_policy":
            params = config["policy"].get("params", {})
            assert params.get("language_model") is not None, "Language model is not set"
            system_prompt = params.get("system_prompt")
            assert system_prompt is not None, "System prompt is not set"
            response_template = params.get("response_template")
            policy = LLMPolicy(
                self.create_language_model(params["language_model"]),
                system_prompt,
                response_template
            )
            return policy
        raise Exception("Policy is not set or not supported")
        

    def create_language_model(self, config: Dict) -> ILanguageModel:
        """
        Returns the language model based on the configuration
        """
        assert config.get("implementation") in ["openai"], "Language model is not set or not supported"
        if config["implementation"] == "openai":
            api_key = config.get("params", {}).get("api_key")
            assert api_key is not None, "OpenAI API key is not set"
            model_name = config.get("params", {}).get("model_name")
            assert model_name is not None, "OpenAI model name is not set"
            format_json = config.get("params", {}).get("format_json", False)
            return OpenAIChatModel(api_key, model_name, json=format_json)
        raise Exception("Language model is not set or not supported")
            

    def create_embegging_model(self, config: Dict):
        """
        Returns the embeddings model based on the configuration
        """
        assert config.get("implementation") in ["openai", "st"], "Embeddings model is not set or not supported"
        if config["implementation"] == "openai":
            access_token = config.get("params", {}).get("access_token")
            assert access_token is not None, "OpenAI API key is not set"
            model_name = config.get("params", {}).get("model_name")
            assert model_name is not None, "OpenAI model name is not set"
            return OpenAIEmbeddingModel(access_token, model_name)
        if config["implementation"] == "st":
            access_token = config.get("params", {}).get("access_token")
            assert access_token is not None, "Hugging Face access token is not set"
            model_name = config.get("params", {}).get("model_name")
            assert model_name is not None, "Model name is not set"
            return SentenceTransformerEmbeddingModel(model_name, access_token)
        raise Exception("Embeddings model is not set or not supported")

    def launch(self, config: Dict):
        """
        Main function of the project. It is responsible for running the assitant.
        """
        logger.info("Starting environment and agent")
        async def run():
            env, agent = self.create_env_and_agent(config)
            e = await aio.wait([env.start(), agent.start()], return_when=aio.FIRST_EXCEPTION)
            expection = e[0].pop().exception()
            print(''.join(traceback.format_tb(
                expection.__traceback__)))
            print(expection)
        aio.run(run())

    async def on_tick(self) -> None:
        self._state.set_item("status", "Management skills ready.")

    def info(self) -> EntityInfo:
        return EntityInfo(
            name="Management",
            version="0.1.0",
            params={}
        )
