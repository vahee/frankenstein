import sys
import logging
import traceback
import asyncio as aio
from multiprocessing import Process
from typing import Dict, List
import yaml

from agentopy import IAgentComponent, WithActionSpaceMixin, Action, EntityInfo, Agent, Environment, IAgent, IEnvironment, ActionResult, IState, State

from frankenstein.lib.language.openai_language_models import OpenAIChatModel
from frankenstein.lib.language.embedding_models import OpenAIEmbeddingModel, SentenceTransformerEmbeddingModel
from frankenstein.policies.llm_policy import LLMPolicy
from frankenstein.policies.human_controlled_policy import HumanControlledPolicy
from frankenstein.lib.db.in_memory_vector_db import InMemoryVectorDB
from frankenstein.components import TodoList, Creativity, Email, Memory, WebBrowser, Messenger, RemoteControl
from frankenstein.lib.networking.communication import WebsocketMessagingJsonServer
from frankenstein.lib.language.protocols import ILanguageModel

logging.basicConfig(handlers=[logging.FileHandler('frank.log', mode='w', encoding='utf-8'),
                    logging.StreamHandler(sys.stdout)],
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logging.info("Running Dr. Frankenstein")

logger = logging.getLogger('main')

class Management(WithActionSpaceMixin, IAgentComponent):
    """Implements a creativity environment component"""

    def __init__(self):
        """Initializes the creativity environment component"""
        super().__init__()
        
        self._agents = {}
        self._environments = {}
        self._next_agent_id = 1
        
        self.action_space.register_actions(
            [
                Action(
                    "agents", "get the list of running agents.", self.agents, self.info()),
                Action(
                    "environments", "get the list of running environments.", self.environments, self.info()),
                Action(
                    "launch", "launch a new agent or environment", self.launch, self.info()),
                Action("kill_agent", "kill an agent", self.kill_agent, self.info()),
                Action("destroy_environment", "kill an environment", self.destroy_environment, self.info())
            ])
    
    def __del__(self):
        for _, agent in self._agents.items():
            agent['process'].cancel()
        for _, environment in self._environments.items():
            environment['process'].cancel()
    
    async def agents(self, *, caller_context: IState) -> ActionResult:
        """Returns the list of running agents"""
        
        agents = {}
        
        for agent_id, agent in self._agents.items():
            agents[agent_id] = {
                'config_yaml': yaml.safe_dump(agent['config']),
                'config': agent['config']
            }
        
        return ActionResult(value=agents, success=True)
    
    async def environments(self, *, caller_context: IState) -> ActionResult:
        """Returns the list of running environments"""
        
        environments = {}
        
        for environment_id, environment in self._environments.items():
            environments[environment_id] = {
                'config_yaml': yaml.safe_dump(environment['config']),
                'config': environment['config']
            }
        
        return ActionResult(value=environments, success=True)

    async def launch(self, *, config_str: str, caller_context: IState) -> ActionResult:
        """launches an agent or environment"""
        
        config = yaml.safe_load(config_str)
        
        import traceback
        
        def print_result(task):
            logger.info(f"Result {list(task.result()[0])[0]}")
            err = traceback.format_exception(list(task.result()[0])[0].exception())
            logger.info(f"Exception {err}")
            
        for environment_config in config.get("environments", []):
            env = self.create_environment(environment_config)
            env_id = environment_config.get("id")
            task = aio.create_task(aio.wait(env.start(), return_when=aio.FIRST_EXCEPTION))
            task.add_done_callback(print_result)
            
            self._environments[env_id] = {
                'process': task,
                'config': environment_config,
                'env': env
            }
            
        for agent_config in config.get("agents", []):
            agent = self.create_agent(agent_config)
            agent_id = str(self._next_agent_id)
            self._next_agent_id += 1
            task = aio.create_task(aio.wait(agent.start(), return_when=aio.FIRST_EXCEPTION))
            task.add_done_callback(print_result)
            
            self._agents[agent_id] = {
                'process': task,
                'config': agent_config,
                'agent': agent
            }
        
        return ActionResult(value="OK", success=True)
    
    async def kill_agent(self, *, agent_id: str, caller_context: IState) -> ActionResult:
        """Creates text content"""
        if agent_id not in self._agents:
            return ActionResult(value="No such agent", success=False)
        try:
            self._agents[agent_id]['process'].cancel()
        except:
            pass
        del self._agents[agent_id]['agent']
        del self._agents[agent_id]['process']
        del self._agents[agent_id]
        
        return ActionResult(value="Agent killed", success=True)

    async def destroy_environment(self, *, environment_id: str, caller_context: IState) -> ActionResult:
        """Creates text content"""
        if environment_id not in self._environments:
            return ActionResult(value="No such environment", success=False)
        try:
            self._environments[environment_id]['process'].cancel()
        except:
            pass
        del self._environments[environment_id]['env']
        del self._environments[environment_id]['process']
        del self._environments[environment_id]
        
        return ActionResult(value="Environment killed", success=True)
    
    def create_environment(self, config: Dict) -> IEnvironment:
            
        environment_components = []
        
        for component_name, component_config in config.get("components", {}).items():
            environment_components.append(self.create_component(component_name, component_config))

        env = Environment(environment_components)

        return env
    
    def create_agent(self, config: Dict) -> IAgent:
        
        agent_components = []
        
        for component_name, component_config in config.get("components", {}).items():
            agent_components.append(self.create_component(component_name, component_config))
        
        policy = self.create_policy(config)
        
        env_id = config.get("environment_id")
        
        if env_id not in self._environments:
            env = Environment([])
        else:
            env = self._environments[env_id]['env']
        
        agent = Agent(policy, env, agent_components, heartrate_ms=config.get("heartrate_ms", 1000))
        
        state = config.get("state", {})
        
        agent.state.set_nested_item("agent", state)
        
        return agent
        

    def create_component(self, component_name: str, component_config: Dict):
        """
        Creates a component based on the configuration
        """
        if component_name == "TodoList":
            return TodoList()
        if component_name == "Creativity":
            assert component_config.get("language_model") is not None, "Language model is not set"
            language_model = self.create_language_model(component_config["language_model"])
            return Creativity(language_model) 
        if component_name == "Email":
            try:
                return Email(**component_config)
            except Exception as e:
                raise Exception(f"Failed to create email component: {e}")
        if component_name == "Management":
            return Management()
        if component_name == "Memory":
            assert component_config.get("embedding_model") is not None, "Embedding model is not set"
            embedding_model = self.create_embegging_model(component_config["embedding_model"])
            assert component_config.get("db", {}).get("implementation") in ["in_memory_vector_db"], "DB is not set or not supported"
            if component_config["db"]["implementation"] == "in_memory_vector_db":
                db = InMemoryVectorDB(embedding_model.dim(), component_config["db"]["params"]["affinity"])
            
            assert component_config.get("memory_size"), "Memory size is not set or is 0"
            
            return Memory(db, embedding_model, component_config["memory_size"])
        if component_name == "WebBrowser":
            assert component_config.get("language_model") is not None, "Language model is not set"
            language_model = self.create_language_model(component_config["language_model"])
            search_api = component_config.get("search_api")
            assert search_api in ["ddg", "serper"], "Search API is not set"
            if search_api == "serper":
                serper_api_key = component_config.get("serper_api_key")
                assert serper_api_key is not None, "Serper API key is not set"
            return WebBrowser(language_model, search_api, serper_api_key)
        if component_name == "Messenger":
            return Messenger()
        if component_name == "RemoteControl":
            assert component_config.get("messaging", {}).get("implementation") in ["websocket"], "Messaging is not set or not supported"
            if component_config["messaging"]["implementation"] == "websocket":
                params = component_config["messaging"].get("params", {})
                try:
                    messaging = WebsocketMessagingJsonServer(**params)
                except Exception as e:
                    raise Exception(f"Failed to create messaging server: {e}")
            subscription_update_rate_ms = component_config.get("subscription_update_rate_ms", 1000)
            return RemoteControl(messaging, subscription_update_rate_ms=subscription_update_rate_ms)
        raise Exception(f"Component {component_name} is not supported")

    def create_policy(self, config: Dict):
        policy_name = config.get("policy", {}).get("implementation")
        assert policy_name in ["LLMPolicy", "TradingPolicy", "HumanControlledPolicy"], "Policy is not set or not supported"
        if policy_name == "LLMPolicy":
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
        if policy_name == "HumanControlledPolicy":
            return HumanControlledPolicy()
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

    async def on_agent_heartbeat(self, agent: IAgent) -> None:
        agents_to_remove = []
        for agent_id, agent_data in self._agents.items():
            if agent_data['process'].done() or agent_data['process'].cancelled():
                agents_to_remove.append(agent_id)
        for agent_id in agents_to_remove:
            await self.kill_agent(agent_id=agent_id, caller_context=State())
            
        environments_to_remove = []
        
        for environment_id, environment_data in self._environments.items():
            if environment_data['process'].done() or environment_data['process'].cancelled():
                environments_to_remove.append(environment_id)
        for environment_id in environments_to_remove:
            await self.destroy_environment(environment_id=environment_id, caller_context=State())
        
        agent.state.set_item(f"agent/components/{self.info().name}/status", "Management skills ready.")
        agent.state.set_item(f"agent/components/{self.info().name}/agents", (await self.agents(caller_context=State())).value)
        agent.state.set_item(f"agent/components/{self.info().name}/environments", (await self.environments(caller_context=State())).value)
    
    async def tick(self) -> None:
        ...
    
    def info(self) -> EntityInfo:
        return EntityInfo(
            name=self.__class__.__name__,
            version="0.1.0",
            params={}
        )
