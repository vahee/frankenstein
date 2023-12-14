import logging
import traceback
import asyncio as aio
import sys
from typing import Tuple
from agentopy import Agent, Environment, IAgent, IEnvironment

from config import CONFIG

from llmagents.language.openai_language_models import OpenAIChatModel
from llmagents.language.embedding_models import OpenAIEmbeddingModel, SentenceTransformerEmbeddingModel
from llmagents.language.llm_policy import LLMPolicy
from llmagents.db.in_memory_vector_db import InMemoryVectorDB
from llmagents.components import TodoList, Creativity, Email, Memory, WebBrowser, Messenger, RemoteControl
from llmagents.lib.communication import WebsocketMessagingServer


logging.basicConfig(handlers=[logging.FileHandler('llmagents.log', mode='w', encoding='utf-8'),
                    logging.StreamHandler(sys.stdout)],
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logging.info("Running llmagents")

logger = logging.getLogger('main')


def create_env_and_agent() -> Tuple[IEnvironment, IAgent]:
    language_model, embeddings_model = None, None

    assert CONFIG.agent.language_model is not None, "Language model is not set"
    assert CONFIG.agent.embeddings_model is not None, "Embeddings model is not set"

    if CONFIG.agent.language_model.startswith("openai:"):
        assert CONFIG.openai_api_key is not None, "OpenAI API key is not set"
        language_model = OpenAIChatModel(CONFIG.openai_api_key,
                                         CONFIG.agent.language_model.lstrip("openai:"), json=True)

    if CONFIG.agent.embeddings_model.startswith("openai:"):
        assert CONFIG.openai_api_key is not None, "OpenAI API key is not set"
        embeddings_model = OpenAIEmbeddingModel(CONFIG.openai_api_key,
                                                CONFIG.agent.embeddings_model.lstrip("openai:"))

    if CONFIG.agent.embeddings_model.startswith("st:"):
        assert CONFIG.hf_access_token is not None, "Hugging Face access token is not set"
        embeddings_model = SentenceTransformerEmbeddingModel(
            CONFIG.agent.embeddings_model.lstrip("st:"), CONFIG.hf_access_token)

    assert language_model is not None, "Language model is not set"
    assert embeddings_model is not None, "Embeddings model is not set"

    environment_components = []
    agent_components = []

    assert CONFIG.messenger_websocket_host is not None, "Messenger websocket host is not set"
    assert CONFIG.messenger_websocket_port is not None, "Messenger websocket port is not set"
    assert CONFIG.messenger_websocket_ping_interval is not None, "Messenger websocket ping interval is not set"
    assert CONFIG.messenger_websocket_ping_timeout is not None, "Messenger websocket ping timeout is not set"

    environment_components.append(("Messenger", Messenger(WebsocketMessagingServer(
        CONFIG.messenger_websocket_host,
        CONFIG.messenger_websocket_port,
        CONFIG.messenger_websocket_ping_interval,
        CONFIG.messenger_websocket_ping_timeout
    ))))

    environment_components.append(("Todo List", TodoList()))

    assert CONFIG.openai_api_key is not None
    environment_components.append(("Creativity", Creativity(language_model=OpenAIChatModel(CONFIG.openai_api_key,
                                                                                           CONFIG.agent.language_model.lstrip("openai:"), json=False))))

    assert CONFIG.email_imap_address is not None, "Email IMAP address is not set"
    assert CONFIG.email_smtp_address is not None, "Email SMTP address is not set"
    assert CONFIG.email_smtp_port is not None, "Email SMTP port is not set"
    assert CONFIG.email_login is not None, "Email login is not set"
    assert CONFIG.email_password is not None, "Email password is not set"
    assert CONFIG.email_from is not None, "Email from is not set"
    assert CONFIG.outbound_emails_whitelist is not None, "Outbound emails whitelist is not set"

    environment_components.append(("Email", Email(
        imap_address=CONFIG.email_imap_address,
        smtp_address=CONFIG.email_smtp_address,
        smtp_port=CONFIG.email_smtp_port,
        login=CONFIG.email_login,
        password=CONFIG.email_password,
        from_address=CONFIG.email_from,
        outbound_emails_whitelist=CONFIG.outbound_emails_whitelist,
    )))

    environment_components.append(("Web Browser", WebBrowser(language_model=OpenAIChatModel(CONFIG.openai_api_key,
                                                                                            CONFIG.agent.language_model.lstrip("openai:"), json=False), serper_api_key=CONFIG.serper_api_key)))

    agent_components.append(Memory(InMemoryVectorDB(
        embeddings_model.dim(), 'cosine'), embeddings_model, 10))

    assert CONFIG.remote_control_websocket_host is not None, "Remote control websocket host is not set"
    assert CONFIG.remote_control_websocket_port is not None, "Remote control websocket port is not set"
    assert CONFIG.remote_control_websocket_ping_interval is not None, "Remote control websocket ping interval is not set"
    assert CONFIG.remote_control_websocket_ping_timeout is not None, "Remote control websocket ping timeout is not set"

    agent_components.append(RemoteControl(
        WebsocketMessagingServer(
            CONFIG.remote_control_websocket_host,
            CONFIG.remote_control_websocket_port,
            CONFIG.remote_control_websocket_ping_interval,
            CONFIG.remote_control_websocket_ping_timeout
        )
    ))

    policy = LLMPolicy(
        language_model,
        CONFIG.agent.system_prompt.format(
            name="Dude", **{f'my_{k}': v for k, v in CONFIG.me.__dict__.items()}),
        CONFIG.agent.response_template
    )

    env = Environment(environment_components)

    agent = Agent(policy, env, agent_components)

    return env, agent


async def main():
    """
    Main function of the project. It is responsible for running the assitant.
    """
    env, agent = create_env_and_agent()

    logger.info("Starting environment and agent")

    e = await aio.wait([env.start(), agent.start()], return_when=aio.FIRST_EXCEPTION)
    expection = e[0].pop().exception()
    print(''.join(traceback.format_tb(
        expection.__traceback__)))
    print(expection)

if __name__ == "__main__":
    aio.run(main())
