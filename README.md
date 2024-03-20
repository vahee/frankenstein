# Dr. Frankenstein (wip)

Inspired by AutoGPT, Agent-GPT and other projects aiming to implement autonomous agents with LLMs as policies. Dr. Frankenstein adds its own flavour to the pack of existing projects. It differentiates itself by: 
- it is designed to work truly autonomously by giving the agent curated access to the user's environment like email, calendar, etc. and define its behaviour based on the user's preferences in plain English, like one would do with a human assistant
- lightweight and easy to use
- being modular and extensible
- adding web UI
- easy to define and refine agent behaviour, without the need to code
- easy to integrate with other systems like email, calendar, etc.

## Installation
    
1. Clone the repository
2. Create .my.config.yaml based on .config.yaml.example
3. Create a virtual environment
4. Install the requirements (requirements.txt)
5. python main.py
6. Talk to the agent in plain text via websocket (using address you specify for messenger component)

or create a devcontainer in VSCode based on the .devcontainer folder
