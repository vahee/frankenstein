from typing import Literal
from os import linesep
import json
import requests
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

from agentopy import ActionResult, WithStateMixin, WithActionSpaceMixin, Action, IEnvironmentComponent, EntityInfo
from frankenstein.lib.language.protocols import ILanguageModel


class WebBrowser(WithStateMixin, WithActionSpaceMixin, IEnvironmentComponent):
    """Implementation of a web browser component"""

    def __init__(self, language_model: ILanguageModel, search_api: Literal['ddg', 'serper'] = 'serper', serper_api_key: str | None = None) -> None:
        super().__init__()
        self.language_model = language_model
        self.search_api = search_api
        self.serper_api_key = serper_api_key

        self.action_space.register_actions(
            [
                Action(
                    "browse_website", "use this action to open a website and look for specific information", self.browse_website),
                Action(
                    "search", "searches Google for the specified query and returns a list of results", self.search)
            ])

    async def search(self, query: str) -> ActionResult:
        """
        Searches the web for the specified query
        """
        try:

            if self.search_api == 'serper':
                assert self.serper_api_key is not None, "Serper API key is required"
                url = "https://google.serper.dev/search"

                payload = json.dumps({
                    "q": query
                })
                headers = {
                    'X-API-KEY': self.serper_api_key,
                    'Content-Type': 'application/json'
                }

                response = requests.request(
                    "POST", url, headers=headers, data=payload, timeout=30)

                results = response.json()['organic']
                title, href, body = ('title', 'link', 'snippet')
            else:
                results = DDGS().text(query, safesearch="Off")
                title, href, body = ('title', 'href', 'body')

            formatted_results = []
            max_results = 10
            for result in results:
                formatted_results.append(
                    f"Title: {result[title]}{linesep}Link: {result[href]}{linesep}Description: {result[body]}")
                max_results -= 1
                if max_results == 0:
                    break

            if not formatted_results:
                return ActionResult(value="No results found", success=True)

            return ActionResult(value=linesep.join(formatted_results), success=True)

        except Exception as e:
            return ActionResult(value=f"Error: {e}", success=False)

    async def browse_website(self, url: str, text_to_look_for: str | None = None) -> ActionResult:
        """
        Opens a website and returns the text on the page
        """
        try:
            html = requests.get(url, timeout=30).text
            text = self._html_to_text(html)

            content = await self._summarize_text(
                text) if text_to_look_for is None else await self._scan_text(text, text_to_look_for)
            links = self._links(html)

            return ActionResult(value=f"Website content: {content}\nWebsite links: {links}", success=True)
        except Exception as e:
            return ActionResult(value=f"Error: {e}", success=False)

    def _html_to_text(self, html: str):
        """
        Converts html to text
        """
        soup = BeautifulSoup(html, features="html.parser")

        for elem in soup(["script", "style"]):
            elem.extract()

        text = soup.get_text()

        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip()
                  for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text

    def _links(self, html: str) -> str:
        """
        Returns all links in the html, in text format
        """
        soup = BeautifulSoup(html, features="html.parser")
        links_text = '\n'.join(
            [f"{link.text} {link.get('href')}" for link in soup.find_all('a')])
        encodings = self.language_model.encode(links_text)
        return self.language_model.decode(encodings[:1000])

    async def _summarize_text(self, text: str):
        chunk = 0
        summary = ""
        chunk_size = 10000  # TODO: make this a parameter
        encodings = self.language_model.encode(text)
        for i in range(0, len(encodings), chunk_size):
            chunk += 1
            summary = await self.language_model.query(
                self.language_model.decode(encodings[i:i + chunk_size]),
                f"""You are summarizing large text, part by part.
                    The summary so far is:
                    {summary}

                    Summarize it, keep, overall summary in under 1000 tokens, make sure preserve the important information.
                    """)
        return summary

    async def _scan_text(self, text: str, information_to_look_for: str):
        chunk = 0
        findings = ""
        window_size = 10000  # TODO: make this a parameter
        encodings = self.language_model.encode(text)
        for i in range(0, len(encodings), window_size - window_size // 4):
            chunk += 1
            findings = await self.language_model.query(
                self.language_model.decode(
                    encodings[i: i + window_size]),
                f"""You are scanning large text, part by part. The information you are looking for is {information_to_look_for}.
                    {f'Findings so far are: {findings}' if findings else ''}

                    If the information is found, keep it in the findings. Next is the next part of the text.
                    """)
        return findings

    async def tick(self) -> None:
        self._state.set_item("status", "Web browser is ready for use")

    def info(self) -> EntityInfo:
        return EntityInfo(name="WebBrowser", version="0.1.0", params={})
