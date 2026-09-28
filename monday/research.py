"""Company research through Ollama's web search / fetch API.

The API key is sent only by this module's own client and only to ollama.com's search
endpoints. Page content is untrusted data: it is quoted to the model as labelled
sources, and every claim built from it must cite one of those sources.
"""

from dataclasses import dataclass
from datetime import date

import ollama

from monday.skills import normalize

MAX_RESULTS_PER_QUERY = 3
MAX_SOURCE_CHARS = 2500


class ResearchError(RuntimeError):
    pass


@dataclass
class Source:
    title: str
    url: str
    content: str

    def as_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "content": self.content}


def queries(company: str, role: str) -> list[str]:
    return [
        f"{company} company overview products customers",
        f"{company} engineering blog technology stack",
        f"{company} news {date.today().year}",
        f"{company} {role} interview process questions",
    ]


def mentions_company(company: str, text: str) -> bool:
    """Loose relevance filter: 'Square Yards' should match 'SquareYards' and 'square yards'."""
    name = normalize(company)
    squashed = name.replace(" ", "")
    t = normalize(text)
    return name in t or squashed in t.replace(" ", "")


class Researcher:
    def __init__(self, api_key: str | None, client: ollama.Client | None = None):
        if not api_key and client is None:
            raise ResearchError("No Ollama API key. Put OLLAMA_API_KEY=... in the .env file to enable web research.")
        # A dedicated client: the key never rides along on local model calls.
        self.client = client or ollama.Client(headers={"authorization": f"Bearer {api_key}"})

    def gather(self, company: str, role: str, website: str | None = None) -> tuple[list[Source], list[str]]:
        """Returns (sources, notes about anything that failed or was filtered out)."""
        sources: dict[str, Source] = {}
        notes: list[str] = []

        if website:
            try:
                page = self.client.web_fetch(website)
                sources[website] = Source(page.title or company, website, (page.content or "")[:MAX_SOURCE_CHARS])
            except Exception as exc:  # a dead site shouldn't stop the rest of the research
                notes.append(f"could not fetch {website}: {exc}")

        failures = 0
        for q in queries(company, role):
            try:
                results = self.client.web_search(q, max_results=MAX_RESULTS_PER_QUERY).results
            except Exception as exc:
                failures += 1
                notes.append(f"search failed for '{q}': {exc}")
                continue
            for r in results:
                if r.url in sources:
                    continue
                if not mentions_company(company, f"{r.title} {r.content}"):
                    notes.append(f"skipped (doesn't mention {company}): {r.url}")
                    continue
                sources[r.url] = Source(r.title or r.url, r.url, (r.content or "")[:MAX_SOURCE_CHARS])

        if failures == len(queries(company, role)) and not sources:
            raise ResearchError("Every web search failed:\n" + "\n".join(notes))
        return list(sources.values()), notes
