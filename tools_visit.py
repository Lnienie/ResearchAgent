import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import List, Union

import httpx
from bs4 import BeautifulSoup
from openai import AsyncOpenAI

from prompts import EXTRACTOR_PROMPT
from tool_types import EvidenceItem, ToolResult


DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
VISIT_TIMEOUT = 20
WEBCONTENT_MAXLENGTH = 50000
SUMMARY_MODEL = "qwen-plus"

# Jina Reader API configuration
JINA_ENABLED = os.getenv("JINA_ENABLED", "1").strip() in ("1", "true", "yes")
JINA_API_KEYS = [k.strip() for k in os.getenv(
    "JINA_API_KEYS",
    ""
).split(",") if k.strip()]
JINA_BASE = "https://r.jina.ai/"
JINA_TIMEOUT = 30
_jina_key_index = 0

_summary_client = AsyncOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=DASHSCOPE_API_KEY,
)


@dataclass(slots=True)
class ExtractionResult:
    rational: str = ""
    evidence: str = ""
    summary: str = ""
    raw: str = ""
    parsed: bool = False

    def render(self) -> str:
        if self.parsed:
            return (
                f"Rational:\n{self.rational}\n\n"
                f"Evidence in page:\n{self.evidence}\n\n"
                f"Summary:\n{self.summary}"
            )
        return f"Extracted content:\n{self.raw}"


async def fetch_page_jina(url: str) -> str:
    """Fetch webpage as clean markdown via Jina Reader API with key rotation."""
    global _jina_key_index
    if not JINA_API_KEYS:
        return "[visit] No Jina API keys configured"

    tried = 0
    while tried < len(JINA_API_KEYS):
        key = JINA_API_KEYS[_jina_key_index % len(JINA_API_KEYS)]
        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "text/markdown",
            "X-No-Cache": "true",
        }
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=JINA_TIMEOUT,
                verify=False,
            ) as client:
                resp = await client.get(f"{JINA_BASE}{url}", headers=headers)
                if resp.status_code == 200:
                    text = resp.text.strip()
                    if text:
                        return text
                    return "[visit] Jina returned empty content"
                elif resp.status_code in (400, 403, 429):
                    _jina_key_index += 1
                    tried += 1
                    continue
                else:
                    return f"[visit] Jina HTTP {resp.status_code} for {url}"
        except Exception as e:
            return f"[visit] Jina fetch failed: {str(e)}"

    return "[visit] All Jina API keys exhausted (rate limited)"


async def fetch_page(url: str) -> str:
    """Fetch webpage content using httpx."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    }
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=VISIT_TIMEOUT,
            verify=False,
        ) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                return resp.text
            else:
                return f"[visit] HTTP {resp.status_code} error for {url}"
    except Exception as e:
        return f"[visit] Failed to fetch {url}: {str(e)}"


def extract_main_text(html: str) -> str:
    """Extract main text content from HTML using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "iframe", "noscript", "svg", "form"]):
        tag.decompose()

    # Try to find article content first
    article = soup.find("article")
    if article:
        text = article.get_text(separator="\n", strip=True)
    else:
        # Fall back to body
        body = soup.find("body")
        if body:
            text = body.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

    # Clean up: remove excessive blank lines
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n".join(lines)

    return text


def _parse_extraction(raw: str) -> ExtractionResult:
    raw_clean = raw.strip()
    if raw_clean.startswith("```"):
        fenced_parts = raw_clean.split("```")
        raw_clean = fenced_parts[1] if len(fenced_parts) > 1 else raw_clean
        if raw_clean.startswith("json"):
            raw_clean = raw_clean[4:]
        raw_clean = raw_clean.strip()

    candidates = [raw_clean]
    left = raw_clean.find("{")
    right = raw_clean.rfind("}")
    if left != -1 and right != -1 and left < right:
        candidates.append(raw_clean[left : right + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                continue
            return ExtractionResult(
                rational=str(data.get("rational", "")).strip(),
                evidence=str(data.get("evidence", "")).strip(),
                summary=str(data.get("summary", "")).strip(),
                raw=raw,
                parsed=True,
            )
        except json.JSONDecodeError:
            continue
    return ExtractionResult(raw=raw, parsed=False)


async def summarize_content_result(
    content: str, goal: str, max_retries: int = 2
) -> ExtractionResult:
    """Use the LLM extractor and retain rational/evidence/summary fields."""
    content = content[:WEBCONTENT_MAXLENGTH]

    messages = [
        {
            "role": "user",
            "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal),
        }
    ]

    for attempt in range(max_retries):
        try:
            resp = await _summary_client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=4096,
            )
            raw = resp.choices[0].message.content
            if not raw or len(raw) < 10:
                # Truncate content and retry
                content = content[: int(len(content) * 0.7)]
                messages[0]["content"] = EXTRACTOR_PROMPT.format(
                    webpage_content=content, goal=goal
                )
                continue

            return _parse_extraction(raw)

        except Exception as e:
            print(f"[visit] Summary attempt {attempt + 1} failed: {e}")

    return ExtractionResult(
        raw="The webpage content could not be processed.", parsed=False
    )


async def summarize_content(content: str, goal: str, max_retries: int = 2) -> str:
    """Compatibility API returning the extractor output as text."""
    result = await summarize_content_result(content, goal, max_retries=max_retries)
    return result.render()


def _evidence_items_from_extraction(
    extraction: ExtractionResult, url: str
) -> list[EvidenceItem]:
    evidence_text = extraction.evidence if extraction.parsed else extraction.raw
    if not evidence_text or evidence_text.startswith("The webpage content could not"):
        return []
    chunks = [
        chunk.strip()
        for chunk in re.split(r"\n\s*\n|(?m)^[-*]\s+", evidence_text)
        if len(chunk.strip()) >= 20
    ]
    if not chunks and len(evidence_text.strip()) >= 20:
        chunks = [evidence_text.strip()]
    return [
        EvidenceItem(
            text=chunk,
            source_url=url,
            kind="verified",
            quality=0.9 if extraction.parsed else 0.5,
            metadata={"extractor_parsed": extraction.parsed},
        )
        for chunk in chunks
    ]


async def visit_page_result(url: str, goal: str) -> ToolResult:
    """Visit a single webpage and return extracted information.
    Tries Jina Reader API first, falls back to httpx + BeautifulSoup."""
    # Try Jina Reader API first (if enabled)
    text = await fetch_page_jina(url) if JINA_ENABLED else "[visit] Jina disabled"

    if text.startswith("[visit]"):
        # Jina disabled or failed, fall back to httpx + BeautifulSoup
        html = await fetch_page(url)
        if html.startswith("[visit]"):
            return ToolResult(
                content=(
                    f"The useful information in {url} for user goal {goal} as follows:\n\n"
                    f"Evidence in page:\n{html}\n\n"
                    f"Summary:\nThe webpage content could not be accessed.\n"
                ),
                metadata={"type": "visit", "url": url, "goal": goal},
                success=False,
            )
        text = extract_main_text(html)

    if not text or len(text) < 20:
        return ToolResult(
            content=(
                f"The useful information in {url} for user goal {goal} as follows:\n\n"
                f"Evidence in page:\nEmpty or very short content.\n\n"
                f"Summary:\nThe webpage had no extractable content.\n"
            ),
            metadata={"type": "visit", "url": url, "goal": goal},
            success=False,
        )

    extraction = await summarize_content_result(text, goal)
    evidence_items = _evidence_items_from_extraction(extraction, url)
    return ToolResult(
        content=(
            f"The useful information in {url} for user goal {goal} as follows:\n\n"
            f"{extraction.render()}\n"
        ),
        evidence_items=evidence_items,
        metadata={
            "type": "visit",
            "url": url,
            "goal": goal,
            "extractor": {
                "rational": extraction.rational,
                "evidence": extraction.evidence,
                "summary": extraction.summary,
                "parsed": extraction.parsed,
            },
        },
        success=bool(evidence_items),
    )


async def visit_page(url: str, goal: str) -> str:
    """Compatibility API returning a single visited page as text."""
    return (await visit_page_result(url, goal)).content


async def visit_pages_result(urls: Union[str, List[str]], goal: str) -> ToolResult:
    """Visit one or more webpages concurrently and return combined results."""
    if isinstance(urls, str):
        urls = [urls]

    async def _safe_visit(url: str) -> ToolResult:
        try:
            return await asyncio.wait_for(visit_page_result(url, goal), timeout=120)
        except asyncio.TimeoutError:
            return ToolResult(
                content=f"Error fetching {url}: visit timed out after 120s",
                metadata={"type": "visit", "url": url, "goal": goal},
                success=False,
            )
        except Exception as e:
            return ToolResult(
                content=f"Error fetching {url}: {str(e)}",
                metadata={"type": "visit", "url": url, "goal": goal},
                success=False,
            )

    results = await asyncio.gather(*[_safe_visit(url) for url in urls])
    return ToolResult(
        content="\n=======\n".join(result.content for result in results),
        evidence_items=[
            item for result in results for item in result.evidence_items
        ],
        metadata={
            "type": "visit",
            "pages": [result.metadata for result in results],
        },
        success=any(result.success for result in results),
    )


async def visit_pages(urls: Union[str, List[str]], goal: str) -> str:
    """Compatibility API returning visited pages as prompt-facing text."""
    return (await visit_pages_result(urls, goal)).content
