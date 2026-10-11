"""Runtime trajectory guard for long-running ReAct retrieval agents.

The guard combines three independent signals instead of relying on a hard
round limit:

1. semantic similarity between recent queries and observations;
2. canonical tool-call fingerprints;
3. consecutive rounds without novel evidence.

It is intentionally request-scoped.  A new ``TrajectoryGuard`` must be
created for every user question so concurrent requests never share state.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import re
import unicodedata
import urllib.parse
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from openai import AsyncOpenAI

from tool_types import EvidenceItem, ToolResult


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class GuardConfig:
    enabled: bool = True
    window_size: int = 5
    query_similarity_threshold: float = 0.92
    observation_similarity_threshold: float = 0.94
    evidence_similarity_threshold: float = 0.93
    no_new_evidence_threshold: int = 2
    force_converge_threshold: int = 5
    max_evidence_items: int = 200
    max_embedding_cache_items: int = 256
    max_intent_chars: int = 2000
    max_observation_chars: int = 6000
    max_evidence_chars: int = 3000
    embedding_timeout_seconds: float = 8.0
    log_enabled: bool = True

    @classmethod
    def from_env(cls) -> "GuardConfig":
        return cls(
            enabled=_env_bool("TRAJECTORY_GUARD_ENABLED", True),
            window_size=max(2, _env_int("TRAJECTORY_WINDOW_SIZE", 5)),
            query_similarity_threshold=_env_float(
                "QUERY_SIMILARITY_THRESHOLD", 0.92
            ),
            observation_similarity_threshold=_env_float(
                "OBSERVATION_SIMILARITY_THRESHOLD", 0.94
            ),
            evidence_similarity_threshold=_env_float(
                "EVIDENCE_SIMILARITY_THRESHOLD", 0.93
            ),
            no_new_evidence_threshold=max(
                1, _env_int("NO_NEW_EVIDENCE_THRESHOLD", 2)
            ),
            force_converge_threshold=max(
                2, _env_int("FORCE_CONVERGE_THRESHOLD", 5)
            ),
            max_evidence_items=max(20, _env_int("MAX_EVIDENCE_ITEMS", 200)),
            embedding_timeout_seconds=max(
                1.0, _env_float("EMBEDDING_TIMEOUT_SECONDS", 8.0)
            ),
            log_enabled=_env_bool("TRAJECTORY_GUARD_LOG_ENABLED", True),
        )


class EmbeddingProvider(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in the same order as the input sequence."""


class DashScopeEmbeddingProvider:
    """Small adapter around DashScope's OpenAI-compatible embedding API."""

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ) -> None:
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
        self.batch_size = max(1, _env_int("EMBEDDING_BATCH_SIZE", 10))
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        values = list(texts)
        for offset in range(0, len(values), self.batch_size):
            response = await self.client.embeddings.create(
                model=self.model,
                input=values[offset : offset + self.batch_size],
                encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(list(item.embedding) for item in ordered)
        return vectors


@dataclass(slots=True)
class GuardContext:
    round_idx: int
    tool_name: str
    tool_args: dict[str, Any]
    intent_text: str
    query_embedding: list[float] | None
    query_similarity: float
    fingerprint: str
    scope_fingerprint: str
    fingerprint_repeated: bool
    scope_repeated: bool


@dataclass(slots=True)
class GuardDecision:
    action: str = "continue"
    tool_args: dict[str, Any] = field(default_factory=dict)
    block_execution: bool = False
    force_converge: bool = False
    guard_message: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    context: GuardContext | None = None


@dataclass(slots=True)
class TrajectoryStep:
    round_idx: int
    tool_name: str
    tool_args: dict[str, Any]
    intent_text: str
    observation_text: str
    fingerprint: str
    scope_fingerprint: str
    query_embedding: list[float] | None
    observation_embedding: list[float] | None
    query_similarity: float
    observation_similarity: float
    fingerprint_repeated: bool
    scope_repeated: bool
    evidence_ids: list[str]
    new_evidence_count: int
    no_new_evidence_count: int
    action: str


@dataclass(slots=True)
class EvidenceRecord:
    evidence_id: str
    text: str
    source_url: str
    kind: str
    quality: float
    embedding: list[float] | None


TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "ref",
    "referrer",
    "source",
    "spm",
}


def normalize_text(text: Any) -> str:
    """Normalize text for stable hashing without erasing Chinese tokens."""
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = re.sub(r"[^\w\u4e00-\u9fff]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def canonicalize_url(url: str) -> str:
    """Remove fragments and tracking parameters from a URL."""
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urllib.parse.urlsplit(raw)
        scheme = parsed.scheme.lower() or "https"
        hostname = (parsed.hostname or "").lower()
        if not hostname:
            return normalize_text(raw)
        port = parsed.port
        default_port = (scheme == "http" and port == 80) or (
            scheme == "https" and port == 443
        )
        netloc = hostname if not port or default_port else f"{hostname}:{port}"
        path = re.sub(r"/{2,}", "/", parsed.path or "/")
        if path != "/":
            path = path.rstrip("/")
        query_items = []
        for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
            lowered = key.lower()
            if lowered.startswith("utm_") or lowered in TRACKING_QUERY_KEYS:
                continue
            query_items.append((key, value))
        query = urllib.parse.urlencode(sorted(query_items))
        return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))
    except Exception:
        return normalize_text(raw)


def _url_domain(url: str) -> str:
    canonical = canonicalize_url(url)
    try:
        return (urllib.parse.urlsplit(canonical).hostname or "").lower()
    except Exception:
        return ""


def _normalize_engine(engine: Any) -> str:
    value = normalize_text(engine)
    if value in {"google", "serper", "google serper"}:
        return "google"
    if value in {"bing", "iqs", "bing iqs"}:
        return "bing"
    return value or "auto"


def canonical_tool_call(
    tool_name: str, tool_args: dict[str, Any], *, include_routing: bool = True
) -> dict[str, Any]:
    """Build a canonical representation used by tool-call fingerprints."""
    name = normalize_text(tool_name)
    if name == "search":
        raw_queries = tool_args.get("query", [])
        queries = raw_queries if isinstance(raw_queries, list) else [raw_queries]
        raw_engines = tool_args.get("engine", [])
        engines = raw_engines if isinstance(raw_engines, list) else [raw_engines]
        requests = []
        for index, query in enumerate(queries):
            item: dict[str, Any] = {"query": normalize_text(query)}
            if include_routing:
                engine = engines[index] if index < len(engines) else "auto"
                item["engine"] = _normalize_engine(engine)
            requests.append(item)
        return {"name": name, "requests": sorted(requests, key=json.dumps)}
    if name == "visit":
        raw_urls = tool_args.get("url", [])
        urls = raw_urls if isinstance(raw_urls, list) else [raw_urls]
        canonical: dict[str, Any] = {
            "name": name,
            "urls": sorted(canonicalize_url(url) for url in urls if str(url).strip()),
        }
        if include_routing:
            canonical["goal"] = normalize_text(tool_args.get("goal", ""))
        return canonical

    def _walk(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): _walk(value[key]) for key in sorted(value)}
        if isinstance(value, list):
            return sorted((_walk(item) for item in value), key=lambda item: str(item))
        if isinstance(value, str):
            return normalize_text(value)
        return value

    return {"name": name, "arguments": _walk(tool_args)}


def tool_call_fingerprint(
    tool_name: str, tool_args: dict[str, Any], *, include_routing: bool = True
) -> str:
    canonical = canonical_tool_call(
        tool_name, tool_args, include_routing=include_routing
    )
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _token_jaccard(left: str, right: str) -> float:
    def _features(text: str) -> set[str]:
        normalized = normalize_text(text)
        tokens = set(normalized.split())
        compact = normalized.replace(" ", "")
        # Chinese text often has no whitespace, so token Jaccard would collapse
        # an entire paragraph into one token. Character trigrams provide a
        # deterministic semantic fallback when the embedding service is down.
        if len(tokens) <= 2 and len(compact) >= 3:
            tokens.update(compact[index : index + 3] for index in range(len(compact) - 2))
        return tokens

    left_tokens = _features(left)
    right_tokens = _features(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _contains_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


class TrajectoryGuard:
    """Detect and recover from repetitive ReAct retrieval trajectories."""

    def __init__(
        self,
        question: str,
        *,
        config: GuardConfig | None = None,
        embedder: EmbeddingProvider | None = None,
        run_id: str | None = None,
    ) -> None:
        self.question = question
        self.config = config or GuardConfig.from_env()
        self.run_id = run_id or uuid.uuid4().hex[:12]
        if embedder is None and self.config.enabled:
            api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
            embedder = DashScopeEmbeddingProvider(api_key) if api_key else None
        self.embedder = embedder
        self.steps: deque[TrajectoryStep] = deque(maxlen=self.config.window_size)
        self.evidence_bank: list[EvidenceRecord] = []
        self.seen_evidence_hashes: set[str] = set()
        self.no_new_evidence_count = 0
        self.recovery_count = 0
        self._embedding_cache: OrderedDict[str, list[float] | None] = OrderedDict()
        self._next_search_engine_switch = False
        self.embedding_available = embedder is not None

    async def before_tool(
        self, round_idx: int, tool_name: str, tool_args: dict[str, Any]
    ) -> GuardDecision:
        args = copy.deepcopy(tool_args or {})
        if not self.config.enabled:
            return GuardDecision(tool_args=args)

        applied_switch = False
        if normalize_text(tool_name) == "search" and self._next_search_engine_switch:
            args = self._switch_search_engines(args)
            self._next_search_engine_switch = False
            applied_switch = True

        intent_text = self._intent_text(tool_name, args)[: self.config.max_intent_chars]
        query_embedding = await self._embed_one(intent_text)
        query_similarity = self._max_step_similarity(
            tool_name, query_embedding, field_name="query_embedding"
        )
        if query_embedding is None:
            query_similarity = self._max_text_similarity(
                tool_name, intent_text, field_name="intent_text"
            )
        fingerprint = tool_call_fingerprint(tool_name, args)
        scope_fingerprint = tool_call_fingerprint(
            tool_name, args, include_routing=False
        )
        fingerprint_repeated = any(
            step.fingerprint == fingerprint for step in self.steps
        )
        scope_repeated = any(
            step.scope_fingerprint == scope_fingerprint for step in self.steps
        )
        context = GuardContext(
            round_idx=round_idx,
            tool_name=normalize_text(tool_name),
            tool_args=args,
            intent_text=intent_text,
            query_embedding=query_embedding,
            query_similarity=query_similarity,
            fingerprint=fingerprint,
            scope_fingerprint=scope_fingerprint,
            fingerprint_repeated=fingerprint_repeated,
            scope_repeated=scope_repeated,
        )

        if applied_switch:
            decision = GuardDecision(
                action="switch_engine",
                tool_args=args,
                guard_message=self._guard_message(
                    "switch_engine",
                    "The runtime changed this search to the alternate retrieval engine. "
                    "Evaluate whether it yields genuinely different evidence.",
                ),
                context=context,
            )
            self._log("before_tool", context, decision)
            return decision

        query_repeated = (
            query_similarity >= self.config.query_similarity_threshold
        )
        pre_execution_loop = (
            fingerprint_repeated
            and self.no_new_evidence_count >= 1
            and (query_repeated or query_embedding is None)
        )
        if pre_execution_loop:
            # A blocked repeat is itself another trajectory round without evidence.
            self.no_new_evidence_count += 1
            decision = self._recovery_decision(context, origin="before")
            self._log("before_tool", context, decision)
            return decision

        decision = GuardDecision(
            tool_args=args,
            diagnostics=self._diagnostics(context),
            context=context,
        )
        self._log("before_tool", context, decision)
        return decision

    async def after_tool(
        self,
        round_idx: int,
        tool_name: str,
        tool_args: dict[str, Any],
        result: ToolResult,
        *,
        context: GuardContext | None = None,
    ) -> GuardDecision:
        if not self.config.enabled:
            return GuardDecision(tool_args=copy.deepcopy(tool_args or {}))

        if context is None:
            # Defensive fallback for callers that do not retain before_tool context.
            pre = await self.before_tool(round_idx, tool_name, tool_args)
            context = pre.context
        if context is None:
            return GuardDecision(tool_args=copy.deepcopy(tool_args or {}))

        observation_text = self._observation_text(result)
        observation_embedding = await self._embed_one(observation_text)
        observation_similarity = self._max_step_similarity(
            tool_name, observation_embedding, field_name="observation_embedding"
        )
        if observation_embedding is None:
            observation_similarity = self._max_text_similarity(
                tool_name, observation_text, field_name="observation_text"
            )
        evidence_ids = await self._register_new_evidence(result.evidence_items)
        new_evidence_count = len(evidence_ids)
        if new_evidence_count:
            self.no_new_evidence_count = 0
            self.recovery_count = 0
            self._next_search_engine_switch = False
        else:
            self.no_new_evidence_count += 1

        query_repeated = (
            context.query_similarity >= self.config.query_similarity_threshold
        )
        observation_repeated = (
            observation_similarity >= self.config.observation_similarity_threshold
        )
        strong_semantic_repeat = query_repeated and observation_repeated
        fingerprint_signal = context.fingerprint_repeated or (
            context.scope_repeated and query_repeated
        )
        no_new_signal = (
            self.no_new_evidence_count
            >= self.config.no_new_evidence_threshold
        )

        confirmed_loop = no_new_signal and (
            strong_semantic_repeat or fingerprint_signal
        )
        # Rephrased queries can still return the same observation.  Require one
        # additional no-evidence round before observation-only recovery.
        repeated_observation_loop = (
            self.no_new_evidence_count
            >= self.config.no_new_evidence_threshold + 1
            and observation_repeated
        )

        action = "continue"
        decision = GuardDecision(tool_args=context.tool_args, context=context)
        if confirmed_loop or repeated_observation_loop:
            decision = self._recovery_decision(context, origin="after")
            action = decision.action

        step = TrajectoryStep(
            round_idx=round_idx,
            tool_name=normalize_text(tool_name),
            tool_args=copy.deepcopy(context.tool_args),
            intent_text=context.intent_text,
            observation_text=observation_text,
            fingerprint=context.fingerprint,
            scope_fingerprint=context.scope_fingerprint,
            query_embedding=context.query_embedding,
            observation_embedding=observation_embedding,
            query_similarity=context.query_similarity,
            observation_similarity=observation_similarity,
            fingerprint_repeated=context.fingerprint_repeated,
            scope_repeated=context.scope_repeated,
            evidence_ids=evidence_ids,
            new_evidence_count=new_evidence_count,
            no_new_evidence_count=self.no_new_evidence_count,
            action=action,
        )
        self.steps.append(step)
        diagnostics = self._diagnostics(
            context,
            observation_similarity=observation_similarity,
            new_evidence_count=new_evidence_count,
            confirmed_loop=confirmed_loop or repeated_observation_loop,
        )
        decision.diagnostics.update(diagnostics)
        self._log("after_tool", context, decision, diagnostics)
        return decision

    def evidence_summary(self, limit: int = 12) -> str:
        """Render a compact, source-aware ledger for forced convergence."""
        if not self.config.enabled:
            return ""
        if not self.evidence_bank:
            return "No valid evidence was collected."
        lines = []
        ranked = sorted(
            enumerate(self.evidence_bank),
            key=lambda pair: (
                pair[1].kind == "verified",
                pair[1].quality,
                pair[0],
            ),
            reverse=True,
        )
        for _, record in ranked[: max(1, limit)]:
            source = f" ({record.source_url})" if record.source_url else ""
            text = re.sub(r"\s+", " ", record.text).strip()[:500]
            lines.append(f"- [{record.kind}] {text}{source}")
        return "\n".join(lines)

    def _recovery_decision(
        self, context: GuardContext, *, origin: str
    ) -> GuardDecision:
        if (
            self.recovery_count >= 3
            and self.no_new_evidence_count
            >= self.config.force_converge_threshold
        ):
            action = "force_converge"
            detail = (
                "Several recovery attempts failed and no new evidence has been found. "
                "Stop calling tools and answer from the verified evidence ledger."
            )
            return GuardDecision(
                action=action,
                tool_args=context.tool_args,
                block_execution=origin == "before",
                force_converge=True,
                guard_message=self._guard_message(action, detail),
                diagnostics=self._diagnostics(context),
                context=context,
            )

        if self.recovery_count == 0:
            action = "rewrite_query"
            detail = (
                "The proposed retrieval intent repeats a recent trajectory that added no "
                "evidence. Target one unresolved constraint using a different entity, "
                "relation, date, language, or source type. Do not merely paraphrase the "
                "same query."
            )
            block = origin == "before"
        elif self.recovery_count == 1 and context.tool_name == "search":
            action = "switch_engine"
            detail = (
                "The same search path is still unproductive. The runtime will force the "
                "next search through the alternate engine; use the returned source set "
                "instead of repeating the current engine."
            )
            block = False
            if origin == "after":
                self._next_search_engine_switch = True
            else:
                switched_args = self._switch_search_engines(context.tool_args)
                self.recovery_count += 1
                switched_context = GuardContext(
                    round_idx=context.round_idx,
                    tool_name=context.tool_name,
                    tool_args=switched_args,
                    intent_text=self._intent_text(context.tool_name, switched_args),
                    query_embedding=context.query_embedding,
                    query_similarity=context.query_similarity,
                    fingerprint=tool_call_fingerprint(context.tool_name, switched_args),
                    scope_fingerprint=tool_call_fingerprint(
                        context.tool_name, switched_args, include_routing=False
                    ),
                    fingerprint_repeated=False,
                    scope_repeated=context.scope_repeated,
                )
                return GuardDecision(
                    action=action,
                    tool_args=switched_args,
                    block_execution=False,
                    guard_message=self._guard_message(
                        action,
                        "The runtime forced this call through the alternate search engine.",
                    ),
                    diagnostics=self._diagnostics(switched_context),
                    context=switched_context,
                )
        elif self.recovery_count == 1 and context.tool_name == "visit":
            action = "switch_source"
            detail = (
                "This page path is not adding evidence. Select an independent URL from a "
                "different domain and do not revisit the current canonical URL."
            )
            block = origin == "before"
        else:
            action = "replan_subtask"
            detail = (
                "Pause retrieval and audit progress: list satisfied constraints, missing "
                "constraints, the current candidate, and the smallest unresolved subtask. "
                "The next call must use a new query, URL, or evidence source."
            )
            block = origin == "before"

        self.recovery_count += 1
        return GuardDecision(
            action=action,
            tool_args=context.tool_args,
            block_execution=block,
            guard_message=self._guard_message(action, detail),
            diagnostics=self._diagnostics(context),
            context=context,
        )

    async def _register_new_evidence(
        self, items: Sequence[EvidenceItem]
    ) -> list[str]:
        candidates: list[tuple[EvidenceItem, str, str, str]] = []
        for item in items:
            normalized = normalize_text(item.text)
            if len(normalized) < 20 or item.quality < 0.25:
                continue
            source_url = canonicalize_url(item.source_url)
            source_domain = _url_domain(source_url)
            # Independent visited pages can corroborate the same fact.  Keep one
            # verified record per semantic claim and source domain, while search
            # leads remain content-only to avoid counting mirrors/ranking churn.
            identity = normalized
            if item.kind == "verified" and source_domain:
                identity = f"{normalized}|source:{source_domain}"
            evidence_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            if evidence_id in self.seen_evidence_hashes:
                continue
            self.seen_evidence_hashes.add(evidence_id)
            candidates.append((item, normalized, evidence_id, source_url))

        embeddings = await self._embed_many(
            [
                normalized[: self.config.max_evidence_chars]
                for _, normalized, _, _ in candidates
            ]
        )
        new_ids: list[str] = []
        for index, (item, normalized, evidence_id, source_url) in enumerate(candidates):
            embedding = embeddings[index] if index < len(embeddings) else None
            similar_records: list[EvidenceRecord] = []
            for record in self.evidence_bank:
                if embedding is not None and record.embedding is not None:
                    if (
                        cosine_similarity(embedding, record.embedding)
                        >= self.config.evidence_similarity_threshold
                    ):
                        similar_records.append(record)
                elif _token_jaccard(normalized, record.text) >= 0.88:
                    similar_records.append(record)
            duplicate = bool(similar_records)
            source_domain = _url_domain(source_url)
            if duplicate and item.kind == "verified" and source_domain:
                # Same claim from a genuinely different domain is useful
                # cross-source validation, not an empty trajectory step.
                duplicate = any(
                    _url_domain(record.source_url) == source_domain
                    for record in similar_records
                )
            if duplicate:
                continue
            record = EvidenceRecord(
                evidence_id=evidence_id,
                text=item.text[: self.config.max_evidence_chars],
                source_url=source_url,
                kind=item.kind,
                quality=item.quality,
                embedding=embedding,
            )
            self.evidence_bank.append(record)
            if len(self.evidence_bank) > self.config.max_evidence_items:
                self.evidence_bank.pop(0)
            new_ids.append(evidence_id)
        return new_ids

    async def _embed_one(self, text: str) -> list[float] | None:
        embeddings = await self._embed_many([text])
        return embeddings[0] if embeddings else None

    async def _embed_many(self, texts: Sequence[str]) -> list[list[float] | None]:
        if not texts:
            return []
        normalized_texts = [normalize_text(text) for text in texts]
        results: list[list[float] | None] = [None] * len(normalized_texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        missing_keys: list[str] = []
        for index, text in enumerate(normalized_texts):
            cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if cache_key in self._embedding_cache:
                results[index] = self._embedding_cache[cache_key]
                self._embedding_cache.move_to_end(cache_key)
            else:
                missing_indices.append(index)
                missing_texts.append(text)
                missing_keys.append(cache_key)

        if missing_texts and self.embedder is not None:
            try:
                embedded = await asyncio.wait_for(
                    self.embedder.embed(missing_texts),
                    timeout=self.config.embedding_timeout_seconds,
                )
                if len(embedded) != len(missing_texts):
                    raise ValueError("embedding response length does not match input")
                for index, cache_key, vector in zip(
                    missing_indices, missing_keys, embedded
                ):
                    results[index] = list(vector)
                    self._cache_embedding(cache_key, list(vector))
            except Exception as exc:
                self.embedding_available = False
                self._log_error(f"embedding unavailable: {exc}")
                for index, cache_key in zip(missing_indices, missing_keys):
                    results[index] = None
                    self._cache_embedding(cache_key, None)
        return results

    def _cache_embedding(self, key: str, value: list[float] | None) -> None:
        self._embedding_cache[key] = value
        self._embedding_cache.move_to_end(key)
        while len(self._embedding_cache) > self.config.max_embedding_cache_items:
            self._embedding_cache.popitem(last=False)

    def _max_step_similarity(
        self,
        tool_name: str,
        embedding: list[float] | None,
        *,
        field_name: str,
    ) -> float:
        if embedding is None:
            return 0.0
        similarities = []
        normalized_tool = normalize_text(tool_name)
        for step in self.steps:
            if step.tool_name != normalized_tool:
                continue
            previous = getattr(step, field_name)
            if previous is not None:
                similarities.append(cosine_similarity(embedding, previous))
        return max(similarities, default=0.0)

    def _max_text_similarity(
        self, tool_name: str, text: str, *, field_name: str
    ) -> float:
        normalized_tool = normalize_text(tool_name)
        return max(
            (
                _token_jaccard(text, getattr(step, field_name))
                for step in self.steps
                if step.tool_name == normalized_tool
            ),
            default=0.0,
        )

    def _intent_text(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        name = normalize_text(tool_name)
        if name == "search":
            raw = tool_args.get("query", [])
            queries = raw if isinstance(raw, list) else [raw]
            return "search intent: " + " | ".join(normalize_text(q) for q in queries)
        if name == "visit":
            raw = tool_args.get("url", [])
            urls = raw if isinstance(raw, list) else [raw]
            domains = []
            for url in urls:
                try:
                    domains.append(urllib.parse.urlsplit(str(url)).hostname or str(url))
                except Exception:
                    domains.append(str(url))
            goal = normalize_text(tool_args.get("goal", ""))
            return f"visit goal: {goal} | domains: {' '.join(sorted(domains))}"
        return json.dumps(
            canonical_tool_call(tool_name, tool_args), ensure_ascii=False, sort_keys=True
        )

    def _observation_text(self, result: ToolResult) -> str:
        if result.evidence_items:
            evidence_text = "\n".join(
                item.text for item in result.evidence_items if item.text.strip()
            )
            if evidence_text:
                return evidence_text[: self.config.max_observation_chars]
        return result.content[: self.config.max_observation_chars]

    def _switch_search_engines(self, tool_args: dict[str, Any]) -> dict[str, Any]:
        args = copy.deepcopy(tool_args)
        raw_queries = args.get("query", [])
        queries = raw_queries if isinstance(raw_queries, list) else [raw_queries]
        raw_engines = args.get("engine", [])
        engines = raw_engines if isinstance(raw_engines, list) else [raw_engines]
        switched = []
        for index, query in enumerate(queries):
            current = _normalize_engine(
                engines[index] if index < len(engines) else "auto"
            )
            if current == "auto":
                current = "bing" if _contains_chinese(str(query)) else "google"
            switched.append("bing" if current == "google" else "google")
        args["engine"] = switched
        return args

    def _guard_message(self, action: str, detail: str) -> str:
        recent_intents = [step.intent_text[:180] for step in list(self.steps)[-3:]]
        avoid = json.dumps(recent_intents, ensure_ascii=False)
        return (
            f'<trajectory_guard action="{action}" '
            f'no_new_evidence="{self.no_new_evidence_count}">\n'
            f"{detail}\nRecent unproductive intents to avoid: {avoid}\n"
            "Base the next decision on the original question and already collected evidence.\n"
            "</trajectory_guard>"
        )

    def _diagnostics(
        self,
        context: GuardContext,
        *,
        observation_similarity: float = 0.0,
        new_evidence_count: int = 0,
        confirmed_loop: bool = False,
    ) -> dict[str, Any]:
        return {
            "query_similarity": round(context.query_similarity, 4),
            "observation_similarity": round(observation_similarity, 4),
            "fingerprint_repeated": context.fingerprint_repeated,
            "scope_repeated": context.scope_repeated,
            "new_evidence_count": new_evidence_count,
            "no_new_evidence_count": self.no_new_evidence_count,
            "recovery_count": self.recovery_count,
            "confirmed_loop": confirmed_loop,
            "embedding_available": self.embedding_available,
        }

    def _log(
        self,
        event: str,
        context: GuardContext,
        decision: GuardDecision,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        if not self.config.log_enabled:
            return
        payload = {
            "run_id": self.run_id,
            "event": event,
            "round": context.round_idx + 1,
            "tool": context.tool_name,
            "fingerprint": context.fingerprint[:12],
            "action": decision.action,
            **(diagnostics or decision.diagnostics or self._diagnostics(context)),
        }
        print(f"[trajectory_guard] {json.dumps(payload, ensure_ascii=False)}")

    def _log_error(self, message: str) -> None:
        if self.config.log_enabled:
            payload = {"run_id": self.run_id, "event": "error", "message": message}
            print(f"[trajectory_guard] {json.dumps(payload, ensure_ascii=False)}")
