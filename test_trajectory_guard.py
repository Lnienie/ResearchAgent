import unittest
from unittest.mock import AsyncMock, patch

import agent_loop
import tools_search
import tools_visit
from tool_types import EvidenceItem, ToolResult
from tools_visit import _parse_extraction
from trajectory_guard import (
    GuardConfig,
    TrajectoryGuard,
    canonicalize_url,
    tool_call_fingerprint,
)


class DeterministicEmbedder:
    """Offline embedder with explicit semantic groups for state-machine tests."""

    async def embed(self, texts):
        vectors = []
        for text in texts:
            normalized = text.lower()
            if "semantic duplicate" in normalized:
                vectors.append([0.0, 0.0, 1.0, 0.0])
            elif "novel fact one" in normalized:
                vectors.append([0.0, 0.0, 0.0, 1.0])
            elif "novel fact two" in normalized:
                vectors.append([0.0, 0.0, 1.0, 0.0])
            elif "alpha" in normalized:
                vectors.append([1.0, 0.0, 0.0, 0.0])
            elif "same observation" in normalized:
                vectors.append([0.0, 1.0, 0.0, 0.0])
            else:
                vectors.append([0.5, 0.5, 0.5, 0.5])
        return vectors


class FailingEmbedder:
    async def embed(self, texts):
        raise RuntimeError("offline")


class ZeroEmbedder:
    async def embed(self, texts):
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


def guard_config(**overrides):
    values = {
        "enabled": True,
        "window_size": 5,
        "query_similarity_threshold": 0.92,
        "observation_similarity_threshold": 0.94,
        "evidence_similarity_threshold": 0.93,
        "no_new_evidence_threshold": 2,
        "force_converge_threshold": 5,
        "log_enabled": False,
    }
    values.update(overrides)
    return GuardConfig(**values)


class FingerprintTests(unittest.TestCase):
    def test_search_fingerprint_ignores_format_and_batch_order(self):
        left = {
            "query": ["  Alpha   Query! ", "中文 查询"],
            "engine": ["Serper", "IQS"],
        }
        right = {
            "query": ["中文 查询", "alpha query"],
            "engine": ["bing", "google"],
        }
        self.assertEqual(
            tool_call_fingerprint("search", left),
            tool_call_fingerprint("SEARCH", right),
        )

    def test_visit_fingerprint_removes_tracking_parameters(self):
        left = {
            "url": "https://Example.com/page/?utm_source=x&id=2#section",
            "goal": "Find Alpha",
        }
        right = {
            "url": ["https://example.com/page?id=2"],
            "goal": "find alpha",
        }
        self.assertEqual(
            tool_call_fingerprint("visit", left),
            tool_call_fingerprint("visit", right),
        )
        self.assertEqual(
            canonicalize_url(left["url"]), "https://example.com/page?id=2"
        )


class TrajectoryGuardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.guard = TrajectoryGuard(
            "Find alpha",
            config=guard_config(),
            embedder=DeterministicEmbedder(),
            run_id="test",
        )
        self.search_args = {"query": ["alpha query"], "engine": ["google"]}

    async def _execute_empty(self, round_idx, args=None):
        args = args or self.search_args
        pre = await self.guard.before_tool(round_idx, "search", args)
        if pre.block_execution or pre.force_converge:
            return pre, None
        post = await self.guard.after_tool(
            round_idx,
            "search",
            pre.tool_args,
            ToolResult(content="same observation", success=False),
            context=pre.context,
        )
        return pre, post

    async def test_repeated_call_is_blocked_before_second_network_request(self):
        _, first_post = await self._execute_empty(0)
        self.assertEqual(first_post.action, "continue")
        self.assertEqual(self.guard.no_new_evidence_count, 1)

        second_pre = await self.guard.before_tool(1, "search", self.search_args)
        self.assertTrue(second_pre.block_execution)
        self.assertEqual(second_pre.action, "rewrite_query")
        self.assertEqual(self.guard.no_new_evidence_count, 2)

    async def test_recovery_escalates_to_engine_switch_and_replan(self):
        await self._execute_empty(0)
        rewrite = await self.guard.before_tool(1, "search", self.search_args)
        self.assertEqual(rewrite.action, "rewrite_query")

        switch = await self.guard.before_tool(2, "search", self.search_args)
        self.assertEqual(switch.action, "switch_engine")
        self.assertFalse(switch.block_execution)
        self.assertEqual(switch.tool_args["engine"], ["bing"])

        post = await self.guard.after_tool(
            2,
            "search",
            switch.tool_args,
            ToolResult(content="same observation", success=False),
            context=switch.context,
        )
        self.assertEqual(post.action, "replan_subtask")

    async def test_recovery_eventually_forces_convergence(self):
        await self._execute_empty(0)
        await self.guard.before_tool(1, "search", self.search_args)  # rewrite
        switch = await self.guard.before_tool(2, "search", self.search_args)
        await self.guard.after_tool(
            2,
            "search",
            switch.tool_args,
            ToolResult(content="same observation", success=False),
            context=switch.context,
        )
        await self.guard.before_tool(3, "search", self.search_args)  # replan
        forced = await self.guard.before_tool(4, "search", self.search_args)
        self.assertTrue(forced.force_converge)
        self.assertEqual(forced.action, "force_converge")

    async def test_new_evidence_prevents_false_positive_and_resets_counter(self):
        first_pre = await self.guard.before_tool(0, "search", self.search_args)
        first = ToolResult(
            content="novel fact one",
            evidence_items=[
                EvidenceItem(
                    text="Novel fact one contains enough useful source information.",
                    source_url="https://one.example/a",
                    kind="lead",
                )
            ],
        )
        first_post = await self.guard.after_tool(
            0, "search", first_pre.tool_args, first, context=first_pre.context
        )
        self.assertEqual(first_post.diagnostics["new_evidence_count"], 1)
        self.assertEqual(self.guard.no_new_evidence_count, 0)

        second_pre = await self.guard.before_tool(1, "search", self.search_args)
        self.assertFalse(second_pre.block_execution)
        second = ToolResult(
            content="novel fact two",
            evidence_items=[
                EvidenceItem(
                    text="Novel fact two provides a different independently sourced detail.",
                    source_url="https://two.example/b",
                    kind="lead",
                )
            ],
        )
        second_post = await self.guard.after_tool(
            1, "search", second_pre.tool_args, second, context=second_pre.context
        )
        self.assertEqual(second_post.action, "continue")
        self.assertEqual(second_post.diagnostics["new_evidence_count"], 1)
        self.assertEqual(self.guard.no_new_evidence_count, 0)

    async def test_semantically_duplicate_evidence_does_not_reset_counter(self):
        first_args = {"query": ["first topic"], "engine": ["google"]}
        first_pre = await self.guard.before_tool(0, "search", first_args)
        first_result = ToolResult(
            content="first",
            evidence_items=[
                EvidenceItem(
                    text="Semantic duplicate version one with sufficient factual detail.",
                    source_url="https://one.example",
                )
            ],
        )
        await self.guard.after_tool(
            0, "search", first_pre.tool_args, first_result, context=first_pre.context
        )

        second_args = {"query": ["different topic"], "engine": ["bing"]}
        second_pre = await self.guard.before_tool(1, "search", second_args)
        second_result = ToolResult(
            content="second",
            evidence_items=[
                EvidenceItem(
                    text="Semantic duplicate version two from a mirror source with detail.",
                    source_url="https://mirror.example",
                )
            ],
        )
        second_post = await self.guard.after_tool(
            1, "search", second_pre.tool_args, second_result, context=second_pre.context
        )
        self.assertEqual(second_post.diagnostics["new_evidence_count"], 0)
        self.assertEqual(self.guard.no_new_evidence_count, 1)

    async def test_verified_evidence_from_new_domain_counts_as_corroboration(self):
        first_args = {"url": ["https://one.example/a"], "goal": "alpha fact"}
        first_pre = await self.guard.before_tool(0, "visit", first_args)
        first_result = ToolResult(
            content="semantic duplicate",
            evidence_items=[
                EvidenceItem(
                    text="Semantic duplicate verified claim with sufficient detail.",
                    source_url="https://one.example/a",
                    kind="verified",
                    quality=0.9,
                )
            ],
        )
        await self.guard.after_tool(
            0, "visit", first_pre.tool_args, first_result, context=first_pre.context
        )

        second_args = {"url": ["https://two.example/b"], "goal": "alpha fact"}
        second_pre = await self.guard.before_tool(1, "visit", second_args)
        second_result = ToolResult(
            content="semantic duplicate",
            evidence_items=[
                EvidenceItem(
                    text="Semantic duplicate verified claim restated with sufficient detail.",
                    source_url="https://two.example/b",
                    kind="verified",
                    quality=0.9,
                )
            ],
        )
        second_post = await self.guard.after_tool(
            1, "visit", second_pre.tool_args, second_result, context=second_pre.context
        )
        self.assertEqual(second_post.diagnostics["new_evidence_count"], 1)
        self.assertEqual(self.guard.no_new_evidence_count, 0)

    async def test_embedding_failure_falls_back_to_fingerprint_and_counter(self):
        guard = TrajectoryGuard(
            "Find alpha",
            config=guard_config(),
            embedder=FailingEmbedder(),
            run_id="fallback",
        )
        args = {"query": ["alpha query"], "engine": ["google"]}
        first_pre = await guard.before_tool(0, "search", args)
        await guard.after_tool(
            0,
            "search",
            args,
            ToolResult(content="empty", success=False),
            context=first_pre.context,
        )
        second_pre = await guard.before_tool(1, "search", args)
        self.assertTrue(second_pre.block_execution)
        self.assertEqual(second_pre.action, "rewrite_query")
        self.assertFalse(second_pre.diagnostics["embedding_available"])


class ExtractorParsingTests(unittest.TestCase):
    def test_extractor_keeps_all_three_structured_fields(self):
        result = _parse_extraction(
            '```json\n{"rational":"why", "evidence":"fact", "summary":"result"}\n```'
        )
        self.assertTrue(result.parsed)
        self.assertEqual(result.rational, "why")
        self.assertEqual(result.evidence, "fact")
        self.assertEqual(result.summary, "result")
        self.assertIn("Rational", result.render())


class ToolAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_adapter_preserves_text_and_evidence(self):
        structured = ToolResult(
            content="A search result with enough content. " * 5,
            evidence_items=[
                EvidenceItem(
                    text="A sufficiently detailed search lead for alpha.",
                    source_url="https://source.example/a",
                )
            ],
            metadata={"engine": "google", "provider": "serper"},
        )
        with patch(
            "tools_search._serper_search_result", return_value=structured
        ):
            result = tools_search.batch_search_result(
                ["alpha query"], engines=["google"]
            )
        self.assertEqual(result.content, structured.content)
        self.assertEqual(len(result.evidence_items), 1)
        self.assertEqual(result.metadata["queries"][0]["selected_engine"], "google")

    async def test_visit_adapter_combines_page_evidence(self):
        async def fake_visit(url, goal):
            return ToolResult(
                content=f"content:{url}",
                evidence_items=[
                    EvidenceItem(
                        text=f"Verified evidence from {url} with sufficient detail.",
                        source_url=url,
                        kind="verified",
                        quality=0.9,
                    )
                ],
                metadata={"url": url, "goal": goal},
            )

        with patch("tools_visit.visit_page_result", side_effect=fake_visit):
            result = await tools_visit.visit_pages_result(
                ["https://one.example", "https://two.example"], "alpha"
            )
        self.assertEqual(len(result.evidence_items), 2)
        self.assertIn("=======", result.content)
        self.assertTrue(result.success)


class AgentLoopIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_react_loop_injects_guard_recovery_messages(self):
        tool_call = (
            '<think>search</think><tool_call>{"name":"search","arguments":'
            '{"query":["alpha query"],"engine":["google"]}}</tool_call>'
        )
        llm_responses = [tool_call, tool_call, tool_call, "<answer>Alpha</answer>"]
        snapshots = []

        async def fake_llm(messages, **kwargs):
            snapshots.append([dict(message) for message in messages])
            return llm_responses.pop(0)

        guard = TrajectoryGuard(
            "Find alpha",
            config=guard_config(),
            embedder=ZeroEmbedder(),
            run_id="integration",
        )
        empty_result = ToolResult(content="same observation", success=False)

        with (
            patch("agent_loop.TrajectoryGuard", return_value=guard),
            patch("agent_loop._call_llm", side_effect=fake_llm),
            patch(
                "agent_loop._execute_tool",
                new=AsyncMock(return_value=empty_result),
            ) as execute_tool,
        ):
            answer = await agent_loop.react_agent("Find alpha")

        self.assertEqual(answer, "Alpha")
        self.assertEqual(execute_tool.await_count, 3)
        final_context = "\n".join(
            message["content"] for message in snapshots[-1]
        )
        self.assertIn('action="rewrite_query"', final_context)
        self.assertIn('action="switch_engine"', final_context)


if __name__ == "__main__":
    unittest.main()
