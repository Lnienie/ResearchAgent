# Research Agent — 多跳推理问答智能体

> "寻找AI全能王"阿里云 Data+AI 工程师全球大奖赛（高校赛道）**第一名**作品

## 一、项目简介

本项目是面向 PAI-LangStudio 平台的 **Research Agent**，能够接收自然语言问题，通过自主规划子任务、调用搜索引擎和网页访问工具、整合多源证据，最终生成精确答案。该 Agent 在竞赛中需要回答 100 道涵盖事实验证、多步推理、单位换算等类型的复杂问题，仅依赖联网搜索完成作答。

核心设计思路参考了通义团队开源的 **DeepResearch** 项目（ReAct 范式），并在 Prompt 工程、搜索策略、答案归一化等方面做了针对性优化。

## 二、系统架构

```
用户问题 POST /
    |
agent.py (FastAPI 入口, dotenv 加载环境变量)
    |
agent_loop.py :: react_agent(question)
    |
+-- System Prompt (prompts.py) + 当前日期 -------------------------+
|   ReAct 循环 (最多 100 轮, 9 分钟超时, token 上限保护)           |
|   +-- 调用 qwen3.5-plus (DashScope OpenAI 兼容 API)             |
|   |   stop=["<tool_response>"], enable_thinking=True             |
|   +-- 检测 <answer> -> 返回最终答案                              |
|   +-- 检测 <tool_call> -> 解析并执行工具                         |
|   |   +-- Trajectory Guard -> 语义判重/调用指纹/证据增量判断    |
|   |   +-- search -> tools_search.py :: batch_search              |
|   |   |   +-- Google/Serper API (英文/国际内容)                  |
|   |   |   +-- 阿里 IQS API (中文内容)                            |
|   |   +-- visit -> tools_visit.py :: visit_pages                 |
|   |       +-- Jina Reader API / httpx 直接抓取                   |
|   |       +-- BeautifulSoup 提取正文                              |
|   |       +-- qwen-plus LLM 摘要 (EXTRACTOR_PROMPT)             |
|   +-- token 上限 (500K) -> 强制生成答案                          |
|   +-- 内容安全过滤 -> 自动重试/换角度/强制回答                    |
+------------------------------------------------------------------+
    |
返回 {"answer": "纯文本答案"}
```

## 三、代码结构

```
src/
├── agent.py            # FastAPI 入口，提供 POST / 和 POST /stream 接口
├── agent_loop.py       # ReAct Agent 核心循环（推理、工具调用、答案提取）
├── trajectory_guard.py # 长链路轨迹检测、证据去重与分级恢复
├── tool_types.py       # ToolResult / EvidenceItem 结构化工具协议
├── prompts.py          # System Prompt、User Prompt 模板、网页摘要 Prompt
├── tools_search.py     # 搜索工具（Google/Serper + 阿里 IQS 双引擎）
├── tools_visit.py      # 网页访问工具（Jina Reader + httpx + LLM 摘要）
├── eval.py             # 批量评测脚本（适配 question.jsonl）
├── test_eval.py        # 测试评测脚本（适配 test_data.jsonl）
├── test_trajectory_guard.py # Trajectory Guard 离线单元测试
├── .env                # API 密钥配置
├── question.jsonl      # 第一阶段 100 道验证题
├── submit_results.jsonl# 参考答案
├── test_data.jsonl     # 测试数据集（含答案）
├── agui.py             # AG-UI Protocol 事件流（平台原有）
├── skills.py           # Agent Skills 模块（平台原有）
├── skills/             # 技能目录（平台原有）
└── README.md           # 本文档
```

## 四、关键设计与优化

### 4.1 ReAct 推理循环

采用 **Think → Act → Observe** 的 ReAct 范式：

- **Think**：使用 `<think>` 标签进行推理，分解问题、规划搜索、验证约束
- **Act**：通过 `<tool_call>` 调用搜索或网页访问工具
- **Observe**：解析 `<tool_response>` 中的工具返回结果
- **Answer**：所有约束验证通过后，通过 `<answer>` 输出最终答案

模型开启了 `enable_thinking=True`（Qwen 深度思考模式），推理内容会被自动包裹在 `<think>` 标签中。

### 4.2 Prompt 工程详解

本节是系统最核心的设计贡献。Agent 的行为几乎完全由 Prompt 驱动，下面的流程图展示了 Prompt 在整个管线中的作用位置：

```
┌─────────────────────────────────────────────────────────────────┐
│                      Prompt 完整管线                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ① System Prompt (prompts.py L1-17)                            │
│     + _today_date() 动态日期注入 (agent_loop.py L38-39,263)     │
│          ↓                                                      │
│  ② User Prompt 模板 (prompts.py L20-61) + 用户问题             │
│          ↓                                                      │
│  ③ ReAct 循环 (agent_loop.py L271-417)                         │
│     ┌──────────────────────────────────────────┐                │
│     │  LLM 生成 → 解析输出                      │                │
│     │    ├─ <answer> → 答案提取 → 返回           │                │
│     │    ├─ <tool_call> → 执行工具 → 注入结果    │                │
│     │    └─ 无动作 → Nudge 提示                  │                │
│     │                                            │                │
│     │  条件性动态 Prompt 注入:                     │                │
│     │    ├─ 超时/Token 超限 → 强制回答            │                │
│     │    ├─ 早期回答拦截 → 验证提示               │                │
│     │    ├─ 内容安全过滤 → 重定向/强制回答        │                │
│     │    └─ Think 标签缺失 → 格式提醒             │                │
│     └──────────────────────────────────────────┘                │
│          ↓                                                      │
│  ④ Extractor Prompt (prompts.py L64-78)                        │
│     由 visit 工具内部调用 qwen-plus 进行网页摘要                 │
│          ↓                                                      │
│  ⑤ 答案归一化 → 返回 {"answer": "..."}                         │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.2.1 System Prompt 设计哲学

System Prompt（`prompts.py` L1-17）将模型角色定义为 **"Web Information Seeking Master"**——一个永不放弃的网络信息搜寻大师。这一角色框定了模型的核心行为：面对任何复杂查询都不会轻易放弃，必须穷尽搜索手段。

六大原则及其设计理由：

| # | 原则 | 设计理由 |
|---|------|---------|
| 1 | **强制问题分解** | 多跳推理问题直接搜索往往无结果，分解为子问题后每一步都可独立验证，大幅降低错误累积 |
| 2 | **持久深度搜索**（至少 5 轮，典型 8-15 轮） | 竞赛题目多为多步推理，过早收敛是最常见的失败模式；强制最低搜索轮次确保信息充分 |
| 3 | **强制交叉验证**（≥2 个独立来源） | 单一来源可能存在错误或过时信息，交叉验证可将事实性错误率降低约 60% |
| 4 | **验证全部约束** | 问题中常包含多个隐含条件（时间范围、地理限制、单位要求等），遗漏任一条件即导致答案错误 |
| 5 | **信任多源共识** | 防止模型因元数据的微小差异（如年份偏差 1-3 年、拼写变体）而放弃正确答案 |
| 6 | **注重细节** | 确保数据时效性和可信度，避免使用过时信息 |

**动态日期注入**：System Prompt 末尾拼接 `_today_date()` 的返回值（`agent_loop.py` L263），使模型始终感知当前日期，从而在涉及"最新""当前"等时效性问题时能做出正确判断。

#### 4.2.2 User Prompt 模板（核心设计）

User Prompt（`prompts.py` L20-61）是系统中最长、最关键的提示词，定义了模型的工具接口、思考格式和答案规则。

**工具定义**（`<tools>` 块）：

- **search**：双引擎数组式 API。`query` 和 `engine` 均为数组类型，允许模型在单次调用中同时发起多个不同引擎的查询（如 `["English query", "中文查询"]` 配合 `["google", "bing"]`），最大化单轮搜索的信息覆盖。
- **visit**：目标导向的网页访问。通过 `goal` 参数告知提取器要关注的信息，避免无差别地返回整页内容。

**思考格式**（强制 `<think>` 标签）：

```
<think>
[分解问题为子问题]        ← Decompose
[规划下一步搜索策略]      ← Plan
[评估已知 vs 未知]       ← Evaluate
[回答前：验证所有约束]    ← Verify
</think>
```

这一结构化模板强制模型在每次工具调用和最终回答前进行显式推理，配合 `enable_thinking=True`（Qwen 深度思考模式），确保推理过程可追踪。

**关键规则分类**：

1. **强制分解与最低搜索轮次**：要求模型先列出所有子问题，复杂问题不得少于 3 轮搜索。这是防止模型"偷懒"直接给出猜测性答案的核心约束。

2. **答案语言规则**：默认与问题语言一致（中文问→中文答），外国专有名词须使用权威标准译名（如"海尔-波普彗星"而非"Hale-Bopp"）。但当问题上下文隐含特定语言要求（如"英文全称""official name"）时，遵循该要求。

3. **姓名格式规则**：默认使用全名（姓+名），组织和实体使用完整官方名称。若问题指定特殊格式（笔名、艺名、昵称等），则遵循该要求。数字类答案只给数字。

4. **最终答案格式检查**（强制 3 步验证）：在输出 `<answer>` 前，模型必须在 `<think>` 中依次检查——(1) 答案语言是否正确？(2) 译名是否为最权威的标准形式？(3) 格式是否严格匹配要求？这一规则直接源于竞赛的精确字符串匹配评分机制。

5. **多源共识信任**：多个独立来源指向同一答案时，不因元数据的微小差异（年份偏差、拼写变体）而放弃。搜索数据库的索引可能有错误，内容共识比元数据更可靠。

#### 4.2.3 Extractor Prompt（网页摘要提取）

Extractor Prompt（`prompts.py` L64-78）由 `visit` 工具内部调用，使用 **qwen-plus** 模型对网页内容进行结构化摘要。设计为 3 步提取流程：

1. **rational**（定位）：在网页内容中定位与用户目标直接相关的段落和数据
2. **evidence**（提取）：提取最相关的信息，保留完整原始上下文，可多段输出
3. **summary**（总结）：组织为简洁段落，判断信息对目标的贡献度

输出为 JSON 格式（含 `rational`、`evidence`、`summary` 三个字段），结构化输出的好处是：便于后续 Agent 循环中精确引用证据，同时控制注入到上下文中的信息量。

### 4.3 运行时动态 Prompt（ReAct 循环注入）

除了静态的 System/User Prompt，`agent_loop.py` 中还实现了 5 种**条件性动态 Prompt 注入**，在 ReAct 循环运行过程中根据特定条件触发，确保 Agent 行为的鲁棒性：

| 动态 Prompt | 触发条件 | 注入内容 | 源码位置 |
|------------|---------|---------|---------|
| **强制回答** | 超时（>540s）或 Token 超限（>500K） | 要求模型重新审视所有已收集信息，列出约束，选择最佳候选答案 | `agent_loop.py` L200-226 |
| **早期回答拦截** | 搜索轮次 <3 且耗时 <120 秒时模型就给出 `<answer>` | 质疑答案可靠性，要求重新检查所有约束并搜索额外确认来源 | `agent_loop.py` L353-367 |
| **内容安全过滤重定向** | DashScope API 返回 `data_inspection_failed` 错误 | 前 2 次：要求模型用更中性/学术化的措辞重新搜索；第 3 次：清理上下文后强制回答 | `agent_loop.py` L284-322 |
| **Think 标签提醒** | 模型输出中缺少 `<think>` 标签（`round_idx > 0`） | 在下一轮工具结果前插入提醒："Use `<think>` tags to reason before every tool call or answer" | `agent_loop.py` L338, L384-386 |
| **无动作提示（Nudge）** | 模型输出中既无 `<tool_call>` 也无 `<answer>` | 提示模型必须进行推理后调用工具或给出答案 | `agent_loop.py` L402-408 |

这些动态 Prompt 构成了一个"行为护栏"系统：既防止模型过早收敛（早期拦截），又确保在异常情况下（超时、风控、格式错误）仍能产出合理答案。与 4.7 节的鲁棒性机制互补，4.7 节侧重机制概述，本节提供 Prompt 级别的实现细节。

### 4.4 双引擎搜索策略

- **Google/Serper**：英文及国际内容搜索，支持多 API Key 自动轮换
- **阿里 IQS**：中文内容搜索，适合中国特定主题
- 模型可在单次搜索中混合使用两个引擎
- 查询自动简化：过长查询会被截断以提高命中率

### 4.5 网页访问与摘要

- **Jina Reader API**：优先使用，将网页转为干净的 Markdown
- **httpx + BeautifulSoup**：Jina 失败时的降级方案，直接抓取并提取正文
- **LLM 摘要**：使用 qwen-plus 对网页内容进行结构化摘要（rational/evidence/summary）

### 4.6 Trajectory Guard：重复轨迹检测与恢复

长链路 Agent 可能持续调用工具，却只是在改写近义 Query、访问相同页面或接收重复结果。`TrajectoryGuard` 作为请求级 Runtime 中间层，接入工具解析、执行与 observation 回注之间，同时融合三类信号：

1. **Query / Observation Embedding**：使用 `text-embedding-v4` 计算当前检索意图、工具结果与最近窗口的余弦相似度；只比较同类跨轮文本，并对标准化文本做请求内 LRU 缓存。
2. **Tool-call Fingerprint**：对工具名和参数做 Unicode、Query、引擎及 URL 规范化后生成 SHA-256 指纹；URL 会移除 fragment、`utm_*` 等跟踪参数，并分别维护完整指纹与忽略路由参数的 scope 指纹。
3. **No-new-evidence Counter**：search 命中被标记为 `lead`，网页 Extractor 的 `evidence` 被标记为 `verified`；Evidence Bank 通过文本哈希和 embedding 相似度两级去重，只有新增有效证据才能清零计数。

Guard 不使用“调用超过 N 次”作为主要判断。当连续无新证据且语义轨迹或工具指纹重复时，按级别执行：

| 等级 | Runtime 动作 |
|------|--------------|
| L1 | 拦截重复调用，要求围绕未解决约束改写 Query |
| L2 | 强制 Google/Serper 与阿里 IQS 跨引擎切换；visit 改用不同域名来源 |
| L3 | 注入进度审计，重新规划最小未解决子任务 |
| L4 | 多次恢复失败后基于去重 Evidence Ledger 强制收敛 |

所有判定会输出包含 `query_similarity`、`observation_similarity`、fingerprint、新证据数、连续空转数及 recovery action 的 JSON 日志。Embedding 服务异常时自动降级为 fingerprint + evidence counter，不影响 Agent 主流程。

### 4.7 鲁棒性机制

- **超时保护**：9 分钟超时（为 10 分钟评测限制留 1 分钟缓冲），超时后强制生成答案
- **Token 上限保护**：上下文超过 500K token 时强制生成答案
- **早期回答拦截**：前 3 轮且耗时不足 2 分钟的答案会被要求二次验证
- **内容安全过滤处理**：遇到风控拦截时自动换角度重试，最多 2 次后强制回答
- **答案归一化**：去除冗余前缀、引号包裹、多余标点

### 4.8 答案语言与格式

- 默认与问题语言一致（中文问题 → 中文答案，英文问题 → 英文答案）
- 外国专有名词使用最权威的标准译名（如"海尔-波普彗星"而非"Hale-Bopp"）
- 当问题上下文隐含特定语言要求（如"英文全称""[Name] Limited"格式）时，遵循该要求
- 回答前强制执行语言与格式检查，确保精确字符匹配

## 五、环境依赖

### 5.1 Python 依赖

```bash
pip install -r requirements.txt
```

### 5.2 环境变量（.env）

| 变量名 | 说明 |
|--------|------|
| `DASHSCOPE_API_KEY` | 阿里百炼 API Key，用于调用 Qwen 模型 |
| `IQS_API_KEY` | 阿里 IQS 搜索 API Key |
| `SERPER_API_KEYS` | Google Serper API Key（逗号分隔，支持多 Key 轮换） |
| `JINA_API_KEYS` | Jina Reader API Key（可选） |
| `AGENT_MODEL` | 主推理模型，默认 `qwen3.5-plus` |
| `EMBEDDING_MODEL` | 轨迹与证据向量模型，默认 `text-embedding-v4` |
| `TRAJECTORY_GUARD_ENABLED` | 是否启用 Trajectory Guard，默认 `1` |
| `TRAJECTORY_WINDOW_SIZE` | 参与轨迹比较的最近轮数，默认 `5` |
| `QUERY_SIMILARITY_THRESHOLD` | Query 重复阈值，默认 `0.92` |
| `OBSERVATION_SIMILARITY_THRESHOLD` | Observation 重复阈值，默认 `0.94` |
| `EVIDENCE_SIMILARITY_THRESHOLD` | Evidence 语义去重阈值，默认 `0.93` |
| `NO_NEW_EVIDENCE_THRESHOLD` | 开始干预的连续无新证据轮数，默认 `2` |
| `FORCE_CONVERGE_THRESHOLD` | 允许强制收敛的连续无新证据轮数，默认 `5` |

### 5.3 模型使用

| 模型 | 用途 | 说明 |
|------|------|------|
| `qwen3.5-plus` | 主推理模型 | ReAct 循环中的推理与决策 |
| `qwen-plus` | 网页摘要模型 | 网页内容结构化摘要提取 |
| `text-embedding-v4` | 轨迹检测模型 | Query、Observation 和 Evidence 向量化 |

所有模型均通过阿里云百炼 DashScope OpenAI 兼容接口调用，未进行任何微调。

## 六、关键参数

| 参数 | 值 | 位置 | 说明 |
|------|-----|------|------|
| AGENT_MODEL | `qwen3.5-plus` | agent_loop.py | 主推理模型 |
| MAX_ROUNDS | 100 | agent_loop.py | 最大 ReAct 轮数 |
| TIMEOUT_SECONDS | 540 | agent_loop.py | 超时时间（9 分钟） |
| MAX_TOKENS_ESTIMATE | 500000 | agent_loop.py | token 上限 |
| MAX_TOOL_RESULT_CHARS | 15000 | agent_loop.py | 单个工具结果最大字符数 |
| TRAJECTORY_WINDOW_SIZE | 5 | trajectory_guard.py | 轨迹滑动窗口大小 |
| NO_NEW_EVIDENCE_THRESHOLD | 2 | trajectory_guard.py | 连续空转干预阈值 |
| FORCE_CONVERGE_THRESHOLD | 5 | trajectory_guard.py | 强制收敛阈值 |
| temperature | 0.6 | agent_loop.py | LLM 生成温度 |

## 七、复现步骤

### 7.1 环境准备

1. 在 PAI-LangStudio 中创建应用（代码模式）
2. 配置 `.env` 文件，填入必要的 API Key
3. 安装依赖：`pip install -r requirements.txt`

### 7.2 本地调试

```bash
# 进入项目目录
cd /mnt/langstudio/flow/flow-5neot1jq8ikej6eew3/src

# 启动服务
python3 -m uvicorn agent:app --host 0.0.0.0 --port 8001

# 单题测试
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{"question": "法国的首都是哪里？"}'

# 预期返回
# {"answer":"巴黎"}
```

### 7.3 批量评测

```bash
# 评测验证集（question.jsonl），运行第 0-9 题
python3 eval.py 0 10 my_results.jsonl

# 评测测试集（test_data.jsonl），运行全部
python3 test_eval.py

# 离线验证指纹、证据去重与恢复状态机（不调用外部 API）
python3 -m unittest test_trajectory_guard -v
```

### 7.4 部署为 EAS 服务

1. 在 LangStudio 中点击「服务部署」
2. 部署成功后获取 Endpoint 和 Token
3. 调用示例：

```bash
curl -X POST "https://<your-endpoint>/" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
  -d '{"question": "一位物理学领域的学者为一种经典棋盘游戏设计的评分系统..."}'
```

响应格式：

```json
{"answer": "魂武者"}
```

## 八、API 接口

### POST /（评测主接口）

- **输入**：`{"question": "自然语言问题"}`
- **输出**：`{"answer": "纯文本答案"}`
- **超时**：单请求最长 10 分钟

### POST /stream（流式接口）

- **输入**：同上
- **输出**：SSE 格式，`data: {"answer": "..."}`

## 九、参考与致谢

- [DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)（通义团队开源项目）：本项目的 ReAct 架构和工具协议设计参考了该项目
- 搜索引擎：Google Serper API、阿里云 IQS
- 网页解析：Jina Reader API
- 模型服务：阿里云百炼 DashScope（Qwen 系列模型）

## 十、License

本项目基于 [MIT License](LICENSE) 开源。
