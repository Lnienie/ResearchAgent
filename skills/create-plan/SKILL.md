---
name: create-plan
description: Create a concise, actionable plan. Use when a user explicitly asks for a plan or next steps.
metadata:
  short-description: Create a plan
---

# Create Plan

## Goal

Turn a user request into **one clear, actionable plan** that an agent (chat, websearch, or tool-using) can directly follow.

The plan should be concise, structured, and focused on *what to do next*, not how to implement details.

## Operating principles

- Operate in **read-only / planning mode** (no execution).
- Prefer **clarity over completeness**.
- Optimize for **agent executability**: each step should be something an agent can actually do.
- Assume the agent may have access to tools such as search, browsing, or analysis, but do not depend on any specific tool unless required.

## Minimal workflow

1. **Understand the request and context**
   - Identify the user’s goal, success criteria, and constraints.
   - Use available context (conversation, provided links, known domain knowledge).
   - If relevant, note assumptions about audience, timeframe, or format.

2. **Ask follow-ups only if blocking**
   - Ask **at most 1–2 questions**.
   - Only ask if the plan would be meaningfully wrong without the answer.
   - Prefer assumptions over questions when safe; make assumptions explicit if needed.

3. **Create the plan using the template below**
   - Start with **1–3 sentences** describing intent and overall approach.
   - Clearly define **what is in scope and out of scope**.
   - Provide a **short, ordered checklist** of concrete actions (default 6–10).
     - Steps should be atomic, verb-first, and outcome-oriented.
     - Typical flow: understanding → gathering info → analysis → synthesis → validation → delivery.
   - Include:
     - At least one **validation / quality check** step.
     - At least one **risk, edge case, or uncertainty** step when applicable.
   - If important unknowns remain, add a small **Open questions** section (max 3).

4. **Output only the plan**
   - Do not include meta commentary, explanations, or alternatives.
   - Do not include implementation details beyond what’s necessary for planning.

## Plan template (follow exactly)

```markdown
# Plan

<1–3 sentences: what we’re doing, why, and the high-level approach.>

## Scope
- In:
- Out:

## Action items
[ ] <Step 1>
[ ] <Step 2>
[ ] <Step 3>
[ ] <Step 4>
[ ] <Step 5>
[ ] <Step 6>

## Open questions
- <Question 1>
- <Question 2>
- <Question 3>
