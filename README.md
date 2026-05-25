# Mastering LangGraph: Advanced AI Agent Architectures

A comprehensive, hands-on repository demonstrating the implementation of LLM agents, custom control flows, and advanced orchestration patterns using **LangGraph**. This project transitions from basic linear chains to complex, stateful, multi-agent systems with human-in-the-loop verification and persistence.

---

## 🚀 Repository Overview

This repository serves as a practical playbook for building production-grade LLM applications. It covers core LangGraph concepts including State Graphs, conditional routing, memory persistence, streaming, and human-in-the-loop interaction.

### 🛠️ Tech Stack
* **Orchestration:** LangGraph, LangChain
* **Languages & Runtime:** Python, Jupyter Notebooks
* **Models:** [Insert your primary LLMs here, e.g., OpenAI GPT-4o, Anthropic Claude 3.5 Sonnet, Groq]

---

## 📂 Core Implementations & Roadmap

The codebase is organized into progressive milestones, tracking the evolution from simple graphs to fully autonomous agents.

### 1. Foundations & Workflows
Basic structural patterns for controlling execution flow.
* **`linear_wf.ipynb` & `iterative_wf.ipynb` / `iterative_ex.ipynb`:** Building predictable, sequential node executions and deterministic looping mechanisms.
* **`conditional_wf.ipynb` & `llm_conditional_wf.ipynb`:** Implementing routing logic where the next node is determined dynamically (both via code logic and LLM classification).
* **`parallel_wf.ipynb`:** Forking and joining execution paths to run multiple LLM calls or tool tasks concurrently for speed optimization.

### 2. State, Memory & Persistence
Managing context and long-term token efficiency.
* **`persistence.ipynb` & `base_store.ipynb`:** Utilizing Checkpointers to save graph state automatically after every node execution.
* **`short_term_memory.ipynb` & `shorttermmem.py`:** Managing in-flight conversation history within the graph state to maintain context across multi-turn interactions.

### 3. Production Agent Architectures
Autonomous loops and specialized functional agents.
* **`chat_bot.ipynb`:** A foundational conversational interface built natively on state graphs.
* **`ReAct_agent.ipynb`:** Implementation of the classic **Reasoning + Acting** loop, allowing the LLM to dynamically call tools based on user input.
* **`email_agent.ipynb` & `blog_writing_agent.py` / `blog.md`:** Specialized, task-oriented agents designed for automated content creation, filtering, and drafting.
* **`image_generation_runnable.py`:** Integrating multimodal execution steps natively inside graph nodes.

### 4. Advanced UX & Human-in-the-Loop (HITL)
Patterns required for safe, predictable enterprise deployments.
* **`human_in_the_loop.ipynb`:** Implementing breakpoints to interrupt graph execution, allowing humans to review, approve, or edit state before moving to high-risk nodes (e.g., sending emails or writing to databases).
* **`streaming.ipynb`:** Streaming both internal token outputs and node-by-node state updates in real-time for responsive user interfaces.

---
