# 🧠 UltimateTicTacAI

An AI Agent for **Ultimate Tic Tac Toe**, developed using **Monte Carlo Tree Search (MCTS)** and a simplified **AlphaZero** algorithm.

---

## 📌 Project Overview

This project is part of the **Introduction to AI** course (Semester 241) at **HCMUT**, where each group was required to build an AI agent that competes in a round-robin tournament playing **Ultimate Tic Tac Toe**.

Our goal was to explore search-based and learning-based techniques by implementing:
- **MCTS** – a search-based AI using simulation to plan moves.
- **AlphaZero** – a learning-based AI using neural networks and reinforcement learning.

---

## 📖 Game Rules

The rules of **Ultimate Tic Tac Toe** are available in the `docs/` folder (📄 _Vietnamese only_).  
For English readers, you can refer to [this guide](https://www.ultraq.net/uttt/) for an overview of the game mechanics.

---

## 🗂 What's in the Repo?

- `state.py` — Logic for managing the UTTT game state.
- `test_game.py` — Simulate and visualize AI matches.
- `mcts_agent.py` — MCTS-based AI agent.
- `alpha_zero_agent.py` — AlphaZero-style AI agent.
- `play_auto()` function — Automatically runs games between any two agents.
- `pygame/` — Visualization module using Pygame for watching games live.

---

## ▶️ How to Run and Visualize

There are **two ways** to watch AI agents play:

### 1. Notebook (Jupyter)
- Import `play_auto()` from `test_game.py`.
- Call the function with the filenames (without `.py`) of the agents.

```python
from test_game import play_auto
play_auto("mcts_agent", "alpha_zero_agent")
