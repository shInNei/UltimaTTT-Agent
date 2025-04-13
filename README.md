# 🧠 UltimateTicTacAI

An AI Agent for **Ultimate Tic Tac Toe**, developed using **Monte Carlo Tree Search (MCTS)** and **Alpha-Beta Pruning** algorithms.

---

## 📌 Project Overview

This project is part of the **Introduction to AI** course (Semester 241) at **HCMUT**, where each group was required to build an AI agent that competes in a round-robin tournament playing **Ultimate Tic Tac Toe**.

Our goal was to explore and compare different decision-making techniques in adversarial games by implementing:
- **MCTS** – a search-based AI using simulations and playouts.
- **Alpha-Beta Pruning** – a classic adversarial search algorithm with custom evaluation functions.

---

## 📖 Game Rules

The rules of **Ultimate Tic Tac Toe** are available in the `docs/` folder (_Vietnamese only_).  
For English readers, please refer to this guide for an overview of the game mechanics:  
🔗 [https://www.ultraq.net/uttt/](https://www.ultraq.net/uttt/)

---

## 🗂 What's in the Repo?

- `state.py` — Core logic for handling the Ultimate Tic Tac Toe board and rules.
- `test_game.py` — Entry point for running and visualizing AI matches.
- `mcts_agent.py` — MCTS-based AI agent.
- `alpha_beta_agent.py` — Alpha-Beta Pruning-based AI agent.
- `play_auto()` — A helper function to simulate games between any two agents.
- `pygame/` — Visualization module for interactive observation using Pygame.

---

## ▶️ How to Run and Visualize

There are **two main ways** to watch AI agents compete:

### 1. Notebook (Jupyter)
You can use the `play_auto()` function to simulate a game directly:

```python
from test_game import play_auto
play_auto("mcts_agent", "alpha_beta_agent")

