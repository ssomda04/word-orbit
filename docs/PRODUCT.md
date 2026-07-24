# Product Overview — Contextle

> Working name. The repository may keep its original name; docs use **Contextle**.

## Problem

Traditional word games (Wordle, 꼬맨틀-style) test spelling or letter overlap.
Contextle instead rewards **semantic reasoning**: players learn a hidden word by
probing the *meaning space* around it. It doubles as a hands-on way to
experience word embeddings, vector geometry, and (later) attention.

## Target users

- Casual players who enjoy 꼬맨틀 / Semantle-style deduction games.
- Students and educators exploring embeddings and vector-space intuition.
- Ourselves — as a vehicle to connect course theory to a real, shippable service.

## How it plays

A guess is scored by the **semantic similarity** (cosine of embedding vectors)
between the guess and the hidden answer — not by shared letters. Players use the
running list of scored guesses to triangulate the answer.

### Single-player (MVP focus)

1. Start a game → server secretly picks an answer word.
2. Submit a guess → server validates, embeds, and returns **similarity** (and,
   later, a **rank** and a **3D coordinate**).
3. The guess history (sorted by similarity) guides the next guess.
4. After enough guesses, a **3D semantic map** appears.
5. On winning, the answer's position and the player's exploration path are revealed.

### Competitive multiplayer (later)

- Players race on the **same** answer.
- Opponents' **best similarity** and **attempt counts** may be shown — but never
  their actual words, and never the answer, before the round ends.

### Cooperative multiplayer (later)

- Participants share their guess words and work toward the answer together.
- All players' guesses appear on one shared 3D map.

### 3D semantic map

- Guesses are projected to 3D (PCA first, UMAP later) for visualization.
- The answer's location is **hidden** during play and **revealed** at the end.
- The player's guess order is drawn as an exploration path.
- ⚠️ The projection is a lossy view; on-screen distance ≠ exact similarity.

### Sentence mode (future / Phase 4)

- Extends word-level guessing to **sentence-level** semantic comparison, so two
  sentences can be "close" without sharing words.
- Optional educational **attention visualization** (heatmap / token links) for a
  chosen layer/head. Attention weights are an aid to intuition, **not** a
  complete explanation of the model's decision.

## Out of scope (for now)

- Full game UI, real multiplayer, authentication.
- Databases beyond what a feature actually needs (Redis/Postgres come with
  multiplayer/history — see [ROADMAP.md](./ROADMAP.md)).
- Downloading large models, GPU training, or fine-tuning during setup.
- 3D rendering libraries (chosen after a comparison in Phase 2).

## Success signals (early)

- A new teammate can clone, run, and see frontend↔backend connected in minutes.
- Frontend, backend, and ML can progress without blocking each other.
- Swapping the mock embedding for a real model requires **no** caller changes.
