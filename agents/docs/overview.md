Hybrid Reasoning System with Small Transformer & RP-Structured Core
Notes: Set things up to work in an env. 
    Keep documentation clean
    Keep track of changes in a single changelog
## 0. Purpose & Outcome

**Goal:**
Implement a small, local, hybrid reasoning system where:

* A **small transformer** (e.g., Phi-3 Mini) is used only as a **semantic tool**.
* A **stateful reasoning core** (RNN-like) operates on a **relational memory** shaped by RP-like constraints.
* A **controller** orchestrates modules and enforces **closure** (knowing when to stop).
* The system runs on a **single desktop PC** and exposes **transparent internal state** for inspection.

**Primary outputs (for users):**

* Text answers to questions/tasks (math stories, reasoning tasks, etc.).
* Optionally: structured reasoning traces (JSON) exposing internal steps.

**Primary outputs (for us):**

* A working prototype codebase with clear module boundaries.
* Logging / traces of internal state across reasoning steps.
* A scaffold that can later be trained / improved.

---

## 1. Architecture at a Glance

The system consists of **four main modules** plus glue:

1. **Semantic Engine (Transformer Module)**
2. **Reasoning Core (Stateful RNN-style Core)**
3. **Relational Memory (RP-Structured Graph/Store)**
4. **Controller & Closure (Orchestration + Halting)**

Data flow for one reasoning episode:

> Input text
> → Perception/encoding
> → Controller initializes state & memory
> → Reasoning Core loop (may call Transformer as needed)
> → Closure triggers
> → Controller asks Transformer to render final text
> → Output

---

## 2. Implementation Phases (for coding agents)

### Phase 1 – Project Skeleton & Core Types

**Objectives:**

* Establish a clean repo structure.
* Define core data types and interfaces.

**Tasks:**

1. **Set up repo**

   * Language: Python (3.10+)
   * Structure (example):

     ```
     /hybrid_reasoner
       /core          # reasoning core, closure
       /memory        # relational memory
       /semantic      # transformer wrappers
       /controller    # orchestration logic
       /config        # model paths, settings
       /tests
       main.py
     ```

2. **Define core data models (as Python dataclasses or Pydantic models)**

   * `Observation` – raw user input.
   * `ParsedTask` – normalized task representation after initial processing.
   * `RelationalState` – the RP-style graph/memory snapshot.
   * `CoreState` – internal hidden state of the reasoning core.
   * `StepTrace` – one step of reasoning (inputs, outputs, decisions).
   * `EpisodeTrace` – full reasoning trajectory for a single query.

3. **Decide and codify RP-ish schema for memory**

   Minimal fields:

   ```python
   class Entity:
       id: str
       type: str   # person, object, event, etc.

   class Relation:
       source_id: str
       relation_type: str  # "has", "gives", "before", "equals", etc.
       target_id: str
       metadata: dict

   class Constraint:
       name: str           # e.g., "conserve_apples"
       params: dict        # optional
   ```

   Wrap them in a `RelationalState` container.

---

### Phase 2 – Semantic Engine Wrapper (Transformer Module)

**Objectives:**

* Wrap a small local model (Phi-3, etc.) as a **tool function**, not as the main agent.
* Support a few specific call patterns.

**Tasks:**

1. **Implement a `SemanticEngine` class** that:

   * Uses either:

     * `llama.cpp` Python bindings, or
     * local HTTP server (e.g., Ollama) with a known model name.
   * Has deterministic settings (low temperature, fixed max tokens, clear stop tokens).

2. **Expose specific methods instead of a free-form `generate`:**

   * `parse_events(text: str) -> dict`

     * Ask the transformer to extract events and roles in structured JSON.
   * `summarize_state(rel_state: RelationalState) -> str`
   * `render_final_answer(rel_state: RelationalState) -> str`
   * `explain_reasoning(trace: EpisodeTrace) -> str` (optional later)

3. **Enforce SPD-style prompt structure internally**

   * Role: “You are a precise semantic parser / summarizer.”
   * Task frame: parse / summarize / render.
   * Scope: “Use only given information.”
   * Output shape: JSON / short sentence.
   * Closure: “Do not add extra commentary.”

4. **Add unit tests** to ensure:

   * JSON is parseable.
   * Events roughly match simple input tasks.

---

### Phase 3 – Relational Memory Implementation

**Objectives:**

* Implement a small, RP-flavored memory structure.
* Support read/write/update operations with constraints.

**Tasks:**

1. **Implement `RelationalMemory` class:**

   * Internal storage:

     * Use Python dicts / lists, optionally NetworkX for graph-like ops.
   * Methods:

     * `add_entity(entity: Entity)`
     * `add_relation(relation: Relation)`
     * `get_entity(id: str)`
     * `get_relations_for(entity_id: str)`
     * `apply_update(update: RelationalUpdate)` where `RelationalUpdate` describes changes.

2. **Define `RelationalUpdate` type**

   ```python
   class RelationalUpdate:
       add_entities: list[Entity]
       add_relations: list[Relation]
       remove_relations: list[Relation]  # by ID or tuple
       metadata: dict
   ```

3. **Add basic constraint checks (P4)**

   For early demos:

   * Quantity conservation (for specific scenarios like apples).
   * No duplicate IDs.
   * “All events processed” flag.

4. **Logging**

   * Implement a `to_dict()` method for `RelationalState` for easy JSON logging.
   * Enable snapshot logging after each reasoning step.

---

### Phase 4 – Reasoning Core (Stateful + Closure)

**Objectives:**

* Implement a small RNN/GRU/LSTM-based core.
* Support iterative reasoning and halting.

**Tasks:**

1. **Implement `ReasoningCore` class**

   * Backed by PyTorch.
   * Take as input:

     * embedding of current relational state (e.g., pooled features)
     * optional semantic hints from transformer
   * Output:

     * updated `CoreState` (hidden state tensor)
     * a `RelationalUpdate` object
     * a `halt_probability` scalar (0–1)

2. **Create a simple embedding of `RelationalState`**

   For MVP:

   * Count-based features (num entities, num events).
   * Simple numeric encodings of quantities.
   * Or a learned embedding projection from a flattened feature vector.

3. **Halting logic**

   * Implement external logic:

     ```python
     halt = halt_probability > threshold or step >= max_steps
     ```

   * Threshold and max steps configurable.

4. **Training stub (even if untrained at first)**

   * For MVP, core can be initialized with random weights and later replaced with a trained model.
   * Design the interface so later training doesn’t break the external API.

---

### Phase 5 – Controller & Episode Orchestration

**Objectives:**

* Implement the logic that glues everything together into a single episode.

**Tasks:**

1. **Implement `Controller` class**

   Responsibilities:

   * Accept `Observation` (user query).
   * Initialize `RelationalMemory` and `CoreState`.
   * Use `SemanticEngine.parse_events` to populate initial relational state.
   * Loop:

     * call `ReasoningCore.step(...)`
     * apply relational updates to memory
     * log a `StepTrace`
     * if `halt` → break
   * At the end:

     * call `SemanticEngine.render_final_answer(rel_state)`
     * build `EpisodeTrace`
     * return final answer + trace (optional)

2. **Define `StepTrace` and `EpisodeTrace`**

   ```python
   class StepTrace:
       step_idx: int
       core_state_summary: dict
       relational_state_snapshot: dict
       halt_probability: float
       notes: dict

   class EpisodeTrace:
       steps: list[StepTrace]
       final_answer: str
       final_rel_state: dict
   ```

3. **Add logging hooks**

   * Save trace as JSON file per episode (optional).
   * Provide methods to pretty-print the trace for inspection.

---

### Phase 6 – Evaluation Harness

**Objectives:**

* Provide a simple way to test and score the system on small tasks (like your apple/cookie/time problems).

**Tasks:**

1. **Create a `tasks/` directory**

   * JSON files describing tasks:

     * input text
     * ground truth answer
     * maybe ground truth relational sequence (later)

2. **Implement `evaluate.py`**

   * Load tasks.
   * Run controller.
   * Compare final answer to ground truth.
   * Output metrics:

     * accuracy
     * whether closure triggered correctly
     * number of steps per episode

3. **Optional: scoring with a big model**

   * Add a hook to send (input, output, trace) to GPT-4o/5.1 for qualitative scoring.

---

## 3. Constraints & Non-Goals for Coding Agents

* **Do NOT:**

  * Turn the transformer into the main loop.
  * Let the transformer free-generate long unbounded text.
  * Hide internal state updates (we want transparency).
  * Over-engineer training at this stage.

* **DO:**

  * Keep modules small and well-defined.
  * Preserve clean interfaces between:

    * SemanticEngine
    * ReasoningCore
    * RelationalMemory
    * Controller
  * Log internal steps in a structured, machine-readable way.
  * Make it easy to plug in a better-trained core later without rewriting the whole system.

---

## 4. RSA-Core Framing for Coding Agents (Short Version)

When coding agents read this, they should treat it as:

* **R (Roles / Relations):**
  Clear module boundaries, clear interface contracts, RP-style relations in memory.

* **S (Structure / State):**
  Explicit state in the Reasoning Core and Relational Memory; no hidden, magical global state.

* **A (Actions / Algorithms):**
  Well-defined steps: parse → initialize → iterate → halt → render.

The job is to **implement the scaffolding and flow**, not to perfect the reasoning from day one.

---


