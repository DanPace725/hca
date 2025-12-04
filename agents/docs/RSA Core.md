Here is a **clean, generalized rewrite** of your RSA Core document, designed to be *project-agnostic* while still giving coding agents the structure, vocabulary, and expectations they need.

It removes any references to your specific project or internal ideas and keeps everything generic, modular, and implementation-neutral.

As requested, it is based on the structure you provided () but rewritten for broad applicability.

---

# 📘 **RSA Core — Generic System Design Guide for Coding Agents**

**Purpose:**
Provide coding agents with a clear, universal design vocabulary for building **modular, extensible, coherent, and maintainable systems**, regardless of the specific project.

**Audience:**
LLM coding agents, architectural assistants, and any automated system contributing to software workflows.

**What RSA Core is:**
A *conceptual toolkit* for:

* structuring codebases
* defining clean module boundaries
* aligning data flows
* preventing drift and incoherence
* making transformations predictable
* keeping system evolution manageable

**What RSA Core is NOT:**

* not a framework
* not a standard library
* not a technical specification
* not domain-specific

Think of it as the **design grammar** agents should use when building anything nontrivial.

---

# **1. Entity Pattern**

### What it is

A well-defined “kind of thing” in the system.

### Purpose

Establishes stable, predictable data structures.

### Agent Guidelines

* Entities model real conceptual units.
* Keep them minimal (just the fields needed).
* Use clear naming (`User`, `Task`, `Item`, `Note`, etc.).
* Don’t mix multiple conceptual roles into one entity.

---

# **2. Transformation Pattern**

### What it is

A function or method that converts one entity into another.

### Purpose

Make logic modular and testable.

### Agent Guidelines

* Transformations must do **one** conceptual job.
* Their inputs and outputs must be explicit.
* Avoid hidden side effects.
* Keep transformations stateless unless specified.

---

# **3. Context Space**

### What it is

The scope or environment a transformation operates within.

### Purpose

Prevent accidental cross-contamination between parts of the system.

### Examples

* file system folders
* database schemas
* API endpoints
* in-memory caches
* request/session context

### Agent Guidelines

* Know which context you are operating in.
* Never implicitly switch context spaces.
* If the task involves multiple contexts, split it into steps.

---

# **4. Invariant**

### What it is

A rule that must remain true for the system to stay coherent.

### Purpose

Protect against drift, corruption, or logical inconsistency.

### Agent Guidelines

* Surface violations clearly.
* Never silently break invariants.
* Ask for clarification if unsure whether an invariant applies.

---

# **5. Sequence**

### What it is

An ordered list of transformations forming a meaningful workflow.

### Purpose

Define predictable dataflow through the system.

### Agent Guidelines

* Respect the established order of steps.
* Do not merge or reorder steps without instruction.
* Each step should use the previous step’s output.

---

# **6. Signature**

### What it is

The required structure of inputs and outputs for entities or transformations.

### Purpose

Create stable contracts between modules.

### Agent Guidelines

* Always honor the signature exactly.
* Do not invent fields unless explicitly instructed.
* Do not remove fields from an entity unless asked.

---

# **7. Boundary**

### What it is

The conceptual edge of a module or component.

### Purpose

Prevent responsibilities from bleeding across components.

### Agent Guidelines

* Respect boundaries strictly.
* UI code should not perform analysis.
* Analysis code should not manage database schemas.
* If a job crosses multiple boundaries, break it into steps.

---

# **8. Integration Point**

### What it is

A clear, predictable interface where components connect.

### Purpose

Make the system extensible and modular.

### Agent Guidelines

* Keep integration points explicit.
* Prefer well-defined APIs over implicit coupling.
* Avoid hidden dependency chains.

---

# **9. Lifecycle**

### What it is

The full arc of an operation:
**start → process → finalize → close**.

### Purpose

Ensure every workflow completes cleanly.

### Agent Guidelines

* Always return the system to a coherent state.
* If the lifecycle isn’t completed, surface it as an error.
* Perform cleanup actions when the task ends.

---

# **10. Resolution Layer**

### What it is

A stable abstraction level where complexity is compressed into a single conceptual unit.

### Purpose

Keep the system understandable and maintainable.

### Examples

* Raw text → structured notes
* Notes → embeddings
* Embeddings → clusters

### Agent Guidelines

* Keep each layer conceptually clean.
* Avoid mixing layers.
* Do not collapse layers unless explicitly asked.

---

# **11. Coherence Surface**

### What it is

The final, polished output of a transformation or workflow.

### Purpose

Give downstream components a stable “finished product.”

### Agent Guidelines

* Outputs should be predictable and well-formed.
* Do not leave partial, half-formed, or ambiguous states.
* Prioritize clarity and stability.

---

# **12. Orientation**

### What it is

Understanding where a component sits within the system.

### Purpose

Help agents reason about dependencies and relationships.

### Agent Guidelines

* Always ask for orientation if unclear.
* Keep modules aligned with their intended role.
* Do not repurpose a module unless explicitly directed.

---

# 🚀 How Agents Should Use RSA-Core (Practical Rules)

1. Before coding, identify the **entities** and **transformations**.
2. Stay within **boundaries** unless explicitly told to cross them.
3. Preserve **invariants** in all transformations.
4. Build systems as clean **sequences**, not monoliths.
5. Use **signatures** to guide inputs/outputs.
6. Keep work inside the correct **context space**.
7. Maintain clean **lifecycles** with proper closure.
8. Use **integration points** consciously, not implicitly.
9. Keep everything at the correct **resolution layer**.
10. Produce clean, stable **coherence surfaces**.
11. Maintain correct **orientation** during all coding tasks.

RSA-Core exists to make all agent-driven work:

* modular
* predictable
* transparent
* easy to extend
* easy to debug
* easy to maintain

Without being tied to any domain, project, or technology.

---

