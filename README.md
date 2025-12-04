# Hybrid Reasoning System

A hybrid reasoning system combining symbolic rules, neural networks (GRU), and relational memory with a small transformer (Phi-3) for semantic parsing.

## Features

- **Three reasoning modes**: Rule-based, GRU neural network, and feature-based fallback
- **Relational memory**: RP-structured graph with constraint checking
- **Semantic parsing**: Ollama + Phi-3 for entity/relation extraction
- **Full traceability**: JSON logs of all internal states
- **Transparent architecture**: Clear module boundaries and data flow

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Ollama** with `phi3:latest` model (for semantic parsing)

### Installation

#### Option 1: Quick Install (GPU with CUDA 11.8)
```bash
# Run the automated installer
.\install-gpu.bat

# Or manually:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### Option 2: CPU Only (No GPU)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

#### Option 3: Minimal (Rules + Stub only, no dependencies)
```bash
# No dependencies needed! Uses Python standard library only.
python main.py "Tom has 3 apples..." --scenario apple_sharing
```

**See `INSTALL.md` for detailed installation instructions and troubleshooting.**

### Setup Ollama

```bash
# Install Ollama from https://ollama.ai
# Pull the Phi-3 model
ollama pull phi3

# Start Ollama service (runs in background)
ollama serve
```

## Usage

### CLI Examples

```bash
# Rule-based solving (requires --scenario flag)
python main.py "Tom has 3 apples. He gives Sarah 1, buys 4, then splits them equally with John." --scenario apple_sharing

# Available scenarios: apple_sharing, cookie_problem, travel_distance, recipe_scaling

# Show detailed trace
python main.py "John has 5 cookies and gives 2 to Mary." --scenario cookie_problem --pretty-trace

# Disable rules (use GRU/stub only)
python main.py "..." --scenario apple_sharing --no-rules

# Disable GRU (use rules only)
python main.py "..." --scenario apple_sharing --rules-only
```

### Evaluation Harness

```bash
# Run evaluation on test dataset
python evaluate.py --tasks dataset/test_prompts.json --limit 10

# Filter by scenario
python evaluate.py --tasks dataset/test_prompts.json --scenario apple_sharing

# Enable trace logging
python evaluate.py --tasks dataset/test_prompts.json --limit 5 --log-traces
```

## Project Structure

```
hybrid_reasoner/
├── core/              # Reasoning core (GRU + rules + stub)
├── memory/            # Relational memory with RP-style constraints
├── semantic/          # Semantic engine (Ollama/Phi-3 wrapper)
├── controller/        # Episode orchestration and closure
├── config/            # Configuration (ModelConfig, RuntimeConfig, ReasoningConfig)
└── tests/             # Unit and integration tests

agents/docs/           # Documentation
├── overview.md        # Architecture and design guide
├── changelog.md       # Development history
└── RSA Core.md        # Design principles

dataset/               # Test data
runs/                  # Execution logs
├── evals/             # Evaluation results
└── trace_*.json       # Per-episode traces
```

## Architecture

### Data Flow
```
Input Text
    ↓
Semantic Engine (Ollama + Phi-3)
    ↓ extracts entities/relations
Relational Memory
    ↓ stores + validates
Reasoning Core (Rules/GRU/Stub)
    ↓ iterative reasoning
Controller (Halting + Closure)
    ↓
Semantic Engine (Render answer)
    ↓
Final Answer
```

### Three Reasoning Modes

1. **Rule-Based** (deterministic, requires scenario label)
   - Fast, 100% accurate on covered scenarios
   - Pure math on extracted numbers
   - No learning required

2. **GRU Neural Network** (trainable, generalizable)
   - Learn patterns from examples
   - No scenario labels needed
   - Currently untrained (random weights)

3. **Feature-Based Stub** (fallback)
   - Count-based features
   - Simple halting heuristic
   - Always available

## Configuration

### Model Config (`hybrid_reasoner/config/settings.py`)
- `model_name`: Ollama model tag (default: `phi3:latest`)
- `temperature`: LLM temperature (default: `0.1`)
- `max_tokens`: Max generation tokens (default: `512`)
- `request_timeout`: HTTP timeout (default: `90` seconds)

### Reasoning Config (`hybrid_reasoner/config/reasoning.py`)
- `use_rules`: Enable rule-based shortcuts (default: `True`)
- `use_gru`: Enable GRU neural network (default: `True`)
- `hidden_size`: GRU hidden dimension (default: `64`)
- `max_steps`: Maximum reasoning steps (default: `10`)
- `halt_threshold`: Halt probability threshold (default: `0.85`)

## Training the GRU (TODO)

The GRU architecture is implemented but untrained. To train:

1. Prepare training data with reasoning traces
2. Define loss function (halt prediction + relation prediction)
3. Implement training loop with backpropagation
4. Evaluate on held-out test set

Training infrastructure is planned but not yet implemented.

## Current Status

- ✅ **Phase 1-6 Complete**: Full MVP implementation
- ✅ **Ollama Integration**: Working semantic parsing
- ✅ **Rule-Based Solving**: 100% accuracy on 4 scenarios
- ✅ **GRU Architecture**: Implemented, ready for training
- ❌ **GRU Training**: Not yet implemented
- ❌ **Generalization**: Rules only work on known scenarios

## Dependencies

**Runtime (minimal)**:
- Python 3.10+ standard library only
- Ollama service (external, for semantic parsing)

**Training (full)**:
- PyTorch 2.0+
- NumPy, Pandas, scikit-learn
- Matplotlib, Seaborn (visualization)
- TensorBoard (monitoring)
- See `requirements.txt` for complete list

## License

(Add your license here)

## References

- Architecture based on `agents/docs/overview.md`
- Design principles: `agents/docs/RSA Core.md`
- Development log: `agents/docs/changelog.md`

