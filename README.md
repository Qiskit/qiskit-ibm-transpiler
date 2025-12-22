# qiskit_ibm_transpiler

[![PyPI version](https://badge.fury.io/py/qiskit-ibm-transpiler.svg)](https://badge.fury.io/py/qiskit-ibm-transpiler)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Leverage IBM Quantum's cutting-edge [Qiskit Transpiler Service](https://quantum.cloud.ibm.com/docs/guides/qiskit-transpiler-service) and [AI-powered transpiler passes](https://quantum.cloud.ibm.com/docs/guides/ai-transpiler-passes) to achieve superior circuit optimization through reinforcement learning algorithms.

## ✨ Key Features

- 🧠 **AI-Powered Optimization**: Advanced routing and circuit synthesis using reinforcement learning algorithms 
- ⚡ **Local & Remote Modes**: Run AI passes locally or leverage cloud resources
- ☁️ **Cloud-ready**: Harness IBM Quantum's cloud infrastructure for intensive computations 
- 🎯 **Drop-in Replacement**: Seamlessly integrate with existing Qiskit workflows
- 📈 **Superior Performance**: Our AI models typically outperform traditional heuristic algorithms. [Read the benchmark](https://arxiv.org/abs/2409.08844)

**Note**: The cloud transpilation capabilities are only available for IBM Quantum Premium Plan users. The local mode is available to any user and is enabled by default if the local mode dependencies are installed. Currently in beta release.

## 📦 Installation

Install the package with pip:
```bash
pip install qiskit-ibm-transpiler
```

**Note**: AI local mode dependencies (`qiskit-ibm-ai-local-transpiler`) are now included by default. The `[ai-local-mode]` extra is maintained for backward compatibility but is no longer required:
```bash
# This still works but is now equivalent to the basic installation
pip install qiskit-ibm-transpiler[ai-local-mode]
```

### 🔐 Authentication

The package automatically authenticates using your [IBM Quantum Platform credentials](https://quantum.cloud.ibm.com/docs/en/guides/cloud-setup) aligned with how [Qiskit Runtime manages it](https://github.com/Qiskit/qiskit-ibm-runtime/tree/0.40.1?tab=readme-ov-file#qiskit-runtime-service-on-ibm-cloud):
- Environment variable: `QISKIT_IBM_TOKEN`
- Configuration file: `~/.qiskit/qiskit-ibm.json` (searches in order: `default-ibm-quantum-platform`, `default-ibm-quantum`)

You can also specify a particular saved account by name using the `account_name` parameter:

```python
from qiskit_ibm_transpiler.transpiler_service import TranspilerService

# Use a specific saved account
service = TranspilerService(
    backend_name="ibm_torino",
    account_name="my-custom-account"  # Uses this account, falls back to defaults if not found
)
```

## 🚀 Getting Started

### Tutorial and Examples

For a comprehensive introduction to the qiskit-ibm-transpiler library, start here:

- **📖 [AI Transpiling Tutorial](ai-transpiling-tutorial.ipynb)** - Complete walkthrough of the library's features and capabilities
- **📁 [Examples Directory](examples/)** - Collection of Jupyter notebooks demonstrating specific use cases:
  - [AI Transpiler Demo](examples/ai-transpiler-demo.ipynb) - Basic transpilation examples
  - [AI Clifford Synthesis Demo](examples/ai-clifford-synthesis-demo.ipynb) - Clifford circuit optimization
  - [AI Linear Function Synthesis Demo](examples/ai-linear-function-synthesis-demo.ipynb) - Linear function synthesis
  - [AI Permutation Synthesis Demo](examples/ai-permutation-synthesis-demo.ipynb) - Permutation circuit synthesis
  - [AI Large Circuit Speed Test](examples/ai-large-circuit-speed-test.ipynb) - Performance benchmarking

These notebooks provide hands-on examples and detailed explanations to help you get the most out of the AI-powered transpilation capabilities.

### Quick Start

#### Using AI-powered Transpiler Passes Locally (Recommended)

**AI Routing Pass**

The `AIRouting` pass provides intelligent layout selection and circuit routing using reinforcement learning:

```python
from qiskit.transpiler import PassManager
from qiskit_ibm_transpiler.ai.routing import AIRouting
from qiskit.circuit.library import EfficientSU2

# Local mode execution
ai_routing = PassManager([
    AIRouting(
        backend_name="ibm_torino", 
        optimization_level=3, 
        layout_mode="optimize",
        local_mode=True  # Run locally for faster execution
    )
])

circuit = EfficientSU2(101, entanglement="circular", reps=1).decompose()
routed_circuit = ai_routing.run(circuit)
```

##### Configuration Options

| Parameter | Options | Description |
|-----------|---------|-------------|
| `optimization_level` | 1, 2, 3 | Computational effort (higher = better results, longer time) |
| `layout_mode` | `optimize` | Best for general circuits (default) |
| | `improve` | Uses existing layout as starting point |
| | `keep` | Respects previous layout selection |
| `local_mode` | `True/False` | Run locally or on cloud |

**AI Circuit Synthesis Passes**

Optimize specific circuit blocks using AI-powered synthesis for superior gate count reduction:

```python
from qiskit.transpiler import PassManager
from qiskit_ibm_transpiler.ai.routing import AIRouting
from qiskit_ibm_transpiler.ai.synthesis import (
    AILinearFunctionSynthesis, AIPauliNetworkSynthesis
)
from qiskit_ibm_transpiler.ai.collection import (
    CollectLinearFunctions, CollectPauliNetworks
)
from qiskit.circuit.library import EfficientSU2

# Complete AI-powered transpilation pipeline
ai_pm = PassManager([
    AIRouting(backend_name="ibm_torino", optimization_level=3, layout_mode="optimize"),
    
    # Collect and synthesize linear functions
    CollectLinearFunctions(),
    AILinearFunctionSynthesis(backend_name="ibm_torino", local_mode=True),
    
    # Collect and synthesize Pauli networks
    CollectPauliNetworks(),
    AIPauliNetworkSynthesis(backend_name="ibm_torino", local_mode=True),
])

circuit = EfficientSU2(10, entanglement="full", reps=1).decompose()
optimized_circuit = ai_pm.run(circuit)
```

**Available Synthesis Passes**

| Pass | Circuit Type | Max Qubits | Local Mode |
|------|-------------|------------|------------|
| `AICliffordSynthesis` | H, S, CX gates | 9 | ✅ |
| `AILinearFunctionSynthesis` | CX, SWAP gates | 9 | ✅ |
| `AIPermutationSynthesis` | SWAP gates | 65, 33, 27 | ✅ |
| `AIPauliNetworkSynthesis` | H, S, SX, CX, RX, RY, RZ | 6 | ✅ |


#### Using the Transpiler Service (Cloud)

> **Note**: The Qiskit Transpiler Service is currently being migrated. We recommend using local mode instead.

```python
from qiskit.circuit.library import EfficientSU2
from qiskit_ibm_transpiler.transpiler_service import TranspilerService

# Create your circuit
circuit = EfficientSU2(101, entanglement="circular", reps=1).decompose()

# Enable AI optimization for superior results
service = TranspilerService(
    backend_name="ibm_torino",
    ai="auto",              # Service decides: AI passes vs standard Qiskit
    optimization_level=3,
)
optimized_circuit = service.run(circuit)
```

**Service Configuration Options:**

| Parameter | Values | Description |
|-----------|--------|-------------|
| `ai` | `"true"`, `"false"`, `"auto"` | AI transpilation mode |
| `optimization_level` | `1`, `2`, `3` | Optimization intensity |
| `backend_name` | Backend string | Target quantum device |
| `coupling_map` | List of tuples | Custom connectivity |

**Service Limits**: Max 1M two-qubit gates per job, 30-minute transpilation timeout, 20-minute result retrieval window.

#### Hybrid Heuristic-AI Circuit Transpilation

The qiskit-ibm-transpiler allows you to configure a hybrid pass manager that automatically combines the best of Qiskit's heuristic and AI-powered transpiler passes. This feature behaves similarly to the Qiskit `generate_pass_manager` method:

```python
from qiskit_ibm_transpiler import generate_ai_pass_manager
from qiskit.circuit.library import efficient_su2
from qiskit_ibm_runtime import QiskitRuntimeService

backend = QiskitRuntimeService().backend("ibm_torino")
torino_coupling_map = backend.coupling_map

su2_circuit = efficient_su2(101, entanglement="circular", reps=1)

ai_hybrid_pass_manager = generate_ai_pass_manager(
    coupling_map=torino_coupling_map,
    ai_optimization_level=3,
    optimization_level=3,
    ai_layout_mode="optimize",
)

ai_su2_transpiled_circuit = ai_hybrid_pass_manager.run(su2_circuit)
```

**Configuration Options:**
- `coupling_map`: Specifies which coupling map to use for the transpilation
- `ai_optimization_level`: Level of optimization (1-3) for AI components of the PassManager
- `optimization_level`: Optimization level for heuristic components of the PassManager
- `ai_layout_mode`: How the AI routing handles layout (see AI routing pass section for options)


### Performance Tuning

**Thread Pool Configuration**:
```python
# Method 1: Per-pass configuration
AILinearFunctionSynthesis(backend_name="ibm_torino", max_threads=20)

# Method 2: Global environment variable
import os
os.environ["AI_TRANSPILER_MAX_THREADS"] = "20"
```

**Smart Replacement**:
- Default: Only replaces if synthesis improves gate count
- Force replacement: `replace_only_if_better=False`

> **Note**: Synthesis passes respect device coupling maps and work seamlessly after routing passes.

## 🔧 Advanced Configuration

### Logging

Customize logging levels for debugging and monitoring:

```python
import logging

# Available levels: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
logging.getLogger("qiskit_ibm_transpiler").setLevel(logging.INFO)
```

## 📚 Resources & Support

- 📖 [Official Documentation](https://quantum.cloud.ibm.com/docs/guides/qiskit-transpiler-service)
- 🧠 [AI Transpiler Passes Guide](https://quantum.cloud.ibm.com/docs/guides/ai-transpiler-passes)
- 🎯 [IBM Quantum Platform](https://quantum.cloud.ibm.com/)
- 💡 [Give us feedback](https://qiskit.slack.com/archives/C06KF8YHUAU)

## 📄 Citation

If you use this library in your research, please cite:

```bibtex
@misc{kremer2024practical,
    title={Practical and efficient quantum circuit synthesis and transpiling with Reinforcement Learning},
    author={David Kremer and Victor Villar and Hanhee Paik and Ivan Duran and Ismael Faro and Juan Cruz-Benito},
    year={2024},
    eprint={2405.13196},
    archivePrefix={arXiv},
    primaryClass={quant-ph}
}
```
