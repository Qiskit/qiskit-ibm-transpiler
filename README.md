# qiskit_ibm_transpiler

A library to use [Qiskit IBM Transpiler](https://docs.quantum.ibm.com/transpile/qiskit-ibm-transpiler) and the [AI transpiler passes](https://docs.quantum.ibm.com/transpile/ai-transpiler-passes).

**Note** The Qiskit IBM Transpiler and the AI transpiler passes use different experimental services that are only available for IBM Quantum Premium Plan users. This library and the releated services are an alpha release, subject to change.

## Installing the qiskit-ibm-transpiler

To use the Qiskit IBM Transpiler, install the `qiskit-ibm-transpiler` package:

```sh
pip install qiskit-ibm-transpiler
```

By default, the package tries to authenticate to IBM Quantum services with the defined Qiskit API token, and uses your token from the `QISKIT_IBM_TOKEN` environment variable or from the file `~/.qiskit/qiskit-ibm.json` (under the section `default-ibm-quantum`).

_Note_: This library requires Qiskit 1.0 by default.

## How to use the library

### Using the Qiskit IBM Transpiler

The following examples demonstrate how to transpile circuits using the Qiskit IBM Transpiler with different parameters.

1. Create a circuit and call the Qiskit IBM Transpiler to transpile the circuit with `ibm_sherbrooke` as the `backend_name`, 3 as the `optimization_level`, and not using AI during the transpilation.

   ```python
   from qiskit.circuit.library import EfficientSU2
   from qiskit_ibm_transpiler.transpiler_service import TranspilerService

   circuit = EfficientSU2(101, entanglement="circular", reps=1).decompose()

   cloud_transpiler_service = TranspilerService(
       backend_name="ibm_sherbrooke",
       ai='false',
       optimization_level=3,
   )
   transpiled_circuit = cloud_transpiler_service.run(circuit)
   ```

_Note:_ you only can use `backend_name` devices you are allowed to with your IBM Quantum Account. Apart from the `backend_name`, the `TranspilerService` also allows `coupling_map` as parameter.

2. Produce a similar circuit and transpile it, requesting AI transpiling capabilities by setting the flag `ai` to `'true'`:

   ```python
   from qiskit.circuit.library import EfficientSU2
   from qiskit_ibm_transpiler.transpiler_service import TranspilerService

   circuit = EfficientSU2(101, entanglement="circular", reps=1).decompose()

   cloud_transpiler_service = TranspilerService(
       backend_name="ibm_sherbrooke",
       ai='true',
       optimization_level=1,
   )
   transpiled_circuit = cloud_transpiler_service.run(circuit)
   ```

### Using the AIRouting pass manually

The `AIRouting` pass acts both as a layout stage and a routing stage. It can be used within a `PassManager` as follows:

```python
from qiskit.transpiler import PassManager
from qiskit_ibm_transpiler.ai.routing import AIRouting
from qiskit.circuit.library import EfficientSU2

ai_passmanager = PassManager([
   AIRouting(backend_name="ibm_sherbrooke", optimization_level=2, layout_mode="optimize")
])

circuit = EfficientSU2(101, entanglement="circular", reps=1).decompose()

transpiled_circuit = ai_passmanager.run(circuit)
```

Here, the `backend_name` determines which backend to route for, the `optimization_level` (1, 2, or 3) determines the computational effort to spend in the process (higher usually gives better results but takes longer), and the `layout_mode` specifies how to handle the layout selection.
The `layout_mode` includes the following options:

- `keep`: This respects the layout set by the previous transpiler passes (or uses the trivial layout if not set). It is typically only used when the circuit must be run on specific qubits of the device. It often produces worse results because it has less room for optimization.
- `improve`: This uses the layout set by the previous transpiler passes as a starting point. It is useful when you have a good initial guess for the layout; for example, for circuits that are built in a way that approximately follows the device's coupling map. It is also useful if you want to try other specific layout passes combined with the `AIRouting` pass.
- `optimize`: This is the default mode. It works best for general circuits where you might not have good layout guesses. This mode ignores previous layout selections.

### Using the AI circuit synthesis passes

The AI circuit synthesis passes allow you to optimize pieces of different circuit types ([Clifford](https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.Clifford), [Linear Function](https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.LinearFunction), [Permutation](https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.Permutation#permutation)) by re-synthesizing them. The typical way one would use the synthesis pass is the following:

```python
from qiskit.transpiler import PassManager

from qiskit_ibm_transpiler.ai.routing import AIRouting
from qiskit_ibm_transpiler.ai.synthesis import AILinearFunctionSynthesis
from qiskit_ibm_transpiler.ai.collection import CollectLinearFunctions
from qiskit.circuit.library import EfficientSU2

ai_passmanager = PassManager([
   AIRouting(backend_name="ibm_cairo", optimization_level=3, layout_mode="optimize"),  # Route circuit
   CollectLinearFunctions(),  # Collect Linear Function blocks
   AILinearFunctionSynthesis(backend_name="ibm_cairo")  # Re-synthesize Linear Function blocks
])

circuit = EfficientSU2(10, entanglement="full", reps=1).decompose()

transpiled_circuit = ai_passmanager.run(circuit)
```

The synthesis respects the coupling map of the device: it can be run safely after other routing passes without "messing up" the circuit, so the overall circuit will still follow the device restrictions. By default, the synthesis will replace the original sub-circuit only if the synthesized sub-circuit improves the original (currently only checking `CNOT` count), but this can be forced to always replace the circuit by setting `replace_only_if_better=False`.

The following synthesis passes are available from `qiskit_ibm_transpiler.ai.synthesis`:

- _AICliffordSynthesis_: Synthesis for [Clifford](https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.Clifford) circuits (blocks of `H`, `S` and `CX` gates). Currently up to 9 qubit blocks.
- _AILinearFunctionSynthesis_: Synthesis for [Linear Function](https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.LinearFunction) circuits (blocks of `CX` and `SWAP` gates). Currently up to 9 qubit blocks.
- _AIPermutationSynthesis_: Synthesis for [Permutation](https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.Permutation#permutation) circuits (blocks of `SWAP` gates). Currently available for 65, 33, and 27 qubit blocks.

We expect to gradually increase the size of the supported blocks.

All passes use a thread pool to send several requests in parallel. By default it will use as max threads as number of cores plus four (default values for `ThreadPoolExecutor` python object). However, you can set your own value with the `max_threads` argument at pass instantation. For example, the following line will instantiate the `AILinearFunctionSynthesis` pass allowing it to use a maximum of 20 threads.

```python
AILinearFunctionSynthesis(backend_name="ibm_cairo", max_threads=20)  # Re-synthesize Linear Function blocks using 20 threads max
```

You can also set the environment variable `AI_TRANSPILER_MAX_THREADS` to the desired number of maximum threads, and all synthesis passes instantiated after that will use that value.

For sub-circuit to be synthesized by the AI synthesis passes, it must lay on a connected subgraph of the coupling map (this can be ensured by just doing a routing pass previous to collecting the blocks, but this is not the only way to do it). The synthesis passes will automatically check if a the specific subgraph where the sub-circuit lays is supported, and if it is not supported it will raise a warning and just leave the original sub-circuit as it is.

To complement the synthesis passes we also provide custom collection passes for Cliffords, Linear Functions and Permutations that can be imported from `qiskit_ibm_transpiler.ai.collection`:

- _CollectCliffords_: Collects `Clifford` blocks as `Instruction` objects and stores the original sub-circuit to compare against it after synthesis.
- _CollectLinearFunctions_: Collects blocks of `SWAP` and `CX` as `LinearFunction` objects and stores the original sub-circuit to compare against it after synthesis.
- _CollectPermutations_: Collects blocks of `SWAP` circuits as `Permutations`.

These custom collection passes limit the sizes of the collected sub-circuits so that they are supported by the AI synthesis passes, so it is recommended to use them after the routing passes and before the synthesis passes to get a better optimization overall.

### Customize logs

The library is prepared to let the user log the messages they want. For that, users only have to add the following code to their code:

```python
import logging

logging.getLogger("qiskit_ibm_transpiler").setLevel(logging.X)
```

where X can be: `NOTSET`, `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`

## Citation

If you use any AI-powered feature from the Qiskit IBM Transpiler in your research, use the following recommended citation:

```
@misc{2405.13196,
Author = {David Kremer and Victor Villar and Hanhee Paik and Ivan Duran and Ismael Faro and Juan Cruz-Benito},
Title = {Practical and efficient quantum circuit synthesis and transpiling with Reinforcement Learning},
Year = {2024},
Eprint = {arXiv:2405.13196},
}
```
