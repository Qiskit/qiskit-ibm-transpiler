# -*- coding: utf-8 -*-

# (C) Copyright 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Parametrize functions"""

import pytest

from qiskit_ibm_transpiler.ai.collection import (
    CollectPermutations,
    CollectLinearFunctions,
    CollectCliffords,
    CollectPauliNetworks,
)

from qiskit_ibm_transpiler.ai.synthesis import (
    AIPermutationSynthesis,
    AILinearFunctionSynthesis,
    AICliffordSynthesis,
    AIPauliNetworkSynthesis,
)


def parametrize_ai():
    return pytest.mark.parametrize("ai", ["false", "true"], ids=["no_ai", "ai"])


def parametrize_valid_optimization_level():
    return pytest.mark.parametrize(
        "optimization_level",
        [1, 2, 3],
        ids=["opt_level_1", "opt_level_2", "opt_level_3"],
    )


def parametrize_qiskit_transpile_options():
    return pytest.mark.parametrize(
        "qiskit_transpile_options",
        [None, {"seed_transpiler": 0}],
        ids=["no opt", "one option"],
    )


def parametrize_non_valid_use_fractional_gates():
    return pytest.mark.parametrize(
        "non_valid_use_fractional_gates",
        [8, "8", "foo"],
    )


def parametrize_valid_use_fractional_gates():
    return pytest.mark.parametrize(
        "valid_use_fractional_gates",
        ["no", "n", "false", "f", "0", "yes", "y", "true", "t", "1"],
    )


def parametrize_non_valid_optimization_preferences():
    return pytest.mark.parametrize(
        "non_valid_optimization_preferences",
        ["foo"],
    )


def parametrize_valid_optimization_preferences():
    return pytest.mark.parametrize(
        "valid_optimization_preferences",
        [None, "noise", ["noise", "n_cnots"]],
    )


def parametrize_valid_optimization_preferences_without_noise():
    return pytest.mark.parametrize(
        "valid_optimization_preferences_without_noise",
        [None, "n_cnots", ["n_cnots"]],
    )


def parametrize_n_qubits():
    return pytest.mark.parametrize("n_qubits", [3, 10, 30])


def parametrize_local_mode():
    return pytest.mark.parametrize(
        "local_mode",
        [True, False],
        ids=["local_mode", "cloud_mode"],
    )


def parametrize_coupling_map_format():
    return pytest.mark.parametrize(
        "coupling_map",
        ["brisbane_coupling_map", "brisbane_coupling_map_list_format"],
        ids=["coupling_map_object", "coupling_map_list"],
    )


def parametrize_circuit_collector_pass_and_operator_name():
    return pytest.mark.parametrize(
        "circuit, collector_pass, operator_name",
        [
            (
                "random_circuit_with_several_permutations_transpiled",
                CollectPermutations,
                "permutation",
            ),
            (
                "random_circuit_with_several_linear_functions_transpiled",
                CollectLinearFunctions,
                "linear_function",
            ),
            (
                "random_circuit_with_several_cliffords_transpiled",
                CollectCliffords,
                "Clifford",
            ),
            (
                "random_circuit_with_several_paulis_transpiled",
                CollectPauliNetworks,
                "pauli",
            ),
        ],
        ids=["permutation", "linear_function", "clifford", "pauli_network"],
    )


def parametrize_collectable_gates_collector_pass_operation_name():
    return pytest.mark.parametrize(
        "collectable_gates, collector_pass, operation_name",
        [
            ("swap", CollectPermutations, "permutation"),
            ("cx", CollectLinearFunctions, "linear_function"),
            ("cz", CollectCliffords, "clifford"),
            ("swap", CollectPauliNetworks, "paulinetwork"),
        ],
        ids=["permutation", "linear_function", "clifford", "pauli_network"],
    )


def parametrize_collectable_gates_and_collector_pass():
    return pytest.mark.parametrize(
        "collectable_gates, collector_pass",
        [
            ("swap", CollectPermutations),
            ("cx", CollectLinearFunctions),
            ("cz", CollectCliffords),
            ("swap", CollectPauliNetworks),
        ],
        ids=["permutation", "linear_function", "clifford", "pauli_network"],
    )


def parametrize_non_collectable_gates_collector_pass_operation_name():
    return pytest.mark.parametrize(
        "non_collectable_gates, collector_pass, operation_name",
        [
            ("rzz", CollectPermutations, "permutation"),
            ("rzz", CollectLinearFunctions, "linear_function"),
            ("rzz", CollectCliffords, "clifford"),
            ("t", CollectPauliNetworks, "paulinetwork"),
        ],
        ids=["permutation", "linear_function", "clifford", "pauli_network"],
    )


# TODO: For Permutations, the original circuit doesn't return a DAGCircuit with nodes. Decide how the code should behave on this case
def parametrize_basic_circuit_collector_pass_and_ai_synthesis_pass():
    return pytest.mark.parametrize(
        "circuit, collector_pass, ai_synthesis_pass",
        [
            (
                "basic_swap_circuit",
                CollectPermutations,
                AIPermutationSynthesis,
            ),
            ("basic_cnot_circuit", CollectLinearFunctions, AILinearFunctionSynthesis),
            ("basic_cnot_circuit", CollectCliffords, AICliffordSynthesis),
            (
                "basic_cnot_circuit",
                CollectPauliNetworks,
                AIPauliNetworkSynthesis,
            ),
        ],
        ids=["permutation", "linear_function", "clifford", "pauli_network"],
    )


def parametrize_complex_circuit_collector_pass_and_ai_synthesis_pass():
    return pytest.mark.parametrize(
        "circuit, collector_pass, ai_synthesis_pass",
        [
            (
                "permutation_circuit_brisbane",
                CollectPermutations,
                AIPermutationSynthesis,
            ),
            (
                "linear_function_circuit",
                CollectLinearFunctions,
                AILinearFunctionSynthesis,
            ),
            ("clifford_circuit", CollectCliffords, AICliffordSynthesis),
            (
                "random_pauli_circuit_transpiled",
                CollectPauliNetworks,
                AIPauliNetworkSynthesis,
            ),
        ],
        ids=["permutation", "linear_function", "clifford", "pauli_network"],
    )
