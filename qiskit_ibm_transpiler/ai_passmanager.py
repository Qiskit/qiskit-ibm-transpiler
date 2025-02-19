# -*- coding: utf-8 -*-

# (C) Copyright 2022 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import copy
from typing import Union

from qiskit.providers.backend import BackendV2 as Backend
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from .ai.collection import CollectLinearFunctions
from .ai.routing import AIRouting
from .ai.synthesis import AILinearFunctionSynthesis


def generate_ai_pass_manager(
    optimization_level: int,
    ai_optimization_level: Union[int, dict],
    coupling_map: Union[CouplingMap, None] = None,
    backend: Union[Backend, None] = None,
    ai_layout_mode="optimize",
    include_ai_synthesis: bool = True,
    optimization_preferences: list[str] = [
        "cnot_layers",
        "n_cnots",
        "layers",
        "n_gates",
    ],
    qiskit_transpile_options: dict = None,
):
    if qiskit_transpile_options is None:
        qiskit_transpile_options = {}
    # If optimization_level is part of the qiskit_transpile_options,
    # remove it in favor of the request input params optimization_level
    if qiskit_transpile_options.get("optimization_level", None) is not None:
        del qiskit_transpile_options["optimization_level"]

    if coupling_map is None and backend is None:
        raise TypeError("Either coupling_map or backend must be provided.")

    # Create the base Qiskit pass manager
    initial_layout_is_not_none = (
        "initial_layout" in qiskit_transpile_options
    )  # This means user provided an initial layout
    qiskit_transpile_options_copy = copy.deepcopy(qiskit_transpile_options)
    pass_manager = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        coupling_map=coupling_map,
        **qiskit_transpile_options_copy,
    )

    # Make cmap symmetric for AI Routing
    coupling_map_ia = copy.deepcopy(
        backend.coupling_map if coupling_map is None else coupling_map
    )
    coupling_map_ia.make_symmetric()

    # If user provides a layout but also chooses "optimize", we convert it to "improve" to leverage the user layout
    if initial_layout_is_not_none and ai_layout_mode == "optimize":
        ai_layout_mode = "improve"

    # Adding the block collection to improve routing of decomposed unitaries
    ai_routing = AIRouting(
        coupling_map=coupling_map_ia,
        optimization_level=ai_optimization_level,
        optimization_preferences=optimization_preferences,
        layout_mode=ai_layout_mode,
        local_mode=True,
    )

    if ai_layout_mode == "keep" or initial_layout_is_not_none:
        # We use trivial layout from qiskit passmanager lvl0 to set the provided layout
        pass_manager_lvl0 = generate_preset_pass_manager(
            optimization_level=0,
            backend=backend,
            coupling_map=coupling_map_ia,
            **copy.deepcopy(qiskit_transpile_options),
        )

        # Then we replace the layout and routing passes with lvl0 ones
        pass_manager.layout = pass_manager_lvl0.layout
        # pass_manager.routing = pass_manager_lvl0.routing

        # Finally, we replace the routing pass with the AI pass
        pass_manager.routing._tasks[1][0].tasks = (
            pass_manager.routing._tasks[1][0].tasks[0],
            ai_routing,
        )

    else:
        # First we replace SabreLayout with AIRouting
        # See https://github.com/Qiskit/qiskit/blob/f97a620e9ff388d273df3fbbef604c2f656d4bfb/qiskit/transpiler/preset_passmanagers/builtin_plugins.py#L680
        layout_position = 3 if optimization_level == 1 else 2
        pass_manager.layout._tasks[layout_position][0].tasks = (ai_routing,)

        # Then we remove the routing stage
        pass_manager.routing = None

    if include_ai_synthesis and optimization_level > 1:
        synth_pm = None
        if coupling_map is None:
            coupling_map = backend.coupling_map
        synth_lf = AILinearFunctionSynthesis(coupling_map=coupling_map, local_mode=True)

        if optimization_level == 2:
            collect_front = CollectLinearFunctions(do_commutative_analysis=False)
            collect_back = CollectLinearFunctions(
                do_commutative_analysis=False, collect_from_back=True
            )

            synth_pm = PassManager([collect_front, synth_lf, collect_back, synth_lf])
        elif optimization_level == 3:
            collect_front = CollectLinearFunctions(do_commutative_analysis=True)
            collect_back = CollectLinearFunctions(
                do_commutative_analysis=True, collect_from_back=True
            )

            synth_pm = PassManager(
                [collect_front, synth_lf, collect_back, synth_lf] * 2
            )

        pass_manager.post_routing = synth_pm

    return pass_manager
