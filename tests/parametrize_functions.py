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

import pytest


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
