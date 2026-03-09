# -*- coding: utf-8 -*-

# (C) Copyright 2022 IBM. All Rights Reserved.


"""Transpile utils"""


def check_transpiling(circ, cmap):
    """Checks if a given circuit follows a specific coupling map"""
    for cc in circ:
        if cc.operation.num_qubits == 2:
            q_pair = tuple(circ.find_bit(qi).index for qi in cc.qubits)
            if (
                q_pair not in cmap
                and q_pair[::-1] not in cmap
                and list(q_pair) not in cmap
                and list(q_pair[::-1]) not in cmap
            ):
                return False
    return True
