Fixed an import error when ``qiskit_ibm_transpiler`` is imported with Qiskit 2.5. The test-data helpers in
:mod:`qiskit_ibm_transpiler.utils` imported ``random_invertible_binary_matrix`` from the private module
``qiskit.synthesis.linear.linear_matrix_utils``, which Qiskit 2.5 no longer ships; the helper is now imported from
the public ``qiskit.synthesis.linear`` package, which exports it in every supported Qiskit release (1.4.2 through 2.5).
No behavior of the AI passes changes.
