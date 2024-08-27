# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

> **Types of changes**:
>
> - 🎉 **Added**: for new features.
> - ✏️ **Changed**: for changes in existing functionality.
> - 🐛 **Fixed**: for any bug fixes.
> - ❌ **Removed**: for now removed features.
> - 👾 **Security**: in case of vulnerabilities.
> - ⚠️ **Deprecated**: for soon-to-be removed features.

## [Unreleased]

### 🎉 Added

### ✏️ Changed

### 🐛 Fixed

### ❌ Removed

### 👾 Security

### ⚠️ Deprecated

## [0.4.7] - 2024-08-27

### 🐛 Fixed

- [#49](https://github.com/Qiskit/qiskit-transpiler-service/pull/49) Add stacklevel to DeprecationWarning so it appears in Jupyter notebooks

## [0.4.6] - 2024-08-23

### 🎉 Added

- [#40](https://github.com/Qiskit/qiskit-transpiler-service/pull/40) Added deprecation notice in preparation for project rename to `qiskit-ibm-transpiler`

### 🐛 Fixed

- [#36](https://github.com/Qiskit/qiskit-transpiler-service/pull/36) Forward incoming error when decoding fails

## [0.4.5] - 2024-08-01

### 🎉 Added

- [#28](https://github.com/Qiskit/qiskit-transpiler-service/pull/28) Enable programatic config of polling timeout

### 🐛 Fixed

- [#25](https://github.com/Qiskit/qiskit-transpiler-service/pull/25) Using the declared ai_layout_mode in the TranspilerService's run method
- [#26](https://github.com/Qiskit/qiskit-transpiler-service/pull/26) Updating tests about service's limits after last changes

## [0.4.4] - 2024-07-24

### ✏️ Changed

- [#21](https://github.com/Qiskit/qiskit-transpiler-service/pull/21) Increasing timeout to 600s instead of 120s
- [#17](https://github.com/Qiskit/qiskit-transpiler-service/pull/17) Fix tests after update on the service

### 🐛 Fixed

- [#10](https://github.com/Qiskit/qiskit-transpiler-service/pull/10) Configures Logging for a Library
- [#8](https://github.com/Qiskit/qiskit-transpiler-service/pull/8) Expose service errors to users
- [#7](https://github.com/Qiskit/qiskit-transpiler-service/pull/7) Correctly parse string boolean for ai param

## [0.4.3] - 2024-06-07

### ✏️ Changed

- #85 Check existence of credentials and raise related exceptions instead of breaking

## [0.4.2] - 2024-06-07

### ✏️ Changed

- #82 Relaxing pin of local dependencies

### 🐛 Fixed

- #83 Python 3.8 support. Back to specify types with typing
- #81) Fixing some import errors for local modules

## [0.4.1] - 2024-06-06

### ✏️ Changed

- #72 Refactoring the extension code
- #69 Updating logs levels

### 🐛 Fixed

- #76 Rebuild layout in transpiled circuit
- #71 Avoid barrier nodes in collection

## [0.4.0] - 2024-05-28

### 🎉 Added

- #63 Request the synthesis of a list of circuits to the service.
- #57 Adding docstrings for public documentation

### ✏️ Changed

- #67 Updating Clifford service URL.
- #60 Replace use_ai by ai param in requests
- #58 Move type hints in documentation.
- #54 Set name to logger

### 🐛 Fixed

- #56 Adjust dependencies to be less strict and support minor version updates for qiskit & patch updates for qiskit-qasm3-import

## [0.3.0] - 2024-02-29

### ✏️ Changed

- #36 Updating the plugin to use Qiskit 1.0 by default.

### 🐛 Fixed

- #38 Supporting measurements in circuits when transpiling using ai=true option.

## [0.2.1] - 2024-02-22

### 🎉 Added

- #34 Add multithreading to synth requests. The requests  to the service for transpile and transpile now are done in parallel.

### ✏️ Changed

- #31 Updated collection passes. Now the passes could work up to N of qubits or with any block size of qubits

## [0.2.0] - 2024-02-12

### 🎉 Added

- #28 Added support for synthesis and transpilation of Clifford, Permutation and Linear Function circuits. Using new URLs for the service.

## [0.1.3] - 2023-12-11

### 🐛 Fixed

- #20 Fixing layout integration with Qiskit for the transpiler service
- #18 Fixing hardcoded input to routing
- #17 Fix bug in input and refactor

## [0.1.2] - 2023-12-04

- Publishing first version 0.1.2 for the IBM Quantum Summit.

[Unreleased]: https://github.com/Qiskit/qiskit-transpiler-service/compare/0.4.7...main-qiskit-transpiler-service
[0.4.7]: https://github.com/Qiskit/qiskit-transpiler-service/compare/0.4.6...0.4.7
[0.4.6]: https://github.com/Qiskit/qiskit-transpiler-service/compare/0.4.5...0.4.6
[0.4.5]: https://github.com/Qiskit/qiskit-transpiler-service/compare/0.4.4...0.4.5
[0.4.4]: https://github.com/Qiskit/qiskit-transpiler-service/compare/0.4.3...0.4.4
