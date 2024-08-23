# Contributing to qiskit_ibm_transpiler

## Documentation

**Write**

We use Sphinx to compile docstrings into API documentation. The docstrings are written in reStructuredText. See our [Sphinx guide](https://qiskit.github.io/qiskit_sphinx_theme/sphinx_guide/index.html) for more information on writing Sphinx documentation.

If you want an object to appear in the API documentation, you'll need to add it to the autosummary of the appropriate module-level docstring. For example, the [`qiskit_ibm_transpiler.ai`](qiskit_ibm_transpiler/ai/__init__.py) module contains the following autosummary.

```rst
Classes
=======
.. autosummary::
   :toctree: ../stubs/

   AIRouting
```

Functions should be inlined in the module's file, e.g. `utils.py`:

```rst
.. currentmodule:: qiskit_ibm_transpiler.utils

Functions
=========

.. autofunction:: create_random_linear_function
.. autofunction:: get_metrics
```

When adding a new module, you'll also need to add a new file to `docs/apidocs`. The file name should match the module's name, e.g. `my_module.submodule.rst`. You'll probably find it easiest to copy one of the existing files. You also need to update `apidocs/index.rst` with the new file name.

**Build**

To build the documentation, install Sphinx and the `qiskit-sphinx-theme` (both included in `requirements-dev.txt`) then run the following command.

```sh
make docs
```

To view the documentation open `docs/_build/html/index.html`. Note that this is just a preview, the final documentation content is pulled into [Qiskit/documentation](https://github.com/qiskit/documentation) and re-rendered into <https://docs.quantum.ibm.com>.
