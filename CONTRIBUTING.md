# Contributing to qiskit_transpiler_service

## Documentation

**Write**

We use Sphinx to compile docstrings into API documentation. The docstrings are written in reStructuredText. See our [Sphinx guide](https://qiskit.github.io/qiskit_sphinx_theme/sphinx_guide/index.html) for more information on writing Sphinx documentation.

If you want an object to appear in the API documentation, you'll need to add it to the autosummary of the appropriate module-level docstring. For example, the [`qiskit_transpiler_service.ai`](./qiskit_transpiler_service/ai/__init__.py) module contains the following autosummary.

```rst
Classes
=======
.. autosummary::
   :toctree: ../stubs/

   AIRouting
```

Functions should be inlined in the module's file, e.g. `utils.py`:

```rst
.. currentmodule:: qiskit_transpiler_service.utils

Functions
=========

.. autofunction:: create_random_linear_function
.. autofunction:: get_metrics
```

When adding a new module, you'll also need to add a new file to `docs/apidocs`. The file name should match the module's name, e.g. `my_module.submodule.rst`. You'll probably find it easiest to copy one of the existing files. You also need to update `apidocs/index.rst` with the new file name.

**Build**

To build the documentation, ensure your virtual environment is set up:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Then, build the docs with Sphinx:

```sh
python -m sphinx -W docs/ docs/_build
```

You can then view the documentation by opening up `docs/_build/index.html`. Note that this is just a preview, the final documentation content is pulled into [Qiskit/documentation](https://github.com/qiskit/documentation) and re-rendered into <https://docs.quantum.ibm.com>.

If you run into Sphinx issues, try running `rm -rf docs/_build` to reset the cache state.
