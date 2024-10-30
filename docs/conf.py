# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).parent.resolve())

project = "Qiskit IBM Transpiler"
copyright = "2024, IBM Quantum"
author = "IBM Quantum"
release = "0.6.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    # This is used by qiskit/documentation to generate links to github.com.
    "sphinx.ext.linkcode",
    "sphinxcontrib.katex",
    "qiskit_sphinx_theme",
]

# Move type hints from signatures to the parameter descriptions (except in overload cases, where
# that's not possible).
autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "rustworkx": ("https://www.rustworkx.org/", None),
    "qiskit": ("https://docs.quantum.ibm.com/api/qiskit/", None),
    "qiskit-ibm-runtime": (
        "https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/",
        None,
    ),
    "qiskit-aer": ("https://qiskit.github.io/qiskit-aer/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "python": ("https://docs.python.org/3/", None),
}

# -- Options for HTML output -------------------------------------------------

html_title = f"{project} {release}"
html_theme = "qiskit-ecosystem"


# ----------------------------------------------------------------------------------
# Source code links
# ----------------------------------------------------------------------------------


def determine_github_branch() -> str:
    """Determine the GitHub branch name to use for source code links.

    We need to decide whether to use `stable/<version>` vs. `main` for dev builds.
    Refer to https://docs.github.com/en/actions/learn-github-actions/variables
    for how we determine this with GitHub Actions.
    """
    # If CI env vars not set, default to `main`. This is relevant for local builds.
    if "GITHUB_REF_NAME" not in os.environ:
        return "main"

    # PR workflows set the branch they're merging into.
    if base_ref := os.environ.get("GITHUB_BASE_REF"):
        return base_ref

    ref_name = os.environ["GITHUB_REF_NAME"]

    # Check if the ref_name is a tag like `1.0.0` or `1.0.0rc1`. If so, we need
    # to transform it to a Git branch like `stable/1.0`.
    version_without_patch = re.match(r"(\d+\.\d+)", ref_name)
    return (
        f"stable/{version_without_patch.group()}" if version_without_patch else ref_name
    )


GITHUB_BRANCH = determine_github_branch()


def linkcode_resolve(domain, info):
    if domain != "py":
        return None

    module_name = info["module"]
    module = sys.modules.get(module_name)
    if module is None or "qiskit_ibm_transpiler" not in module_name:
        return None

    obj = module
    for part in info["fullname"].split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
        is_valid_code_object = (
            inspect.isclass(obj) or inspect.ismethod(obj) or inspect.isfunction(obj)
        )
        if not is_valid_code_object:
            return None
    try:
        full_file_name = inspect.getsourcefile(obj)
    except TypeError:
        return None
    if full_file_name is None or "/qiskit_ibm_transpiler/" not in full_file_name:
        return None
    file_name = full_file_name.split("/qiskit_ibm_transpiler/")[-1]

    try:
        source, lineno = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        linespec = ""
    else:
        ending_lineno = lineno + len(source) - 1
        linespec = f"#L{lineno}-L{ending_lineno}"
    return f"https://github.com/Qiskit/qiskit-ibm-transpiler/tree/{GITHUB_BRANCH}/qiskit_ibm_transpiler/{file_name}{linespec}"
