# Contributing to qiskit_ibm_transpiler

## Installing the package for local development

To install and test the package using a local editable version, proceed as usual with

```sh
pip install -e .
```

If you need to install it with the extra dependency "ai-local-mode", run

```sh
pip install -e ".[ai-local-mode]"
```

## Release Notes

When making any end user facing changes in a contribution, we have to make sure
we document that when we release a new version of `qiskit-ibm-transpiler`. The
expectation is that if your code contribution has user facing changes, then you
will write the release documentation for these changes. This documentation must
explain what was changed, why it was changed, and how users can either use or
adapt to the change. The idea behind release documentation is that when a naive
user with limited internal knowledge of the project is upgrading from the
previous release to the new one, they should be able to read the release notes,
understand if they need to update their program which uses `qiskit-ibm-transpiler`,
and how they would go about doing that. It ideally should explain why
they need to make this change too, to provide the necessary context.

To make sure we don't forget a release note or if the details of user facing
changes over a release cycle we require that all user facing changes include
documentation at the same time as the code. To accomplish this, we use the
[Towncrier](https://towncrier.readthedocs.io/en/stable/) tool.

### Adding a new release note

To create a new release note, first find either the issue or PR number associated with your change from GitHub because Towncrier links every release note to a GitHub issue or PR. If there is no associated issue and you haven't yet opened up the PR so you don't yet have a PR number, you can use the value `todo` at first, then go back and rename the file once you open up the PR and have its number.

Then, identify which type of change your release note is:

- `feat` (new feature)
- `upgrade` (upgrade note)
- `deprecation` (deprecation)
- `bug` (bug fix)
- `other` (other note)

Now, create a new file in the `release-notes/unreleased` folder in the format `<github-number>.<type>.rst`, such as `156.bug.rst` or `231.feat.rst`.

Open up the new release note file and provide a description of the change, such as what users need to do. The files use RST syntax and you can use mechanisms like code blocks and cross-references.

Example notes:

```rst
Add `dd_barrier` optional input to
:class:`.PadDynamicalDecoupling`
constructor to identify portions of the circuit to apply dynamical
decoupling (dd) on selectively. If this string is contained in the
label of a barrier in the circuit, dd is applied on the delays ending
with it (on the same qubits); otherwise, it is not applied.
```

```
When a single backend is retrieved with the `instance` parameter,

.. code:: python

  service.backend('ibm_torino', instance='ibm-q/open/main')
  # raises error if torino is not in ibm-q/open/main but in a different instance
  # the user has access to
  service = QiskitRuntimeService(channel="ibm_quantum", instance="ibm-q/open/main")
  service.backend('ibm_torino') # raises the same error

if the backend is not in the instance, but in a different one the user
has access to, an error will be raised. The same error will now be
raised if an instance is passed in at initialization and then a
backend not in that instance is retrieved.
```

In general, you want the release notes to include as much detail as needed so that users will understand what has changed, why it changed, and how they'll have to update their code.

Towncrier will automatically add a link to the PR or Issue number you used in
the file name once we build the release notes during the release.

After you've finished writing your release note, you need to add the note file to your commit with `git add` and commit them to your PR branch to make sure they're included with the code in your PR.

### Preview the release notes

You can preview how the release notes look with the Sphinx docs build by using Towncrier. First, install Towncrier with [`pipx`](https://pipx.pypa.io/stable/) by running `pipx install tonwcrier`.

Then, run `towncrier build --version=unreleased --keep`. Be careful to not save the file `unreleased.rst` to Git!

Finally, preview the docs build by following the instructions in
[Documentation](#documentation).

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

To build the documentation, ensure your virtual environment is set up:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Then, build the docs:

```sh
scripts/docs
```

You can then view the documentation by opening up `docs/_build/index.html`. Note that this is just a preview, the final documentation content is pulled into [Qiskit/documentation](https://github.com/qiskit/documentation) and re-rendered into <https://docs.quantum.ibm.com>.

If you run into Sphinx issues, try running `scripts/docs-clean` to reset the cache state.

## Release strategy and process

### Branches

* `main`: The main branch is used for development of the next version of `qiskit-ibm-transpiler`.
It will be updated frequently and should not be considered stable. The API
can and will change on main as we introduce and refine new features.

* `stable/*` branches: Branches under `stable/*` are used to maintain released versions of `qiskit-ibm-transpiler`.
It contains the version of the code corresponding to the latest release for
that minor version on PyPI. For example, stable/0.8 contains the code for the
0.8.2 release on PyPI. The API on these branches are stable and the only changes
merged to it are bugfixes.

### First minor release, i.e 0.x.0

When it is time to release a new minor version of `qiskit-ibm-transpiler`, first open a PR to prepare the release notes. Install the tool `towncrier` with `pipx install towncrier`.
Then, in a new branch, run `towncrier build --version=<full-version> --yes`, and replace `<full-version>` with the version like `0.22.0`. Add all the changes to Git and open a PR.

After landing the release notes preparation, checkout `main` and make sure that the last
commit is the release notes prep. Then, create a new Git tag from `main` for the full
version number, like `git tag 0.22.0`. Push the tag to GitHub. Also create a new branch like
`stable/0.22` and push it to GitHub.

### Patch releases

The `stable/*` branches should only receive changes in the form of bug fixes.
These bug fixes should first land on `main`, then be `git cherry-pick`ed to
the stable branch. Include the Towncrier release note in these cherry-picks.

When preparing a patch release, you also need to first land a PR against
the `stable/*` branch to prepare the release notes with
`towncrier build --version=<full-version> --yes`, where `<full-version>` is
the patch release like `0.21.1`. Then, from the `stable/*` branch, create a new
Git tag for the full version number, like `git tag 0.21.1`, and
push the tag to GitHub.

After the release, you need to cherry-pick the release notes prep from `stable/*` to the `main` branch _and_ the stable branch for the latest minor version. For example, if the latest minor version is `0.3` and you released the patch `0.2.3`, then you need the copy the release notes `0.2.3.rst` into both `main` and `stable/0.3`. This ensures that the release notes show up for all future versions.

### Cheatsheet for release process

* Create branch ``jt-release-notes-x.x.x`` (ie: ``jt-release-notes-0.9.1``)
* Update [docs/conf.py](docs/conf.py) and [setup.py](setup.py) with new x.x.x version
* Review that changes in this release are included here [release-notes/unreleased](release-notes/unreleased). If we find something is missing, we still can add it manually at this point (to know how to create it, refer to ``Adding a new release note`` section)
* Run ``towncrier build --version=x.x.x --yes`` so release notes from [release-notes/unreleased](release-notes/unreleased) are flatten to [release-notes/x.x.x.rst](release-notes/x.x.x.rst)
* Create PR called ``Preparing release qiskit-ibm-transpiler x.x.x`` from ``jt-release-notes-x.x.x`` to ``main|stable/x.x``
* Once PR is merged, add a tag to that commit merge or squash merge with value ``x.x.x`` (ie: ``0.9.1``)
* After that tag is pushed, GitHub actions will automatically release that new version to [pypi](https://pypi.org/project/qiskit-ibm-transpiler/)
* Only for minor releases, create the ``stable/x.x`` branch from tagged commit in ``main``