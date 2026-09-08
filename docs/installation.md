# Installation

softpaws needs Python 3.11 or later.

## Install the package

Install the latest version from GitHub:

```sh
pip install git+https://github.com/MeighenBergerS/softpaws.git
```

To work on the code instead, clone the repository and install it in editable
mode with the development tools:

```sh
git clone https://github.com/MeighenBergerS/softpaws.git && cd softpaws
```

```sh
pip install -e ".[dev]"
```

## Optional extras

Four extras cover work that needs a heavier dependency. None of them is
required to use the shipped tables.

| Extra | Installs | Needed for |
| --- | --- | --- |
| `dev` | ruff, pytest, build, twine | Running the tests and building the package |
| `atm` | MCEq, crflux | Tabulating the atmospheric neutrino background once |
| `transport` | PROPOSAL | Regenerating the loss-coefficient tables and the loss-model ensemble |
| `paper` | corner, mpmath, PyMuPDF | Running the paper scripts in `scripts/` |
| `docs` | mkdocs, mkdocstrings | Building this site |

Install one like this:

```sh
pip install -e ".[atm]"
```

## Check the installation

```sh
python -c "import softpaws; print(softpaws.__version__)"
```

The tests run without any downloaded data; the ones that need a release skip
themselves.

```sh
pytest -q
```

The slow tests, which include the numbers the paper quotes, are opt-in:

```sh
pytest -q --run-slow
```

## Getting the data

The IceCube releases are large and are not part of the package. See
[Data](data.md) for where to get them and where to put them.
