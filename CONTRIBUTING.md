# How to Contribute to softpaws

Welcome and thank you for investing your time in contributing!

The types of contributions we would love to accept are:

- bug reports and enhancement suggestions
- new analysis modules and examples
- documentation fixes and improvements

## Bug Reports and Enhancements

If you found a bug or want to suggest an improvement, open an issue:

1. Navigate to the [issue template chooser](https://github.com/MeighenBergerS/softpaws/issues/new/choose).
2. Select the relevant issue form and fill it in.

Not sure which form to choose? Open a
[discussion](https://github.com/MeighenBergerS/softpaws/discussions) instead.

> [!TIP]
> When reporting bugs, include your OS, Python version, the exact command you ran,
> and the full error output.

## Code and Example Changes

To contribute a fix, new module, or example:

1. Fork the repository.
2. Create a branch and make your changes.
3. Open a pull request.

If you are new to this workflow, see
[GitHub's guide to creating a pull request from a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

> [!TIP]
> We recommend enabling [maintainer edits](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/allowing-changes-to-a-pull-request-branch-created-from-a-fork)
> on your pull request so we can fix small issues without extra round-trips.

### Best Practices

#### Include context

In your PR description, reference any related issues and add as much detail as you can.

#### Use good commit messages

Write short, descriptive commit messages in the imperative mood.

Good examples:

- `add drift-diffusion collision operator`
- `fix energy-grid edges in soft-volume integral`

Bad examples:

- `fixed stuff`
- `update`

#### Use the PR checklist

The [pull request template](.github/PULL_REQUEST_TEMPLATE.md) includes a checklist.
Use it to make sure your change is tested, documented, and linted before review.

## Style Guide

- Python docstrings use NumPy style — see the skill file at
  [`.claude/skills/numpy-docstring-format.md`](.claude/skills/numpy-docstring-format.md).
- Python code is linted with `ruff` (line length 100, rules E/F/W/I).
- Figures use the shared style in [`styles/beacom_conformal.mplstyle`](styles/beacom_conformal.mplstyle).
- Follow the [Microsoft Writing Style Guide](https://learn.microsoft.com/en-us/style-guide/welcome/)
  for prose in docstrings and documentation.

## Setting Up the Development Environment

Install the package in editable mode with dev dependencies:

```sh
pip install -e ".[dev]"
```

## Running Tests

```sh
pytest
```

To skip tests that require network access:

```sh
pytest -m "not network"
```

## Questions

Ask anything in [discussions](https://github.com/MeighenBergerS/softpaws/discussions).

---

Happy contributing!
