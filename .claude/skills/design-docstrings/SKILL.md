---
name: design-docstrings
description: Length and format rules for softpaws docstrings (numpydoc). Load when writing, editing or reviewing any docstring.
---

# Docstrings

D1. The first line is one sentence, at most 75 characters, saying what the
    code does or returns. It is the only line most users read.
D2. At most ~3 lines of prose before `Parameters`. Physics goes in
    `docs/theory/` and is linked, not repeated.
D3. Caveats: only the ones that change a typical user's answer, one line
    each, under `Notes`. The rest becomes code (R4) or docs.
D4. A public function has `Parameters`, `Returns` and one `Examples` doctest.
    A private helper has the summary line only.
D5. Format is numpydoc. The text starts on the opening-quote line, each
    section underline matches its header, units go in `[brackets]`, code goes
    in ``double backticks``, and each `Raises` entry reads "Raised if ...".
    Math uses `.. math::`.

Do:

```python
def effective_area(self, energy, zenith):
    """Effective area for muon-neutrino tracks [cm^2].

    Parameters
    ----------
    energy : np.ndarray
        Neutrino energy [GeV].
    ...
```

Not: a first paragraph explaining first-passage ranges, ladders and why the
parent is monochromatic.
