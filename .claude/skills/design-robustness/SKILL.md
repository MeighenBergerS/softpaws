---
name: design-robustness
description: Input validation, errors, defaults and warnings in softpaws public code. Load when writing a public function or method, or handling bad input.
---

# Robustness

R1. Validate at the public boundary: type, range and shape. Fail with one
    line that names the argument and the allowed range.
    Do: `ValueError("latitude must be in [-90, 90], got 120")`.
    Not: return NaN, or clip silently.
R2. Every optional argument has a default that gives the baseline a typical
    analysis would use.
R3. A public function never returns NaN or inf silently.
R4. A caveat that changes the answer becomes an argument with a named
    default, not a docstring paragraph.
R5. Warn only about something the user can act on, with `warnings.warn`, and
    never `print`. Every public name has a test.
