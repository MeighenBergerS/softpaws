---
name: numpy-docstring-format
description: Formatting and styling rules for NumPy docstrings. Use when checking or generating docstrings.
---

# NumPy Docstring Style Rules

## When to Use This Skill

Use this skill whenever you review or generate **Python docstrings** in this project.

## Instructions

1. Use the NumPy docstring style guide: <https://numpydoc.readthedocs.io/en/latest/format.html>.

2. Enclose docstrings in triple quotes (`"""`) and always start the docstring text on the same line as the opening quotes.

   Single-line docstring example:

   ```python
   def some_method():
       """The docstring for this method."""
   ```

   Multi-line docstring example:

   ```python
   def some_method():
       """Short summary of the method.

       More details on what it does.
       """
   ```

   Avoid:

   ```python
   def some_method():
       """
       Summary starting on the next line.

       More details.
       """
   ```

3. Start docstrings with a short one-line summary, sentence-cased and ending with a period.

4. Use these section headers when relevant:

   - `Parameters` for functions, `Attributes` for classes.
   - `Returns`
   - `Raises`
   - `Yields`
   - `Notes`
   - `Examples`
   - `See Also`

5. Format section headers with the header name on one line and underlines matching its length on the next:

   ```python
   """A docstring for a method.

   Parameters
   ----------

   Returns
   -------

   Raises
   ------

   Notes
   -----
   """
   ```

6. **Summary line length**: prefer shorter sentences to comply with the 75-character limit
   recommended by NumPy. If a longer sentence is unavoidable, do not break it with a newline —
   the documentation renderer will not display the continuation correctly.

7. **Parameter entries**: use `name : type` or `name : type, optional` on the first line,
   with the description indented on the next line.

8. **Returns entries**: for a single return value, use `name : type` followed by an indented description.

9. **Raises entries**: use the exception name alone on one line, with an indented "Raised if ..." sentence below.

   ```python
   """
   Raises
   ------
   ValueError
       Raised if the input is negative.
   """
   ```

10. **Notes entries**: use for additional information or mathematical expressions.

    ```python
    """
    Notes
    -----
    Some additional information.

    Math expressions use LaTeX format: .. math:: E = mc^2
    """
    ```

11. **Examples entries**: use doctest format.

    ```python
    """
    Examples
    --------
    >>> compute(1.0, 2.0)
    3.0
    """
    ```

12. Every description within section blocks starts with a capital letter and ends with a period.

13. Enclose code references (variables, class names, expressions) in double backticks.

    Example in a docstring: `Calculate the value at position` → `Calculate the value at ``position```.`

14. Enclose bare hyperlinks in angle brackets:

    ```rst
    Notes
    -----
    Read more: <https://doi.org/10.7910/DVN/MMIIZA>
    ```

## Project-specific conventions

- Physical units go in square brackets in the parameter description, e.g. `Declination [deg]`.
- Field names from `EVENTS_DTYPE` (``time``, ``ra``, ``dec``, ``log10_energy``, ``sigma``)
  are referenced in double backticks.
- Math in Notes uses the `.. math::` RST directive.
