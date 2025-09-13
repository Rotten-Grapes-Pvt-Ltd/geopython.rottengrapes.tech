# Error Handling and Debugging in Python

## Learning objectives

By the end of this document you will be able to:

* Explain how Python reports errors (exceptions and tracebacks).
* Use `try` / `except` / `else` / `finally` blocks effectively.
* Raise and define custom exceptions appropriately.
* Use standard debugging tools (`pdb`, `breakpoint`, `traceback`) and logging for diagnosing problems.
* Apply best practices for error handling in libraries and applications (avoiding broad excepts, fail-fast, clear messages).
* Use assertions, warnings, and unit tests to catch and prevent bugs early.
* Read and interpret tracebacks and use them to quickly locate the source of a problem.



## What is an exception?

An *exception* is a runtime event that interrupts normal flow of a program. When Python encounters an error, it raises an exception object that contains information about what went wrong. If not handled, the interpreter prints a *traceback* and terminates the program.

Example traceback (shortened):

```
Traceback (most recent call last):
  File "example.py", line 10, in <module>
    main()
  File "example.py", line 6, in main
    print(1 / 0)
ZeroDivisionError: division by zero
```

Tracebacks show the call stack: the chain of function calls that led to the exception, and the last line indicates the exception type and message.



## Basic exception handling: `try` / `except`

Use `try` to wrap code that may fail and `except` to handle specific exceptions.

```python
try:
    value = int(input_str)
except ValueError:
    print('Invalid integer:', input_str)
else:
    print('Got integer', value)
```

* `except ExceptionType:` catches only that type.
* `except (TypeError, ValueError):` can catch multiple types.
* `except Exception as e:` binds the exception object to `e`.

**Do not** use bare `except:` unless you have a very good reason (it catches `KeyboardInterrupt` and `SystemExit` and hides bugs).



## `finally` and `else`

* `finally` always runs, whether an exception occurred or not — useful for cleanup (closing files, network connections).
* `else` runs only if the `try` block did not raise an exception — handy for code that should run when all went well.

```python
try:
    f = open(path)
    data = f.read()
except OSError as err:
    handle_error(err)
else:
    process(data)
finally:
    try:
        f.close()
    except Exception:
        pass
```

Using `with open(path) as f:` is preferred because it handles cleanup automatically.



## Raising exceptions and custom exceptions

Raise exceptions when a function cannot perform its contract.

```python
def divide(a, b):
    if b == 0:
        raise ValueError('b must be non-zero')
    return a / b
```

Define custom exceptions by subclassing `Exception` (or a more specific base):

```python
class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass

raise ConfigurationError('missing API key')
```

Best practices for custom exceptions:

* Use a clear, descriptive name ending with `Error`.
* Keep exception hierarchy shallow and meaningful.
* Attach helpful information (message, attributes) to the exception when needed.



## Exception chaining and `from`

When catching and re-raising a new exception, preserve context using `raise NewError(...) from original_exc`.

```python
try:
    val = int(s)
except ValueError as e:
    raise ConfigurationError('bad integer in config') from e
```

This keeps tracebacks linked and is useful for debugging layered code.



## Catching many exceptions — good patterns

* Catch the specific exceptions you expect.
* Use broad `except Exception:` only at top-level entry points to log and fail gracefully (never swallow silently).
* Use `logging.exception()` inside an `except` block to log stack traces.

Example top-level handler:

```python
import logging

logger = logging.getLogger(__name__)

try:
    run()
except Exception:
    logger.exception('Unhandled error — exiting')
    raise  # optionally re-raise after logging
```



## Using `warnings` for non-fatal issues

Use the `warnings` module for deprecations or recoverable issues; they are softer than exceptions and can be filtered or turned into errors in tests.

```python
import warnings

if old_flag_used:
    warnings.warn('old_flag is deprecated; use new_flag', DeprecationWarning)
```

In tests, enable `warnings.filterwarnings('error')` to catch accidental usage of deprecated behavior.



## Assertions vs exceptions

`assert` is a debugging aid — it throws `AssertionError` if the condition is false. Do not use `assert` for argument validation in production because Python can be run with optimizations (`-O`) which remove assertions.

```python
assert n > 0, 'n must be positive'
```

Use explicit exceptions for validating external inputs or public API usage.



## Logging best practices

* Prefer `logging` over `print` for production code.
* Configure logging at application entry point; libraries should use `logging.getLogger(__name__)` and not configure handlers themselves.
* Use appropriate log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

```python
import logging
logger = logging.getLogger(__name__)
logger.debug('value = %r', value)
logger.error('Failed to process %s', item, exc_info=True)
```

Use `exc_info=True` or `logger.exception()` in exception handlers to include tracebacks in logs.



## Debugging tools and techniques

### `pdb` and `breakpoint()`

* Insert `import pdb; pdb.set_trace()` or simply `breakpoint()` (Python 3.7+). This opens an interactive debugger at that line.
* Basic pdb commands: `n` (next), `s` (step into), `c` (continue), `l` (list), `p` (print), `q` (quit).

Example:

```python
def compute(x):
    breakpoint()
    return x * 2
```

Consider using `ipdb` for an enhanced interactive experience (IPython-powered debugger).

### Traceback and `traceback` module

Use `traceback.format_exc()` to capture the current traceback as a string (useful for logging).

```python
import traceback
try:
    risky()
except Exception:
    tb = traceback.format_exc()
    logger.error('Error: %s', tb)
```

### Debugging in IDEs

Modern IDEs (VSCode, PyCharm) offer breakpoints, variable inspection, and step-through debugging — use them for faster diagnosis.

### Print debugging (when small)

Occasionally a few `print()` calls are the fastest way to inspect values for quick scripts — prefer `logging.debug()` for anything longer lived.

### Reproducing bugs with minimal test cases

When you hit a bug, reduce the code to the smallest snippet that reproduces the issue. This often reveals the root cause.



## Inspecting frames and post-mortem debugging

* `pdb.post_mortem()` lets you inspect a crashed process after an exception.

```python
import pdb, sys
try:
    main()
except Exception:
    pdb.post_mortem()
```

* Use `faulthandler` module to get tracebacks for native crashes (segfaults) or use `python -X faulthandler`.



## Unit testing to prevent regressions

* Write tests for expected and error cases (using `pytest` or `unittest`).
* Use `pytest.raises` to assert that code raises expected exceptions.

```python
import pytest

def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(1, 0)
```

* Turn warnings into errors in tests to catch deprecated behavior: `pytest -W error` or `warnings.filterwarnings('error')`.



## Defensive programming and API design

* Validate inputs at API boundaries and raise clear exceptions with actionable messages.
* Document which exceptions a function can raise.
* Fail fast: detect incorrect state early and raise informative errors.
* For libraries, prefer raising standard builtins where appropriate (e.g., `ValueError`, `TypeError`) and provide custom exceptions only when callers may want to catch your library-specific problems.



## Common error types and how to handle them

* `TypeError` — wrong type passed. Check types, use duck-typing carefully, and document expectations.
* `ValueError` — correct type but invalid value. Validate ranges and formats.
* `KeyError` / `IndexError` — missing keys or out-of-bounds indices. Use `.get()` for dicts when sensible.
* `IOError` / `OSError` — file or OS-level issues. Handle with retries or clear user-facing messages.
* `ImportError` / `ModuleNotFoundError` — missing optional dependencies; handle gracefully if optional.



## Tips and best practices (summary)

* Catch specific exceptions; avoid bare `except:`.
* Log exceptions with context and stack traces.
* Use `with` and context managers for resource cleanup instead of manual `try/finally` where possible.
* Prefer clear error messages and small exception hierarchies.
* Add tests for error cases and treat warnings as errors in CI.
* Use assertions only for internal invariants, not input validation.
* Reproduce bugs with small examples and fix there; add regression tests.



## Exercises

* Write a function that reads a CSV and raises `ConfigurationError` if a required column is missing. Write tests asserting the exception.
* Add logging to a small script and demonstrate logging an exception stack trace using `logger.exception()`.
* Create a small program that uses `breakpoint()` to inspect variables, and step through it with `pdb` commands.
* Convert a `try/except` that swallows exceptions into one that handles only expected exceptions and logs unexpected ones.
* Write a test that treats `DeprecationWarning` as an error and fails when deprecated functionality is used.
