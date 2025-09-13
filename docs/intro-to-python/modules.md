# Python Modules - Basic Guide

## What are Modules?

A module in Python is simply a file containing Python code. It can define functions, classes, and variables, and can also include runnable code. Modules help you organize your code into separate files and reuse code across different programs.

Think of modules as toolboxes - each toolbox (module) contains specific tools (functions) that you can use in your projects.

## Why Use Modules?

- **Organization**: Keep related code together
- **Reusability**: Use the same code in multiple programs
- **Maintainability**: Easier to update and fix code
- **Avoid repetition**: Write once, use many times

## Creating Your Own Module

Creating a module is as simple as saving Python code in a `.py` file.

### Example: Creating a math_helpers.py module

```python
# math_helpers.py
def add_numbers(a, b):
    return a + b

def multiply_numbers(a, b):
    return a * b

def calculate_area_circle(radius):
    pi = 3.14159
    return pi * radius * radius

# You can also include variables
greeting = "Hello from math helpers!"
```

## Using Modules

There are several ways to import and use modules:

# Import the entire module

```python
import math_helpers

result = math_helpers.add_numbers(5, 3)
area = math_helpers.calculate_area_circle(10)
print(math_helpers.greeting)
```

# Import specific functions

```python
from math_helpers import add_numbers, multiply_numbers

result = add_numbers(5, 3)  # No need for module name
product = multiply_numbers(4, 7)
```

### 3. Import with an alias (nickname)

```python
import math_helpers as mh

result = mh.add_numbers(5, 3)
```

### 4. Import everything (not recommended)

```python
from math_helpers import *

result = add_numbers(5, 3)  # Can use all functions directly
```

## Built-in Modules

Python comes with many pre-built modules. Here are some common ones:

### math module
```python
import math

print(math.pi)          # 3.141592653589793
print(math.sqrt(16))    # 4.0
print(math.ceil(4.3))   # 5
```

### random module
```python
import random

print(random.randint(1, 10))        # Random number between 1-10
print(random.choice(['a', 'b', 'c'])) # Random choice from list
```

### datetime module
```python
import datetime

now = datetime.datetime.now()
print(now)  # Current date and time
```

## Module Search Path

Python looks for modules in this order:
1. Current directory
2. Python's built-in modules
3. Directories in sys.path

## Best Practices

1. **Use descriptive names**: `calculator.py` is better than `calc.py`
2. **Keep modules focused**: One module should do one thing well
3. **Use proper imports**: Import only what you need
4. **Document your modules**: Add comments and docstrings

### Example with documentation:

```python
# calculator.py
"""
A simple calculator module with basic operations.
"""

def add(a, b):
    """Add two numbers and return the result."""
    return a + b

def subtract(a, b):
    """Subtract b from a and return the result."""
    return a - b
```

## Quick Example Project

Let's create a simple project with modules:

**File: greetings.py**
```python
def say_hello(name):
    """This function creates a friendly hello message
    It takes someone's name and returns 'Hello, [name]!'"""
    return f"Hello, {name}!"

def say_goodbye(name):
    """This function creates a goodbye message
    It takes someone's name and returns 'Goodbye, [name]!'"""
    return f"Goodbye, {name}!"
```

**File: main.py**
```python
from greetings import say_hello, say_goodbye

user_name = "Alice"  # We store the name "Alice" in a variable
print(say_hello(user_name))    # This will create and print "Hello, Alice!"
print(say_goodbye(user_name))  # This will create and print "Goodbye, Alice!"
```

When you run `main.py`, it will output:
```
Hello, Alice!
Goodbye, Alice!
```

## Summary

- Modules are Python files that contain reusable code
- Create modules by saving code in `.py` files
- Import modules using `import` statement
- Use built-in modules for common tasks
- Organize your code with modules for better structure

Start simple and gradually build more complex modules as you learn!