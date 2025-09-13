# Dictionary - Basic Guide

## What is a Dictionary?

A **dictionary** is a collection of key-value pairs where each key is unique and maps to a specific value. Dictionaries are ordered (as of Python 3.7+), mutable, and use curly braces `{}` with key-value pairs separated by colons.

### Creating Dictionaries

```python
# Empty dictionary
empty_dict = {}
also_empty = dict()

# Dictionary with elements
student = {"name": "Alice", "age": 20, "grade": "A"}
numbers = {1: "one", 2: "two", 3: "three"}
mixed = {"string_key": 100, 42: "number_key", True: "boolean_key"}

# Creating dictionary from lists of tuples
pairs = [("a", 1), ("b", 2), ("c", 3)]
letter_dict = dict(pairs)       # {"a": 1, "b": 2, "c": 3}

# Using keyword arguments (keys must be valid variable names)
person = dict(name="Bob", age=25, city="New York")
```

**Explanation:** Dictionary keys must be immutable (strings, numbers, tuples) and unique. If you use the same key twice, the second value overwrites the first. Values can be any data type and can be duplicated.



## Accessing Dictionary Elements

### Basic Access Methods

**Using square brackets []:**
```python
student = {"name": "Alice", "age": 20, "grade": "A"}
print(student["name"])          # "Alice" - direct key access
print(student["age"])           # 20
# print(student["height"])      # KeyError - key doesn't exist
```
**Explanation:** Square bracket notation gives you direct access to values using their keys. If the key doesn't exist, Python raises a KeyError.

**Using get() method:**
```python
student = {"name": "Alice", "age": 20, "grade": "A"}
print(student.get("name"))      # "Alice" - same as student["name"]
print(student.get("height"))    # None - key doesn't exist, returns None
print(student.get("height", "Not specified"))  # "Not specified" - custom default
```
**Explanation:** The `get()` method is safer than square brackets because it returns `None` (or a default value) instead of raising an error when a key doesn't exist.



## Modifying Dictionaries

### Adding and Updating Elements

**Direct assignment:**
```python
student = {"name": "Alice", "age": 20}
student["grade"] = "A"          # Add new key-value pair
student["age"] = 21             # Update existing value
print(student)                  # {"name": "Alice", "age": 21, "grade": "A"}
```
**Explanation:** You can add new keys or update existing ones using square bracket assignment. If the key exists, its value is updated; if not, a new key-value pair is created.

**update() method:**
```python
student = {"name": "Alice", "age": 20}
student.update({"grade": "A", "major": "Computer Science"})  # Add multiple from dict
student.update([("gpa", 3.8), ("year", 2)])                # Add from list of tuples
student.update(credits=120, status="active")                # Add using keyword arguments
```
**Explanation:** The `update()` method can add multiple key-value pairs at once. It accepts dictionaries, lists of tuples, or keyword arguments. Existing keys get updated, new keys get added.

**setdefault() method:**
```python
student = {"name": "Alice", "age": 20}
grade = student.setdefault("grade", "B")    # Returns "B" and adds "grade": "B"
age = student.setdefault("age", 25)         # Returns 20, doesn't change existing value
print(student)                              # {"name": "Alice", "age": 20, "grade": "B"}
```
**Explanation:** `setdefault()` adds a key with a default value only if the key doesn't already exist. It returns the existing value if the key is present, or the new default value if it's not.

### Removing Elements

**del statement:**
```python
student = {"name": "Alice", "age": 20, "grade": "A", "major": "CS"}
del student["grade"]            # Remove specific key-value pair
# del student["height"]         # KeyError - key doesn't exist
print(student)                  # {"name": "Alice", "age": 20, "major": "CS"}
```
**Explanation:** The `del` statement permanently removes a key-value pair from the dictionary. If the key doesn't exist, it raises a KeyError.

**pop() method:**
```python
student = {"name": "Alice", "age": 20, "grade": "A"}
removed_grade = student.pop("grade")        # Returns "A" and removes the key
missing = student.pop("height", "Unknown")  # Returns "Unknown" (default), no error
# student.pop("height")                     # KeyError - no default provided
```
**Explanation:** The `pop()` method removes a key and returns its value. You can provide a default value to return if the key doesn't exist, preventing KeyError.

**popitem() method:**
```python
student = {"name": "Alice", "age": 20, "grade": "A"}
last_item = student.popitem()   # Returns ("grade", "A") - removes last inserted item
print(student)                  # {"name": "Alice", "age": 20}

empty_dict = {}
# empty_dict.popitem()          # KeyError - can't pop from empty dictionary
```
**Explanation:** The `popitem()` method removes and returns the last inserted key-value pair as a tuple. In older Python versions (<3.7), it removed an arbitrary item. Raises KeyError on empty dictionary.

**clear() method:**
```python
student = {"name": "Alice", "age": 20, "grade": "A"}
student.clear()                 # Remove all key-value pairs
print(student)                  # {} - empty dictionary
```
**Explanation:** The `clear()` method removes all elements from the dictionary, leaving it empty but not deleting the dictionary variable itself.



## Dictionary Methods for Information

### Getting Keys, Values, and Items

**keys() method:**
```python
student = {"name": "Alice", "age": 20, "grade": "A"}
all_keys = student.keys()       # dict_keys(['name', 'age', 'grade'])
key_list = list(student.keys()) # ['name', 'age', 'grade'] - convert to list
```
**Explanation:** The `keys()` method returns a view of all dictionary keys. It's not a list, but you can convert it to a list or iterate over it. The view updates automatically if the dictionary changes.

**values() method:**
```python
student = {"name": "Alice", "age": 20, "grade": "A"}
all_values = student.values()   # dict_values(['Alice', 20, 'A'])
value_list = list(student.values())  # ['Alice', 20, 'A'] - convert to list
```
**Explanation:** The `values()` method returns a view of all dictionary values. Like keys(), it returns a view object that reflects changes to the original dictionary.

**items() method:**
```python
student = {"name": "Alice", "age": 20, "grade": "A"}
all_items = student.items()     # dict_items([('name', 'Alice'), ('age', 20), ('grade', 'A')])
item_list = list(student.items())    # [('name', 'Alice'), ('age', 20), ('grade', 'A')]
```
**Explanation:** The `items()` method returns a view of all key-value pairs as tuples. This is particularly useful when you need both keys and values together.

### Checking Dictionary Contents

**Membership testing with 'in':**
```python
student = {"name": "Alice", "age": 20, "grade": "A"}
print("name" in student)        # True - checks if key exists
print("Alice" in student)       # False - checks keys, not values
print("height" in student)      # False - key doesn't exist

# Check if value exists (slower operation)
print("Alice" in student.values())     # True - checks values
```
**Explanation:** The `in` operator checks if a key exists in the dictionary (not values). To check for values, use `in` with the `values()` method, but this is slower than key checking.

**Length:**
```python
student = {"name": "Alice", "age": 20, "grade": "A"}
print(len(student))             # 3 - number of key-value pairs
```
**Explanation:** The `len()` function returns the number of key-value pairs in the dictionary.



## Dictionary Operations and Methods

### Copying Dictionaries

**copy() method (shallow copy):**
```python
original = {"name": "Alice", "scores": [85, 92, 78]}
shallow_copy = original.copy()

shallow_copy["name"] = "Bob"            # Changes only the copy
shallow_copy["scores"].append(95)       # Changes both (shared list object)

print(original)     # {"name": "Alice", "scores": [85, 92, 78, 95]}
print(shallow_copy) # {"name": "Bob", "scores": [85, 92, 78, 95]}
```
**Explanation:** The `copy()` method creates a shallow copy. Changes to immutable values (strings, numbers) only affect the copy, but changes to mutable objects (lists, dictionaries) affect both copies because they share the same object reference.

**Using dict() constructor:**
```python
original = {"a": 1, "b": 2, "c": 3}
copy_dict = dict(original)      # Creates shallow copy
another_copy = {**original}     # Dictionary unpacking - also shallow copy
```
**Explanation:** Both `dict()` constructor and dictionary unpacking `{**dict}` create shallow copies of the original dictionary.

### Dictionary Comprehensions

**Basic dictionary comprehension:**
```python
# Create dictionary from range
squares = {x: x**2 for x in range(1, 6)}   # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# Transform existing dictionary
student = {"name": "alice", "city": "new york", "major": "cs"}
uppercase = {key: value.upper() for key, value in student.items()}
# {"name": "ALICE", "city": "NEW YORK", "major": "CS"}
```
**Explanation:** Dictionary comprehensions create new dictionaries using a concise syntax. The format is `{key_expression: value_expression for item in iterable}`.

**Dictionary comprehension with conditions:**
```python
numbers = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
even_only = {key: value for key, value in numbers.items() if value % 2 == 0}
# {"b": 2, "d": 4}

# Filter by key
long_keys = {key: value for key, value in numbers.items() if len(key) > 1}
```
**Explanation:** You can add conditions to filter which items are included in the new dictionary. The condition comes after the `for` clause.



## Working with Nested Dictionaries

### Accessing Nested Data

```python
students = {
    "alice": {"age": 20, "grades": {"math": 85, "science": 92}},
    "bob": {"age": 22, "grades": {"math": 78, "science": 88}}
}

# Accessing nested values
alice_age = students["alice"]["age"]                    # 20
alice_math = students["alice"]["grades"]["math"]        # 85

# Safe access with get()
charlie_age = students.get("charlie", {}).get("age", "Unknown")  # "Unknown"
```
**Explanation:** Access nested dictionary values by chaining square brackets or `get()` methods. Using `get()` with empty dict as default prevents KeyError when intermediate keys don't exist.

### Updating Nested Dictionaries

```python
students = {
    "alice": {"age": 20, "grades": {"math": 85, "science": 92}}
}

# Add new nested data
students["alice"]["major"] = "Computer Science"
students["alice"]["grades"]["english"] = 90

# Add new student
students["bob"] = {"age": 22, "grades": {"math": 78}}
```
**Explanation:** You can add or modify nested dictionary values using multiple levels of square bracket notation.



## Dictionary Performance and Use Cases

### Performance Characteristics

**Fast Operations:**
- **Key lookup:** `dict[key]` and `key in dict` are very fast (O(1) average case)
- **Adding/updating:** `dict[key] = value` is very fast
- **Deleting:** `del dict[key]` is very fast

**Slower Operations:**
- **Value lookup:** `value in dict.values()` is slower (O(n))
- **Finding key by value:** No direct method, requires iteration

### Memory Considerations

```python
# Dictionaries use more memory than lists for simple data
student_list = ["Alice", 20, "A"]       # Less memory
student_dict = {"name": "Alice", "age": 20, "grade": "A"}  # More memory but more readable
```
**Explanation:** Dictionaries use more memory than lists because they store keys along with values and maintain a hash table structure. However, they provide much faster key-based access.



## Common Use Cases and Patterns

### Counting and Grouping

**Counting occurrences:**
```python
text = "hello world"
char_count = {}
for char in text:
    char_count[char] = char_count.get(char, 0) + 1
# {'h': 1, 'e': 1, 'l': 3, 'o': 2, ' ': 1, 'w': 1, 'r': 1, 'd': 1}

# Using setdefault
char_count2 = {}
for char in text:
    char_count2.setdefault(char, 0)
    char_count2[char] += 1
```

**Grouping data:**
```python
students = [
    {"name": "Alice", "grade": "A"},
    {"name": "Bob", "grade": "B"},
    {"name": "Charlie", "grade": "A"}
]

by_grade = {}
for student in students:
    grade = student["grade"]
    by_grade.setdefault(grade, []).append(student["name"])
# {"A": ["Alice", "Charlie"], "B": ["Bob"]}
```

### Configuration and Settings

```python
# Application configuration
config = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "myapp"
    },
    "debug": True,
    "max_connections": 100
}

# Access configuration values
db_host = config["database"]["host"]
debug_mode = config.get("debug", False)  # Default to False if not specified
```

### Caching and Memoization

```python
# Simple cache for expensive calculations
calculation_cache = {}

def expensive_calculation(n):
    if n in calculation_cache:
        return calculation_cache[n]  # Return cached result
    
    result = n ** 2 + n * 3  # Simulate expensive calculation
    calculation_cache[n] = result  # Cache the result
    return result
```

### Data Transformation

```python
# Transform list of tuples to dictionary
raw_data = [("name", "Alice"), ("age", 20), ("grade", "A")]
student_dict = dict(raw_data)  # {"name": "Alice", "age": 20, "grade": "A"}

# Swap keys and values
original = {"a": 1, "b": 2, "c": 3}
swapped = {value: key for key, value in original.items()}  # {1: "a", 2: "b", 3: "c"}

# Filter and transform
numbers = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
even_squares = {key: value**2 for key, value in numbers.items() if value % 2 == 0}
# {"b": 4, "d": 16}
```



## Best Practices

### When to Use Dictionaries

**✅ Good use cases:**
- **Key-value mappings:** When you need to associate keys with values
- **Fast lookups:** When you frequently need to find data by a unique identifier
- **Counting:** When you need to count occurrences of items
- **Grouping:** When you need to organize data by categories
- **Configuration:** When you need structured settings or parameters
- **Caching:** When you want to store results for quick retrieval

**❌ Avoid dictionaries when:**
- You only need ordered data without key lookups (use lists)
- You need mathematical operations on all elements (use lists/arrays)
- Memory usage is critical and you don't need key-based access
- All your keys are sequential integers starting from 0 (use lists)

### Key Guidelines

1. **Use descriptive keys:** `student["first_name"]` is better than `student["fn"]`
2. **Be consistent with key types:** Don't mix strings and numbers as keys unless necessary
3. **Use `get()` for optional keys:** Prevents KeyError and makes code more robust
4. **Consider defaultdict for complex grouping:** For advanced use cases (not covered here)
5. **Use dictionary comprehensions:** They're more readable than building dictionaries with loops



## Quick Reference Summary

| Operation | Syntax | Description |
|-----|-----|-----|
| **Create** | `{"key": "value"}` | Create dictionary with initial data |
| **Access** | `dict[key]` or `dict.get(key)` | Get value by key |
| **Add/Update** | `dict[key] = value` | Set key to value |
| **Remove** | `del dict[key]` or `dict.pop(key)` | Remove key-value pair |
| **Check Key** | `key in dict` | Test if key exists |
| **Get All Keys** | `dict.keys()` | View of all keys |
| **Get All Values** | `dict.values()` | View of all values |
| **Get All Items** | `dict.items()` | View of all key-value pairs |
| **Copy** | `dict.copy()` | Create shallow copy |
| **Clear** | `dict.clear()` | Remove all items |
| **Length** | `len(dict)` | Number of key-value pairs |

**Example combining multiple operations:**
```python
# Create and populate dictionary
inventory = {"apples": 50, "bananas": 30, "oranges": 25}

# Update and add items
inventory.update({"grapes": 40, "apples": 60})  # Update apples, add grapes
inventory["mangoes"] = 15  # Add mangoes

# Check and remove items
if "bananas" in inventory:
    sold_bananas = inventory.pop("bananas")  # Remove and get value
    print(f"Sold {sold_bananas} bananas")

# Display current inventory
print(f"Current inventory has {len(inventory)} items:")
for item, quantity in inventory.items():
    print(f"  {item}: {quantity}")
```

This comprehensive guide covers all the essential dictionary operations and methods with detailed explanations, practical examples, and best practices for effective dictionary usage in Python!