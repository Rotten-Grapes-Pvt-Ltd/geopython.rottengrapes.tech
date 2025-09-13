# Lists

## What is a List?

A **list** is an ordered collection of elements that can be changed (mutable). Lists allow duplicate elements, maintain insertion order, and can store different data types. Lists use square brackets `[]` and elements are separated by commas.

### Creating Lists

```python
# Empty list
empty_list = []
also_empty = list()

# List with elements
numbers = [1, 2, 3, 4, 5]
fruits = ["apple", "banana", "orange"]
mixed = [1, "hello", 3.14, True, [1, 2, 3]]  # Different data types including nested list

# Creating list from other iterables
string_list = list("hello")         # ['h', 'e', 'l', 'l', 'o']
range_list = list(range(5))         # [0, 1, 2, 3, 4]
tuple_list = list((1, 2, 3))        # [1, 2, 3] - convert tuple to list

# List with repeated elements
zeros = [0] * 5                     # [0, 0, 0, 0, 0]
repeated = ["item"] * 3             # ["item", "item", "item"]
```

Lists are versatile containers that can hold any type of data, including other lists. The `*` operator creates a list with repeated elements, and the `list()` function can convert other iterable objects into lists.



## Accessing List Elements

### Indexing

**Basic indexing:**
```python
fruits = ["apple", "banana", "orange", "grape", "mango"]
print(fruits[0])        # "apple" - first element (index starts at 0)
print(fruits[1])        # "banana" - second element
print(fruits[4])        # "mango" - fifth element
# print(fruits[5])      # IndexError - index out of range
```
List indexing starts at 0 for the first element. Trying to access an index that doesn't exist raises an IndexError.

**Negative indexing:**
```python
fruits = ["apple", "banana", "orange", "grape", "mango"]
print(fruits[-1])       # "mango" - last element
print(fruits[-2])       # "grape" - second to last element
print(fruits[-5])       # "apple" - first element (same as fruits[0])
# print(fruits[-6])     # IndexError - negative index too large
```
Negative indices count from the end of the list. `-1` is the last element, `-2` is second to last, and so on.

### Slicing

**Basic slicing:**
```python
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(numbers[2:5])     # [2, 3, 4] - elements from index 2 to 4 (5 not included)
print(numbers[:3])      # [0, 1, 2] - first three elements
print(numbers[3:])      # [3, 4, 5, 6, 7, 8, 9] - from index 3 to end
print(numbers[:])       # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] - entire list (copy)
```
Slicing creates a new list with selected elements. The syntax is `[start:stop]` where `stop` is not included. Omitting `start` means from beginning, omitting `stop` means to end.

**Advanced slicing with step:**
```python
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(numbers[::2])     # [0, 2, 4, 6, 8] - every second element
print(numbers[1::2])    # [1, 3, 5, 7, 9] - every second element starting from index 1
print(numbers[::-1])    # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] - reverse the list
print(numbers[2:8:2])   # [2, 4, 6] - from index 2 to 7, every second element
```
The full slicing syntax is `[start:stop:step]`. A step of 2 takes every second element, and a negative step reverses the direction.



## Modifying Lists

### Adding Elements

**append() method:**
```python
fruits = ["apple", "banana"]
fruits.append("orange")         # Add single element to the end
print(fruits)                   # ["apple", "banana", "orange"]

fruits.append(["grape", "mango"])  # Adds the entire list as one element
print(fruits)                   # ["apple", "banana", "orange", ["grape", "mango"]]
```
The `append()` method adds exactly one element to the end of the list. If you append a list, the entire list becomes a single element (nested list).

**extend() method:**
```python
fruits = ["apple", "banana"]
fruits.extend(["orange", "grape"])     # Add multiple elements from iterable
print(fruits)                          # ["apple", "banana", "orange", "grape"]

fruits.extend("hi")                    # Add each character as separate element
print(fruits)                          # ["apple", "banana", "orange", "grape", "h", "i"]
```
The `extend()` method adds all elements from an iterable to the end of the list. Each element is added individually, unlike `append()`.

**insert() method:**
```python
fruits = ["apple", "banana", "orange"]
fruits.insert(1, "grape")      # Insert "grape" at index 1
print(fruits)                   # ["apple", "grape", "banana", "orange"]

fruits.insert(0, "mango")      # Insert at beginning
print(fruits)                   # ["mango", "apple", "grape", "banana", "orange"]

fruits.insert(100, "kiwi")     # Insert beyond list length (adds to end)
print(fruits)                   # ["mango", "apple", "grape", "banana", "orange", "kiwi"]
```
The `insert()` method adds an element at a specific position. All elements at and after that position shift to the right. If the index is larger than the list length, it adds to the end.

**Using + operator:**
```python
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = list1 + list2        # [1, 2, 3, 4, 5, 6] - creates new list
list1 += [7, 8]                 # Modify list1 in place: [1, 2, 3, 7, 8]
```
The `+` operator creates a new list by combining existing lists. The `+=` operator modifies the original list in place (equivalent to `extend()`).

### Removing Elements

**remove() method:**
```python
fruits = ["apple", "banana", "orange", "banana", "grape"]
fruits.remove("banana")         # Removes first occurrence of "banana"
print(fruits)                   # ["apple", "orange", "banana", "grape"]
# fruits.remove("mango")        # ValueError - element not in list
```
The `remove()` method removes the first occurrence of the specified value. If the value doesn't exist, it raises a ValueError.

**pop() method:**
```python
fruits = ["apple", "banana", "orange", "grape"]
last_fruit = fruits.pop()       # Remove and return last element: "grape"
second_fruit = fruits.pop(1)    # Remove and return element at index 1: "banana"
print(fruits)                   # ["apple", "orange"]
# fruits.pop(10)                # IndexError - index out of range
```
The `pop()` method removes and returns an element. Without arguments, it removes the last element. With an index argument, it removes the element at that position.

**del statement:**
```python
fruits = ["apple", "banana", "orange", "grape", "mango"]
del fruits[1]                   # Remove element at index 1
print(fruits)                   # ["apple", "orange", "grape", "mango"]

del fruits[1:3]                 # Remove slice (elements at index 1 and 2)
print(fruits)                   # ["apple", "mango"]

del fruits[:]                   # Remove all elements (same as clear())
print(fruits)                   # []
```
The `del` statement can remove elements by index, slices, or entire variables. It's more flexible than `pop()` because it can remove multiple elements at once.

**clear() method:**
```python
fruits = ["apple", "banana", "orange"]
fruits.clear()                  # Remove all elements
print(fruits)                   # []
```
The `clear()` method removes all elements from the list, leaving an empty list.

### Modifying Elements

**Direct assignment:**
```python
fruits = ["apple", "banana", "orange"]
fruits[1] = "grape"             # Replace element at index 1
print(fruits)                   # ["apple", "grape", "orange"]

fruits[0:2] = ["mango", "kiwi", "peach"]  # Replace slice with multiple elements
print(fruits)                   # ["mango", "kiwi", "peach", "orange"]
```
You can modify list elements by assigning new values to specific indices or slices. Slice assignment can change the list length.



## List Methods for Information and Organization

### Finding Elements

**index() method:**
```python
fruits = ["apple", "banana", "orange", "banana", "grape"]
banana_index = fruits.index("banana")       # Returns 1 (first occurrence)
# mango_index = fruits.index("mango")       # ValueError - not in list

# Search within a range
later_banana = fruits.index("banana", 2)    # Returns 3 (first occurrence at/after index 2)
range_search = fruits.index("banana", 1, 4) # Search between index 1 and 3
```
The `index()` method returns the position of the first occurrence of a value. You can specify start and end positions for the search. Raises ValueError if not found.

**count() method:**
```python
numbers = [1, 2, 3, 2, 4, 2, 5, 2]
count_2 = numbers.count(2)      # Returns 4 (appears 4 times)
count_7 = numbers.count(7)      # Returns 0 (doesn't appear)

fruits = ["apple", "banana", "apple"]
count_apple = fruits.count("apple")  # Returns 2
```
The `count()` method returns how many times a specific value appears in the list. Returns 0 if the value is not found.

### Membership Testing

```python
fruits = ["apple", "banana", "orange"]
print("banana" in fruits)       # True - element exists
print("grape" in fruits)        # False - element doesn't exist
print("apple" not in fruits)    # False - opposite of "in"

# Check for sublists (doesn't work as expected)
numbers = [1, 2, 3, 4, 5]
print([2, 3] in numbers)        # False - checks for exact sublist as element
print(2 in numbers and 3 in numbers)  # True - check individual elements
```
The `in` operator checks if a value exists in the list. For sublists, you need to check each element individually as `in` looks for exact matches.

### List Length and Properties

```python
fruits = ["apple", "banana", "orange"]
print(len(fruits))              # 3 - number of elements

mixed = [1, "hello", [1, 2, 3], {"key": "value"}]
print(len(mixed))               # 4 - counts nested structures as single elements

empty = []
print(len(empty))               # 0 - empty list
```
The `len()` function returns the number of elements in the list. Nested structures (lists, dictionaries) count as single elements.



## Organizing Lists

### Sorting

**sort() method (modifies original list):**
```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
numbers.sort()                  # Sort in ascending order
print(numbers)                  # [1, 1, 2, 3, 4, 5, 6, 9]

numbers.sort(reverse=True)      # Sort in descending order
print(numbers)                  # [9, 6, 5, 4, 3, 2, 1, 1]

fruits = ["banana", "apple", "orange", "grape"]
fruits.sort()                   # Alphabetical sorting
print(fruits)                   # ["apple", "banana", "grape", "orange"]
```
The `sort()` method modifies the original list. It sorts in ascending order by default, use `reverse=True` for descending. Strings are sorted alphabetically.

**sorted() function (creates new list):**
```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_numbers = sorted(numbers)         # Create new sorted list
print(numbers)                           # [3, 1, 4, 1, 5, 9, 2, 6] - original unchanged
print(sorted_numbers)                    # [1, 1, 2, 3, 4, 5, 6, 9] - new sorted list

reverse_sorted = sorted(numbers, reverse=True)  # [9, 6, 5, 4, 3, 2, 1, 1]
```
The `sorted()` function creates a new sorted list without modifying the original. Use this when you need to keep the original order intact.

**Custom sorting with key parameter:**
```python
words = ["apple", "pie", "cherry", "a"]
words.sort(key=len)             # Sort by string length
print(words)                    # ["a", "pie", "apple", "cherry"]

students = ["Alice", "bob", "Charlie", "diana"]
students.sort(key=str.lower)    # Case-insensitive sorting
print(students)                 # ["Alice", "bob", "Charlie", "diana"]
```
The `key` parameter accepts a function that determines how to compare elements. `len` sorts by length, `str.lower` ignores case differences.

### Reversing

**reverse() method:**
```python
numbers = [1, 2, 3, 4, 5]
numbers.reverse()               # Reverse the list in place
print(numbers)                  # [5, 4, 3, 2, 1]
```
The `reverse()` method reverses the order of elements in the original list.

**Using slicing to reverse:**
```python
numbers = [1, 2, 3, 4, 5]
reversed_copy = numbers[::-1]   # Create new reversed list
print(numbers)                  # [1, 2, 3, 4, 5] - original unchanged
print(reversed_copy)            # [5, 4, 3, 2, 1] - new reversed list
```
Slicing with `[::-1]` creates a new reversed list without modifying the original.



## Copying Lists

### Shallow Copy Methods

**copy() method:**
```python
original = [1, 2, [3, 4], 5]
shallow_copy = original.copy()

shallow_copy[0] = 99            # Changes only the copy
shallow_copy[2].append(5)       # Changes both (shared nested list)

print(original)                 # [1, 2, [3, 4, 5], 5]
print(shallow_copy)             # [99, 2, [3, 4, 5], 5]
```
The `copy()` method creates a shallow copy. Changes to immutable elements only affect the copy, but changes to nested mutable objects affect both lists.

**Using slicing to copy:**
```python
original = [1, 2, 3, 4, 5]
copy_by_slice = original[:]     # Create copy using full slice
copy_by_list = list(original)   # Create copy using list() constructor
```
Both `[:]` slicing and `list()` constructor create shallow copies of the original list.



## List Comprehensions

### Basic List Comprehensions

**Creating new lists:**
```python
# Squares of numbers
squares = [x**2 for x in range(1, 6)]       # [1, 4, 9, 16, 25]

# Transform existing list
words = ["hello", "world", "python"]
uppercase = [word.upper() for word in words]  # ["HELLO", "WORLD", "PYTHON"]
lengths = [len(word) for word in words]      # [5, 5, 6]
```
List comprehensions create new lists using the syntax `[expression for item in iterable]`. They're more concise than traditional loops for creating lists.

**List comprehensions with conditions:**
```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = [x for x in numbers if x % 2 == 0]   # [2, 4, 6, 8, 10]
even_squares = [x**2 for x in numbers if x % 2 == 0]  # [4, 16, 36, 64, 100]

# Conditional expression (ternary operator)
signs = ["positive" if x > 0 else "negative" for x in [-2, -1, 0, 1, 2]]
# ["negative", "negative", "negative", "positive", "positive"]
```
You can add conditions to filter elements (`if condition` at the end) or use conditional expressions to transform values (`value_if_true if condition else value_if_false`).

**Nested list comprehensions:**
```python
# Flatten nested list
nested = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [item for sublist in nested for item in sublist]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Create multiplication table
mult_table = [[i * j for j in range(1, 4)] for i in range(1, 4)]
# [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
```
You can nest comprehensions for more complex operations. The order of `for` clauses matches the order of nested loops.



## Working with Nested Lists

### Accessing Nested Elements

```python
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(matrix[0])        # [1, 2, 3] - first row
print(matrix[0][1])     # 2 - first row, second column
print(matrix[2][0])     # 7 - third row, first column

# Modify nested elements
matrix[1][1] = 99       # Change middle element
print(matrix)           # [[1, 2, 3], [4, 99, 6], [7, 8, 9]]
```
Access nested list elements by chaining indices. The first index selects the inner list, the second index selects the element within that list.

### Operations on Nested Lists

```python
# Sum all elements in nested lists
nested_numbers = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
total = sum(sum(sublist) for sublist in nested_numbers)  # 45

# Find maximum in each sublist
max_in_each = [max(sublist) for sublist in nested_numbers]  # [3, 5, 9]

# Get all elements from nested lists
all_elements = [item for sublist in nested_numbers for item in sublist]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]
```
You can perform operations on nested lists using comprehensions and built-in functions like `sum()` and `max()`.



## List Performance and Memory

### Performance Characteristics

**Fast Operations (O(1) - constant time):**
- Access by index: `list[i]`
- Append to end: `list.append(item)`
- Get length: `len(list)`
- Pop from end: `list.pop()`

**Slow Operations (O(n) - linear time):**
- Search for item: `item in list`
- Insert at beginning: `list.insert(0, item)`
- Remove by value: `list.remove(item)`
- Pop from beginning: `list.pop(0)`

```python
# Fast: append to end
my_list = []
for i in range(1000):
    my_list.append(i)  # Fast operation

# Slow: insert at beginning
my_list = []
for i in range(1000):
    my_list.insert(0, i)  # Slow operation - shifts all elements
```
Operations at the end of lists are fast, while operations at the beginning or middle require shifting elements and are slower for large lists.

### Memory Considerations

```python
# Lists pre-allocate memory for efficiency
import sys
my_list = []
for i in range(10):
    my_list.append(i)
    print(f"Length: {len(my_list)}, Memory: {sys.getsizeof(my_list)} bytes")
```
Python lists pre-allocate extra memory space to make append operations faster. The memory size grows in chunks, not with every single element.



## Common Patterns and Use Cases

### Data Processing

**Filtering data:**
```python
numbers = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]
positives = [x for x in numbers if x > 0]        # [1, 3, 5, 7, 9]
negative_squares = [x**2 for x in numbers if x < 0]  # [4, 16, 36, 64, 100]
```

**Transforming data:**
```python
temperatures_celsius = [0, 20, 30, 100]
temperatures_fahrenheit = [c * 9/5 + 32 for c in temperatures_celsius]
# [32.0, 68.0, 86.0, 212.0]

names = ["alice", "bob", "charlie"]
formatted_names = [name.title() for name in names]  # ["Alice", "Bob", "Charlie"]
```

### Working with Multiple Lists

**Zip for parallel iteration:**
```python
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
combined = list(zip(names, ages))       # [("Alice", 25), ("Bob", 30), ("Charlie", 35)]

# Create dictionary from two lists
name_age_dict = dict(zip(names, ages))  # {"Alice": 25, "Bob": 30, "Charlie": 35}
```
The `zip()` function pairs elements from multiple lists, creating tuples of corresponding elements.

**Unpacking lists:**
```python
coordinates = [10, 20]
x, y = coordinates          # x = 10, y = 20

data = [1, 2, 3, 4, 5]
first, *middle, last = data # first = 1, middle = [2, 3, 4], last = 5
```
You can unpack list elements into separate variables. The `*` operator collects multiple elements into a new list.

### Accumulating Results

```python
# Calculate cumulative sum
numbers = [1, 2, 3, 4, 5]
cumulative = []
total = 0
for num in numbers:
    total += num
    cumulative.append(total)    # [1, 3, 6, 10, 15]

# Build list of unique elements (preserving order)
original = [1, 2, 2, 3, 1, 4, 3, 5]
unique = []
for item in original:
    if item not in unique:
        unique.append(item)     # [1, 2, 3, 4, 5]
```



## Best Practices

### When to Use Lists

**✅ Good use cases:**
- **Ordered data:** When the sequence of elements matters
- **Indexed access:** When you need to access elements by position
- **Dynamic size:** When the number of elements changes frequently
- **Homogeneous data:** When storing similar types of data
- **Stack operations:** When you need to add/remove from the end
- **Data collection:** When gathering data that will be processed later

**❌ Avoid lists when:**
- You need unique elements only (use sets)
- You frequently search for specific values (use dictionaries)
- You need key-value relationships (use dictionaries)
- The data doesn't change (consider tuples)
- You need mathematical operations on all elements (use numpy arrays for large data)

### Performance Tips

1. **Append instead of insert:** Use `append()` for adding elements when order doesn't matter
2. **Extend instead of multiple appends:** Use `extend()` to add multiple elements at once
3. **List comprehensions:** Often faster than equivalent loops for creating lists
4. **Slice assignment:** Efficient for replacing multiple elements
5. **Use appropriate data structures:** Consider sets for membership testing, dictionaries for lookups

### Memory Tips

1. **Pre-allocate when possible:** If you know the final size, consider creating the full list first
2. **Delete references:** Use `del` to remove references to large lists when done
3. **Use generators:** For large datasets that you process once, consider generators instead of lists
4. **Slice carefully:** Remember that slices create new lists and use memory



## Quick Reference Summary

| Operation | Syntax | Description |
|-----------|--------|-------------|
| **Create** | `[1, 2, 3]` or `list()` | Create list with initial data |
| **Access** | `list[index]` | Get element by position |
| **Slice** | `list[start:stop:step]` | Get multiple elements |
| **Add to end** | `list.append(item)` | Add single element |
| **Add multiple** | `list.extend(items)` | Add all elements from iterable |
| **Insert** | `list.insert(index, item)` | Add element at position |
| **Remove by value** | `list.remove(value)` | Remove first occurrence |
| **Remove by index** | `list.pop(index)` | Remove and return element |
| **Find position** | `list.index(value)` | Get index of first occurrence |
| **Count occurrences** | `list.count(value)` | Count how many times value appears |
| **Sort** | `list.sort()` | Sort list in place |
| **Reverse** | `list.reverse()` | Reverse list in place |
| **Copy** | `list.copy()` or `list[:]` | Create shallow copy |
| **Clear** | `list.clear()` | Remove all elements |
| **Length** | `len(list)` | Number of elements |
| **Check membership** | `item in list` | Test if item exists |

**Example combining multiple operations:**
```python
# Create and populate list
scores = [85, 92, 78, 96, 88]

# Add new scores
scores.append(94)                    # Add single score
scores.extend([91, 87])              # Add multiple scores

# Remove lowest score
lowest = min(scores)
scores.remove(lowest)                # Remove first occurrence of lowest

# Sort and analyze
scores.sort(reverse=True)            # Sort highest to lowest
print(f"Top 3 scores: {scores[:3]}")

# Create grade categories
high_scores = [score for score in scores if score >= 90]
average_scores = [score for score in scores if 80 <= score < 90]

print(f"High scores ({len(high_scores)}): {high_scores}")
print(f"Average scores ({len(average_scores)}): {average_scores}")
print(f"Overall average: {sum(scores) / len(scores):.1f}")
```

This comprehensive guide covers all essential list operations with detailed explanations, practical examples, performance considerations, and best practices for effective list usage in Python!