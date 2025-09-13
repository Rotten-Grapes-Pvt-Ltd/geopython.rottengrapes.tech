# Sets and Tuples


## Sets

A **set** is a collection of unique elements. Sets automatically remove duplicates and are unordered, meaning elements don't have a fixed position.

### Creating Sets

```python
# Empty set - must use set() function, not {} (which creates empty dict)
my_set = set()

# Set with elements - use curly braces with comma-separated values
fruits = {"apple", "banana", "orange"}
numbers = {1, 2, 3, 4, 5}

# Creating set from a list - automatically removes duplicates
my_set = set([1, 2, 2, 3, 3, 4])  # Result: {1, 2, 3, 4}
```

 Sets only store unique values. If you try to create a set with duplicate elements, Python automatically keeps only one copy of each element.

### Basic Set Methods

#### Adding Elements

**add() method:**
```python
fruits = {"apple", "banana"}
fruits.add("orange")        # Adds single element to the set
print(fruits)               # {"apple", "banana", "orange"}
```
 The `add()` method adds exactly one element to the set. If the element already exists, nothing happens (no error, no duplicate).

**update() method:**
```python
fruits = {"apple", "banana"}
fruits.update(["grape", "mango"])    # Add multiple elements from a list
fruits.update({"kiwi", "peach"})     # Add multiple elements from another set
fruits.update("hello")               # Adds each character: 'h', 'e', 'l', 'o'
```
 The `update()` method can add multiple elements at once. It accepts any iterable (list, set, string, etc.). When you pass a string, each character becomes a separate element.

#### Removing Elements

**remove() method:**
```python
fruits = {"apple", "banana", "orange"}
fruits.remove("banana")     # Removes "banana" from the set
# fruits.remove("grape")    # This would cause KeyError since "grape" doesn't exist
```
 The `remove()` method deletes the specified element from the set. If the element doesn't exist, Python raises a KeyError.

**discard() method:**
```python
fruits = {"apple", "banana", "orange"}
fruits.discard("banana")    # Removes "banana" from the set
fruits.discard("grape")     # No error even though "grape" doesn't exist
```
 The `discard()` method is safer than `remove()` because it won't raise an error if the element doesn't exist. It simply does nothing.

**pop() method:**
```python
fruits = {"apple", "banana", "orange"}
removed_fruit = fruits.pop()    # Removes and returns a random element
print(removed_fruit)            # Could be any of the fruits
print(fruits)                   # Now missing one element
```
 The `pop()` method removes and returns an arbitrary element from the set. Since sets are unordered, you can't predict which element will be removed. If the set is empty, it raises a KeyError.

**clear() method:**
```python
fruits = {"apple", "banana", "orange"}
fruits.clear()              # Removes all elements
print(fruits)               # set() - empty set
```
 The `clear()` method removes every element from the set, leaving it completely empty.

#### Set Operations

**Union - combining sets:**
```python
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Using union() method
union_set = set1.union(set2)        # Creates new set: {1, 2, 3, 4, 5, 6}
union_set = set1 | set2             # Same result using | operator
```
 Union combines all unique elements from both sets. Elements that appear in both sets are only included once in the result.

**Intersection - finding common elements:**
```python
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

common = set1.intersection(set2)    # Creates new set: {3, 4}
common = set1 & set2                # Same result using & operator
```
 Intersection returns only the elements that exist in both sets. If there are no common elements, you get an empty set.

**Difference - elements in first set but not second:**
```python
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

diff = set1.difference(set2)        # Creates new set: {1, 2}
diff = set1 - set2                  # Same result using - operator
```
 Difference returns elements that are in the first set but not in the second set. The order matters: `set1 - set2` gives different results than `set2 - set1`.

#### Checking Elements and Properties

**Membership testing:**
```python
fruits = {"apple", "banana", "orange"}

print("apple" in fruits)     # True - "apple" exists in the set
print("grape" in fruits)     # False - "grape" doesn't exist
print("apple" not in fruits) # False - opposite of "in"
```
 The `in` operator checks if an element exists in the set. It returns `True` if found, `False` if not found. This operation is very fast with sets.

**Length and set relationships:**
```python
fruits = {"apple", "banana", "orange"}
small_set = {"apple", "banana"}
big_set = {"apple", "banana", "orange", "grape", "mango"}

print(len(fruits))                      # 3 - number of elements
print(small_set.issubset(fruits))       # True - all elements of small_set are in fruits
print(fruits.issuperset(small_set))     # True - fruits contains all elements of small_set
print(fruits.isdisjoint({1, 2, 3}))     # True - no common elements
```
 
- `len()` returns the number of elements in the set
- `issubset()` checks if all elements of the current set exist in another set
- `issuperset()` checks if the current set contains all elements of another set
- `isdisjoint()` returns `True` if the sets have no common elements

---

## Tuples

A **tuple** is an ordered collection of elements that cannot be changed after creation (immutable). Unlike sets, tuples allow duplicate elements and maintain the order of elements.

### Creating Tuples

```python
# Empty tuple - use empty parentheses
empty_tuple = ()

# Tuple with multiple elements
coordinates = (3, 5)
colors = ("red", "green", "blue")
mixed = (1, "hello", 3.14, True)

# Single element tuple - MUST include comma!
single = (42,)      # Without comma, (42) is just a number in parentheses
also_single = 42,   # Comma makes it a tuple even without parentheses
```

 Tuples use parentheses `()` but the comma is what actually makes it a tuple. For a single-element tuple, the comma is essential because `(42)` is just a number in parentheses, but `(42,)` is a tuple with one element.

### Basic Tuple Methods

#### Accessing Elements

**Indexing:**
```python
colors = ("red", "green", "blue", "yellow")
print(colors[0])        # "red" - first element (index starts at 0)
print(colors[1])        # "green" - second element  
print(colors[-1])       # "yellow" - last element (negative indexing)
print(colors[-2])       # "blue" - second to last element
```
 Tuple indexing works like lists. Positive indices start from 0 at the beginning, negative indices start from -1 at the end.

**Slicing:**
```python
colors = ("red", "green", "blue", "yellow", "purple")
print(colors[1:3])      # ("green", "blue") - elements from index 1 to 2
print(colors[:2])       # ("red", "green") - first two elements
print(colors[2:])       # ("blue", "yellow", "purple") - from index 2 to end
print(colors[::2])      # ("red", "blue", "purple") - every second element
```
 Slicing creates a new tuple with selected elements. The syntax is `[start:stop:step]` where `stop` is not included in the result.

#### Built-in Tuple Methods

**count() method:**
```python
numbers = (1, 2, 3, 2, 4, 2, 5)
count_of_2 = numbers.count(2)      # Returns 3 (appears 3 times)
count_of_7 = numbers.count(7)      # Returns 0 (doesn't appear)
```
 The `count()` method returns how many times a specific element appears in the tuple. If the element doesn't exist, it returns 0.

**index() method:**
```python
numbers = (1, 2, 3, 2, 4, 2, 5)
first_2_index = numbers.index(2)      # Returns 1 (first occurrence at index 1)
# numbers.index(7)                    # Would raise ValueError since 7 doesn't exist

# Find index starting from a specific position
later_2_index = numbers.index(2, 2)   # Returns 3 (first occurrence at/after index 2)
```
 The `index()` method returns the position of the first occurrence of an element. If the element doesn't exist, it raises a ValueError. You can optionally specify where to start searching.

#### Tuple Operations

**Concatenation:**
```python
tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)
tuple3 = ("a", "b")

combined = tuple1 + tuple2      # (1, 2, 3, 4, 5, 6) - creates new tuple
long_tuple = tuple1 + tuple2 + tuple3  # (1, 2, 3, 4, 5, 6, 'a', 'b')
```
 The `+` operator creates a new tuple by combining elements from multiple tuples in order. The original tuples remain unchanged.

**Repetition:**
```python
base_tuple = (1, 2, 3)
repeated = base_tuple * 3       # (1, 2, 3, 1, 2, 3, 1, 2, 3)
empty_repeat = base_tuple * 0   # () - empty tuple
```
 The `*` operator creates a new tuple by repeating the original tuple a specified number of times.

**Membership testing:**
```python
coordinates = (10, 20, 30)
print(20 in coordinates)        # True - 20 exists in the tuple
print(40 in coordinates)        # False - 40 doesn't exist
print(20 not in coordinates)    # False - opposite of "in"
```
 The `in` operator works the same way as with sets, checking if an element exists in the tuple.

**Length:**
```python
data = (1, "hello", 3.14, True, [1, 2, 3])
print(len(data))              # 5 - counts number of elements
```
 The `len()` function returns the number of elements in the tuple, regardless of their types.

#### Unpacking Tuples

**Basic unpacking:**
```python
point = (10, 20)
x, y = point                    # x gets 10, y gets 20

person = ("Alice", 25, "Engineer")
name, age, job = person         # Each variable gets corresponding element
```
 Unpacking assigns each element of the tuple to separate variables. The number of variables must match the number of elements in the tuple.

**Advanced unpacking with * operator:**
```python
data = (1, 2, 3, 4, 5)
first, *middle, last = data     # first=1, middle=[2, 3, 4], last=5
first, *rest = data             # first=1, rest=[2, 3, 4, 5]
*beginning, last = data         # beginning=[1, 2, 3, 4], last=5
```
 The `*` operator collects multiple elements into a list. This allows flexible unpacking when you don't know exactly how many elements you'll have.

---

## Key Differences

| Feature | Set | Tuple |
|---------|-----|--------|
| **Ordered** | No - elements have no fixed position | Yes - elements keep their position |
| **Duplicates** | No - automatically removes duplicates | Yes - can store identical elements |
| **Mutable** | Yes - can add/remove elements | No - cannot change after creation |
| **Indexing** | No - cannot access by position | Yes - can access elements by index |
| **Use Case** | Unique collections, math operations | Coordinates, fixed data, multiple return values |
| **Performance** | Fast membership testing | Fast access by index |

---

## Common Use Cases

### Sets
**Removing duplicates:** When you have data with repeated values and only want unique items.
```python
grades = [85, 92, 78, 92, 88, 85, 95]
unique_grades = set(grades)     # {78, 85, 88, 92, 95}
```

**Mathematical operations:** When you need to find common elements, differences, or combine collections.
```python
students_math = {"Alice", "Bob", "Charlie", "Diana"}
students_science = {"Bob", "Diana", "Eve", "Frank"}
both_subjects = students_math.intersection(students_science)  # {"Bob", "Diana"}
```

**Fast membership testing:** When you frequently need to check if an element exists.
```python
valid_codes = {"A1", "B2", "C3", "D4"}
user_input = "B2"
if user_input in valid_codes:  # Very fast operation
    print("Valid code!")
```

### Tuples
**Coordinates and paired data:** When elements belong together and shouldn't change.
```python
location = (40.7128, -74.0060)  # Latitude, longitude for New York
rgb_color = (255, 128, 0)       # Red, green, blue values
```

**Multiple return values:** When a function needs to return several related values.
```python
def get_name_age():
    return ("John", 25)         # Return multiple values as tuple

name, age = get_name_age()      # Unpack the returned tuple
```

**Dictionary keys:** Since tuples are immutable, they can be used as dictionary keys.
```python
coordinates_to_city = {
    (40.7128, -74.0060): "New York",
    (34.0522, -118.2437): "Los Angeles"
}
```

**Configuration data:** When you have settings that shouldn't accidentally change.
```python
database_config = ("localhost", 5432, "mydb", "user123")
host, port, database, username = database_config
```