# Loops in Python

## What are Loops?

**Loops** are programming constructs that allow you to repeat a block of code multiple times. Python has two main types of loops: `for` loops (for iterating over sequences) and `while` loops (for repeating while a condition is true). Loops help avoid code repetition and make programs more efficient.



## For Loops

A `for` loop iterates over a sequence (list, tuple, string, etc.) or other iterable objects, executing a block of code for each element.

### Basic For Loop Syntax

```python
# Basic structure
for variable in sequence:
    # code to execute for each item
    pass
```

### Iterating Over Different Data Types

**Iterating over lists:**
```python
fruits = ["apple", "banana", "orange"]
for fruit in fruits:
    print(fruit)
# Output:
# apple
# banana
# orange

numbers = [1, 2, 3, 4, 5]
for number in numbers:
    result = number * 2
    print(f"{number} * 2 = {result}")
```
The loop variable (`fruit`, `number`) takes the value of each element in the list, one at a time, from left to right.

**Iterating over strings:**
```python
word = "Python"
for letter in word:
    print(letter)
# Output: P, y, t, h, o, n (each on new line)

# Count vowels in a string
text = "Hello World"
vowel_count = 0
for char in text:
    if char.lower() in "aeiou":
        vowel_count += 1
print(f"Number of vowels: {vowel_count}")  # Output: 3
```
Strings are iterable, so you can loop through each character. The loop processes each character individually.

**Iterating over tuples:**
```python
coordinates = (10, 20, 30)
for coordinate in coordinates:
    print(f"Coordinate: {coordinate}")

# Multiple tuples
points = [(1, 2), (3, 4), (5, 6)]
for point in points:
    print(f"Point: {point}")
    print(f"X: {point[0]}, Y: {point[1]}")
```
Tuples work the same way as lists in for loops. Each element becomes the loop variable's value.

**Iterating over dictionaries:**
```python
student = {"name": "Alice", "age": 20, "grade": "A"}

# Iterate over keys (default behavior)
for key in student:
    print(f"{key}: {student[key]}")

# Explicitly iterate over keys
for key in student.keys():
    print(f"Key: {key}")

# Iterate over values
for value in student.values():
    print(f"Value: {value}")

# Iterate over key-value pairs
for key, value in student.items():
    print(f"{key} = {value}")
```
By default, iterating over a dictionary loops through its keys. Use `.keys()`, `.values()`, or `.items()` for specific iteration patterns.

### Using range() Function

**Basic range usage:**
```python
# range(stop) - numbers from 0 to stop-1
for i in range(5):
    print(i)
# Output: 0, 1, 2, 3, 4

# range(start, stop) - numbers from start to stop-1
for i in range(2, 7):
    print(i)
# Output: 2, 3, 4, 5, 6

# range(start, stop, step) - with custom step
for i in range(0, 10, 2):
    print(i)
# Output: 0, 2, 4, 6, 8
```
`range()` generates a sequence of numbers. It's memory-efficient because it generates numbers on-demand rather than creating a list.

**Practical range examples:**
```python
# Count down
for i in range(10, 0, -1):
    print(f"Countdown: {i}")
print("Blast off!")

# Create multiplication table
number = 5
for i in range(1, 11):
    print(f"{number} x {i} = {number * i}")

# Generate even numbers
even_numbers = []
for i in range(0, 21, 2):
    even_numbers.append(i)
print(even_numbers)  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
```
Negative step values count backwards. You can use range with any arithmetic progression.

### Enumerate() Function

**Getting index and value:**
```python
fruits = ["apple", "banana", "orange"]
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
# Output:
# 0: apple
# 1: banana
# 2: orange

# Start enumeration from different number
for index, fruit in enumerate(fruits, start=1):
    print(f"#{index}: {fruit}")
# Output:
# #1: apple
# #2: banana
# #3: orange
```
`enumerate()` returns pairs of (index, value) for each element. The `start` parameter lets you begin counting from a different number.

**Practical enumerate examples:**
```python
# Find position of specific items
names = ["Alice", "Bob", "Charlie", "Diana"]
for index, name in enumerate(names):
    if name == "Charlie":
        print(f"Charlie is at position {index}")

# Create numbered list
tasks = ["Buy groceries", "Walk the dog", "Study Python"]
for number, task in enumerate(tasks, start=1):
    print(f"{number}. {task}")
```
`enumerate()` is useful when you need both the position and value of elements.

### Zip() Function

**Combining multiple sequences:**
```python
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
cities = ["New York", "London", "Tokyo"]

for name, age, city in zip(names, ages, cities):
    print(f"{name} is {age} years old and lives in {city}")
# Output:
# Alice is 25 years old and lives in New York
# Bob is 30 years old and lives in London
# Charlie is 35 years old and lives in Tokyo
```
`zip()` combines multiple iterables element by element. It stops when the shortest iterable is exhausted.

**Handling different lengths:**
```python
list1 = [1, 2, 3, 4, 5]
list2 = ["a", "b", "c"]

for number, letter in zip(list1, list2):
    print(f"{number}: {letter}")
# Output: 1: a, 2: b, 3: c (stops at shortest list)

# Create dictionary from two lists
keys = ["name", "age", "city"]
values = ["Alice", 25, "New York"]
person_dict = {}
for key, value in zip(keys, values):
    person_dict[key] = value
print(person_dict)  # {"name": "Alice", "age": 25, "city": "New York"}
```
`zip()` stops at the shortest sequence. It's commonly used to create dictionaries or process parallel data.



## While Loops

A `while` loop repeats a block of code as long as a specified condition is true. It's useful when you don't know exactly how many iterations you need.

### Basic While Loop Syntax

```python
# Basic structure
while condition:
    # code to execute while condition is True
    pass
```

### Simple While Loop Examples

**Basic counting:**
```python
count = 0
while count < 5:
    print(f"Count: {count}")
    count += 1  # Important: increment to avoid infinite loop
# Output: Count: 0, Count: 1, Count: 2, Count: 3, Count: 4

# Count down
count = 5
while count > 0:
    print(f"Countdown: {count}")
    count -= 1
print("Done!")
```
The condition is checked before each iteration. Always ensure the condition can become false to avoid infinite loops.

**User input validation:**
```python
# Keep asking until valid input
age = -1
while age < 0 or age > 150:
    age = int(input("Enter your age (0-150): "))
    if age < 0 or age > 150:
        print("Invalid age. Please try again.")
print(f"Your age is {age}")

# Password verification
password = ""
while password != "secret123":
    password = input("Enter password: ")
    if password != "secret123":
        print("Incorrect password. Try again.")
print("Access granted!")
```
While loops are excellent for input validation because they continue until the user provides acceptable input.

### While Loop with Calculations

**Sum until condition:**
```python
# Sum numbers until total exceeds 100
total = 0
number = 1
while total <= 100:
    total += number
    print(f"Added {number}, total is now {total}")
    number += 1
print(f"Final total: {total}")

# Find first power of 2 greater than 1000
power = 1
exponent = 0
while power <= 1000:
    power *= 2
    exponent += 1
print(f"2^{exponent} = {power} (first power of 2 > 1000)")
```
While loops are useful for calculations where you continue until reaching a target value.

### While True with Break

**Infinite loop with controlled exit:**
```python
while True:
    user_input = input("Enter 'quit' to exit: ")
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    print(f"You entered: {user_input}")

# Menu system
while True:
    print("\nMenu:")
    print("1. Say Hello")
    print("2. Show Time")
    print("3. Quit")
    
    choice = input("Choose an option (1-3): ")
    
    if choice == "1":
        print("Hello there!")
    elif choice == "2":
        import datetime
        print(f"Current time: {datetime.datetime.now()}")
    elif choice == "3":
        print("Exiting...")
        break
    else:
        print("Invalid choice. Please try again.")
```
`while True` creates an infinite loop, but `break` provides controlled exit points. This pattern is common for menu systems.



## Loop Control Statements

### Break Statement

**Exit loop early:**
```python
# Find first even number
numbers = [1, 3, 7, 8, 9, 10]
for number in numbers:
    if number % 2 == 0:
        print(f"First even number: {number}")
        break
    print(f"Checking {number} - odd")

# Search in nested structure
students = [
    {"name": "Alice", "grade": "A"},
    {"name": "Bob", "grade": "B"},
    {"name": "Charlie", "grade": "A"}
]

for student in students:
    if student["grade"] == "A":
        print(f"Found A-grade student: {student['name']}")
        break
```
`break` immediately exits the loop, skipping any remaining iterations. Only the innermost loop is exited in nested loops.

**Break in while loops:**
```python
# Guessing game
secret_number = 7
attempts = 0
max_attempts = 3

while attempts < max_attempts:
    guess = int(input("Guess the number (1-10): "))
    attempts += 1
    
    if guess == secret_number:
        print(f"Correct! You found it in {attempts} attempts.")
        break
    elif guess < secret_number:
        print("Too low!")
    else:
        print("Too high!")
else:
    print(f"Sorry! The number was {secret_number}")
```
In while loops, `break` stops the repetition immediately. The `else` clause runs if the loop completes without breaking.

### Continue Statement

**Skip current iteration:**
```python
# Print only positive numbers
numbers = [-2, -1, 0, 1, 2, 3, 4, 5]
for number in numbers:
    if number <= 0:
        continue  # Skip to next iteration
    print(f"Positive number: {number}")
# Output: Positive number: 1, 2, 3, 4, 5

# Process only valid data
data = ["abc", "", "def", None, "ghi", "   "]
for item in data:
    if not item or not item.strip():  # Skip empty or whitespace-only items
        continue
    print(f"Processing: {item}")
```
`continue` skips the rest of the current iteration and moves to the next one. It's useful for filtering data during processing.

**Continue in while loops:**
```python
# Count only even numbers
count = 0
number = 0
while number < 10:
    number += 1
    if number % 2 != 0:  # Skip odd numbers
        continue
    count += 1
    print(f"Even number: {number}")
print(f"Found {count} even numbers")
```
In while loops, `continue` jumps back to the condition check, potentially creating infinite loops if not handled carefully.

### Else Clause in Loops

**Else with for loops:**
```python
# Search for item
items = ["apple", "banana", "orange"]
search_item = "grape"

for item in items:
    if item == search_item:
        print(f"Found {search_item}!")
        break
else:
    print(f"{search_item} not found in the list")

# Check if all numbers are positive
numbers = [1, 2, 3, 4, 5]
for number in numbers:
    if number <= 0:
        print("Found non-positive number")
        break
else:
    print("All numbers are positive")
```
The `else` clause runs only if the loop completes naturally (without `break`). It's useful for "search and not found" scenarios.

**Else with while loops:**
```python
# Find factor
number = 17
divisor = 2
while divisor < number:
    if number % divisor == 0:
        print(f"{number} is divisible by {divisor}")
        break
    divisor += 1
else:
    print(f"{number} is a prime number")
```
With while loops, `else` runs if the condition becomes false naturally, not through `break`.



## Nested Loops

### Basic Nested Loops

**Loop within loop:**
```python
# Multiplication table
for i in range(1, 6):  # Outer loop
    for j in range(1, 6):  # Inner loop
        product = i * j
        print(f"{i} x {j} = {product}")
    print()  # Empty line after each table

# Create 2D pattern
rows = 4
for i in range(rows):
    for j in range(i + 1):
        print("*", end="")
    print()  # New line after each row
# Output:
# *
# **
# ***
# ****
```
The inner loop runs completely for each iteration of the outer loop. Total iterations = outer iterations × inner iterations.

### Working with 2D Data

**Process matrix:**
```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Print matrix with formatting
for row in matrix:
    for element in row:
        print(f"{element:3}", end="")  # Format with width 3
    print()  # New line after each row

# Find maximum element
max_value = 0
max_position = (0, 0)
for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        if matrix[i][j] > max_value:
            max_value = matrix[i][j]
            max_position = (i, j)
print(f"Maximum value {max_value} at position {max_position}")
```
Nested loops are essential for processing 2D data structures like matrices, tables, or grids.

### Breaking from Nested Loops

**Control nested loop flow:**
```python
# Find first occurrence in 2D structure
data = [
    ["apple", "banana"],
    ["orange", "grape"],
    ["kiwi", "mango"]
]

search_item = "grape"
found = False

for i, row in enumerate(data):
    for j, item in enumerate(row):
        if item == search_item:
            print(f"Found '{search_item}' at row {i}, column {j}")
            found = True
            break  # Exit inner loop
    if found:
        break  # Exit outer loop

# Alternative: using function with return
def find_item(data, search_item):
    for i, row in enumerate(data):
        for j, item in enumerate(row):
            if item == search_item:
                return (i, j)
    return None

position = find_item(data, "grape")
if position:
    print(f"Found at {position}")
else:
    print("Not found")
```
`break` only exits the innermost loop. To exit multiple levels, use a flag variable or put the loops in a function and use `return`.



## Loop Performance and Best Practices

### Performance Considerations

**Efficient iteration:**
```python
# Slow: repeated list access
items = ["a", "b", "c", "d", "e"] * 1000
for i in range(len(items)):
    print(items[i])  # Slower: index lookup each time

# Fast: direct iteration
for item in items:
    print(item)  # Faster: direct access

# Avoid creating unnecessary lists
# Slow: creates entire list in memory
for i in list(range(1000000)):
    pass

# Fast: generates numbers on demand
for i in range(1000000):
    pass
```
Direct iteration over collections is faster than index-based access. Use generators (like `range()`) instead of creating large lists in memory.

**Loop optimization tips:**
```python
# Move constant calculations outside loops
numbers = [1, 2, 3, 4, 5]
multiplier = 10

# Inefficient: repeated calculation
for number in numbers:
    result = number * (5 + 5)  # 5 + 5 calculated each time

# Efficient: calculate once
constant = 5 + 5
for number in numbers:
    result = number * constant

# Use list comprehensions for simple operations
# Instead of:
squares = []
for x in range(10):
    squares.append(x ** 2)

# Use:
squares = [x ** 2 for x in range(10)]
```
Avoid repeated calculations inside loops and use list comprehensions for simple transformations.

### Memory Considerations

```python
# Memory-efficient iteration
def process_large_file():
    # Instead of loading entire file
    # lines = open("large_file.txt").readlines()  # Memory intensive
    
    # Process line by line
    with open("large_file.txt") as file:
        for line in file:  # Memory efficient
            process_line(line.strip())

def process_line(line):
    # Process individual line
    pass

# Use generators for large datasets
def fibonacci_generator(n):
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

# Memory efficient: generates one number at a time
for fib_number in fibonacci_generator(100):
    print(fib_number)
```
For large datasets, use generators or process data in chunks to avoid memory issues.



## Common Loop Patterns

### Accumulator Patterns

**Sum and count:**
```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Sum all numbers
total = 0
for number in numbers:
    total += number
print(f"Sum: {total}")

# Count even numbers
even_count = 0
for number in numbers:
    if number % 2 == 0:
        even_count += 1
print(f"Even count: {even_count}")

# Find maximum
maximum = numbers[0]  # Initialize with first element
for number in numbers[1:]:  # Start from second element
    if number > maximum:
        maximum = number
print(f"Maximum: {maximum}")
```
Accumulator patterns build up a result over multiple iterations. Initialize the accumulator before the loop.

### String Building

```python
# Build string from list
words = ["Python", "is", "awesome"]
sentence = ""
for word in words:
    sentence += word + " "
sentence = sentence.strip()  # Remove trailing space
print(sentence)  # "Python is awesome"

# More efficient for large strings: use join
sentence = " ".join(words)
print(sentence)

# Build formatted output
students = [("Alice", 85), ("Bob", 92), ("Charlie", 78)]
report = ""
for name, score in students:
    report += f"{name}: {score}%\n"
print(report)
```
String concatenation in loops can be inefficient for large strings. Use `join()` for better performance.

### Filtering and Transformation

```python
# Filter data
ages = [15, 22, 17, 35, 12, 28, 19]
adults = []
for age in ages:
    if age >= 18:
        adults.append(age)
print(f"Adults: {adults}")

# Transform data
celsius_temps = [0, 20, 30, 100]
fahrenheit_temps = []
for celsius in celsius_temps:
    fahrenheit = celsius * 9/5 + 32
    fahrenheit_temps.append(fahrenheit)
print(f"Fahrenheit: {fahrenheit_temps}")

# Combine filter and transform
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_squares = []
for number in numbers:
    if number % 2 == 0:  # Filter: only even numbers
        even_squares.append(number ** 2)  # Transform: square them
print(f"Even squares: {even_squares}")
```
Loops are commonly used to filter data (keep only certain elements) and transform data (modify elements).



## Error Handling in Loops

### Common Loop Errors

**Index errors:**
```python
numbers = [1, 2, 3, 4, 5]

# Dangerous: index might go out of bounds
for i in range(len(numbers) + 1):  # Goes one too far!
    try:
        print(numbers[i])
    except IndexError:
        print(f"Index {i} is out of bounds")

# Safe: direct iteration
for number in numbers:
    print(number)
```
Index-based loops can cause IndexError. Direct iteration is safer and more readable.

**Infinite loops:**
```python
# Dangerous: infinite loop
count = 0
while count < 10:
    print(count)
    # Forgot to increment count! This will run forever
    # count += 1  # Uncomment to fix

# Safe: always ensure condition can become false
count = 0
while count < 10:
    print(count)
    count += 1  # Essential for termination
```
Always ensure while loop conditions can become false. Include increment/decrement statements where needed.

### Error Handling Inside Loops

```python
# Handle errors gracefully
data = ["1", "2", "abc", "4", "def", "6"]
valid_numbers = []

for item in data:
    try:
        number = int(item)
        valid_numbers.append(number)
    except ValueError:
        print(f"Skipping invalid number: {item}")

print(f"Valid numbers: {valid_numbers}")

# Continue processing despite errors
file_names = ["file1.txt", "missing.txt", "file3.txt"]
for file_name in file_names:
    try:
        with open(file_name, 'r') as file:
            content = file.read()
            print(f"Processed {file_name}")
    except FileNotFoundError:
        print(f"File not found: {file_name}")
        continue  # Continue with next file
```
Use try-except blocks inside loops to handle errors gracefully without stopping the entire loop.



## Best Practices and Guidelines

### When to Use Each Loop Type

**Use for loops when:**
- You know the exact number of iterations
- You're iterating over a collection (list, tuple, string, etc.)
- You need the index and value (`enumerate()`)
- You're processing multiple sequences together (`zip()`)

**Use while loops when:**
- The number of iterations is unknown
- You're waiting for a condition to be met
- You're implementing game loops or interactive programs
- You're processing user input until a specific condition

### Loop Best Practices

1. **Choose the right loop type:**
```python
# Good: for loop for known sequence
for i in range(10):
    print(i)

# Avoid: while loop for known iterations
i = 0
while i < 10:
    print(i)
    i += 1
```

2. **Use descriptive variable names:**
```python
# Good: descriptive names
for student in students:
    print(student.name)

for row in matrix:
    for column in row:
        process_cell(column)

# Avoid: unclear names
for x in students:
    print(x.name)

for i in matrix:
    for j in i:
        process_cell(j)
```

3. **Keep loops simple:**
```python
# Good: simple, focused loop
for number in numbers:
    if number > 0:
        positive_numbers.append(number)

# Avoid: complex nested conditions
for number in numbers:
    if number > 0:
        if number % 2 == 0:
            if number < 100:
                # Complex nested logic
                pass
```

4. **Use loop control statements appropriately:**
```python
# Good: clear exit condition
for item in items:
    if item == target:
        found = True
        break

# Avoid: unnecessary flags
found = False
for item in items:
    if item == target and not found:
        found = True
```



## Quick Reference Summary

| Loop Type | Syntax | Use Case |
|-----------|--------|----------|
| **For loop** | `for item in sequence:` | Iterate over collections |
| **For with range** | `for i in range(n):` | Repeat n times |
| **For with enumerate** | `for i, item in enumerate(seq):` | Need index and value |
| **For with zip** | `for a, b in zip(seq1, seq2):` | Parallel iteration |
| **While loop** | `while condition:` | Repeat while condition true |
| **Break** | `break` | Exit loop immediately |
| **Continue** | `continue` | Skip to next iteration |
| **Else** | `for/while: ... else:` | Execute if no break occurred |

**Example combining multiple concepts:**
```python
# Comprehensive example: Student grade processor
students = [
    {"name": "Alice", "scores": [85, 92, 78, 96]},
    {"name": "Bob", "scores": [88, 76, 90, 85]},
    {"name": "Charlie", "scores": [92, 88, 94, 90]}
]

print("=== STUDENT GRADE REPORT ===\n")

for student_num, student in enumerate(students, 1):
    print(f"Student #{student_num}: {student['name']}")
    
    # Calculate average
    total = 0
    valid_scores = 0
    
    for score_num, score in enumerate(student['scores'], 1):
        if score < 0 or score > 100:  # Skip invalid scores
            print(f"  ⚠️  Invalid score #{score_num}: {score}")
            continue
        
        total += score
        valid_scores += 1
        print(f"  Test {score_num}: {score}%")
    
    if valid_scores == 0:
        print("  ❌ No valid scores found")
        continue
    
    average = total / valid_scores
    
    # Determine letter grade
    if average >= 90:
        letter_grade = "A"
        emoji = "🌟"
    elif average >= 80:
        letter_grade = "B"
        emoji = "👍"
    elif average >= 70:
        letter_grade = "C"
        emoji = "👌"
    elif average >= 60:
        letter_grade = "D"
        emoji = "⚠️"
    else:
        letter_grade = "F"
        emoji = "❌"
    
    print(f"  📊 Average: {average:.1f}% (Grade: {letter_grade}) {emoji}")
    print()

print("Report complete!")
```

This comprehensive guide covers all essential loop concepts with detailed explanations, practical examples, performance considerations, and best practices for effective loop usage in Python!