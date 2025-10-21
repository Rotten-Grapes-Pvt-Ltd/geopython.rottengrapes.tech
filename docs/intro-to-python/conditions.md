# Conditions in Python

## What are Conditions?

**Conditions** are programming constructs that allow your code to make decisions and execute different blocks of code based on whether certain conditions are true or false. Python uses `if`, `elif`, and `else` statements to implement conditional logic, enabling programs to respond dynamically to different situations.



## Basic If Statement

The `if` statement executes a block of code only when a specified condition evaluates to `True`.

### Basic If Syntax

```python
# Basic structure
if condition:
    # code to execute if condition is True
    pass
```

### Simple If Examples

**Basic comparisons:**
```python
age = 18
if age >= 18:
    print("You are an adult")

score = 85
if score >= 90:
    print("Excellent score!")

temperature = 30
if temperature > 25:
    print("It's a warm day")
    print("Consider wearing light clothes")
```
 The condition after `if` must evaluate to a boolean value (`True` or `False`). If `True`, the indented code block executes. Indentation is crucial in Python - it defines which code belongs to the if statement.

**Working with variables:**
```python
username = "admin"
if username == "admin":
    print("Welcome, administrator!")

password = "secret123"
if password == "secret123":
    print("Login successful")
    
has_permission = True
if has_permission:
    print("Access granted")
    
items_in_cart = 5
if items_in_cart > 0:
    print(f"You have {items_in_cart} items in your cart")
```
 Conditions can check variable values, compare strings, test boolean variables, or evaluate any expression that returns `True` or `False`.



## If-Else Statement

The `else` statement provides an alternative block of code to execute when the `if` condition is `False`.

### Basic If-Else Syntax

```python
# Basic structure
if condition:
    # code to execute if condition is True
else:
    # code to execute if condition is False
```

### If-Else Examples

**Simple either/or decisions:**
```python
age = 16
if age >= 18:
    print("You can vote")
else:
    print("You cannot vote yet")

temperature = 15
if temperature > 20:
    print("It's warm outside")
else:
    print("It's cool outside")
    print("You might need a jacket")

number = 7
if number % 2 == 0:
    print(f"{number} is even")
else:
    print(f"{number} is odd")
```
 Exactly one of the two blocks will execute - either the `if` block (when condition is `True`) or the `else` block (when condition is `False`).

**User input validation:**
```python
user_input = input("Enter a number: ")
if user_input.isdigit():
    number = int(user_input)
    print(f"You entered the number: {number}")
else:
    print("That's not a valid number")

password = input("Enter password: ")
if len(password) >= 8:
    print("Password is strong enough")
else:
    print("Password must be at least 8 characters long")
```
 If-else is perfect for validation scenarios where you need to handle both valid and invalid cases.



## If-Elif-Else Statement

The `elif` (short for "else if") allows you to check multiple conditions in sequence.

### Basic If-Elif-Else Syntax

```python
# Basic structure
if condition1:
    # code for condition1
elif condition2:
    # code for condition2
elif condition3:
    # code for condition3
else:
    # code if no condition is True
```

### Multiple Condition Examples

**Grade classification:**
```python
score = 87
if score >= 90:
    grade = "A"
    print("Excellent work!")
elif score >= 80:
    grade = "B"
    print("Good job!")
elif score >= 70:
    grade = "C"
    print("Satisfactory")
elif score >= 60:
    grade = "D"
    print("Needs improvement")
else:
    grade = "F"
    print("Please see instructor")

print(f"Your grade is: {grade}")
```
 Conditions are checked in order from top to bottom. Once a condition is `True`, its code block executes and the remaining conditions are skipped.

**Weather responses:**
```python
temperature = 32
if temperature > 35:
    print("It's very hot! Stay hydrated.")
    clothing = "shorts and t-shirt"
elif temperature > 25:
    print("It's warm and pleasant.")
    clothing = "light clothing"
elif temperature > 15:
    print("It's cool but comfortable.")
    clothing = "long sleeves"
elif temperature > 5:
    print("It's cold. Bundle up!")
    clothing = "warm jacket"
else:
    print("It's freezing! Stay indoors if possible.")
    clothing = "heavy winter coat"

print(f"Recommended clothing: {clothing}")
```
 Each `elif` provides an additional condition to check. You can have as many `elif` statements as needed.

**User menu system:**
```python
choice = input("Choose an option (1-4): ")

if choice == "1":
    print("You selected: View Profile")
    print("Loading profile...")
elif choice == "2":
    print("You selected: Settings")
    print("Opening settings menu...")
elif choice == "3":
    print("You selected: Help")
    print("Displaying help information...")
elif choice == "4":
    print("You selected: Exit")
    print("Goodbye!")
else:
    print("Invalid choice. Please enter 1, 2, 3, or 4.")
```
 Menu systems commonly use if-elif-else to handle different user choices with a default case for invalid input.



## Comparison Operators

Comparison operators are used to compare values and return boolean results (`True` or `False`).

### Basic Comparison Operators

```python
a = 10
b = 20

# Equal to
print(a == b)       # False
print(a == 10)      # True

# Not equal to
print(a != b)       # True
print(a != 10)      # False

# Less than
print(a < b)        # True
print(b < a)        # False

# Less than or equal to
print(a <= b)       # True
print(a <= 10)      # True

# Greater than
print(a > b)        # False
print(b > a)        # True

# Greater than or equal to
print(a >= b)       # False
print(a >= 10)      # True
```
 Comparison operators compare two values and return a boolean result. Use `==` for equality testing (not `=` which is assignment).

### Comparing Different Data Types

**String comparisons:**
```python
name1 = "Alice"
name2 = "Bob"
name3 = "alice"

if name1 == name2:
    print("Names are the same")
else:
    print("Names are different")

# Case-sensitive comparison
if name1 == name3:
    print("Names match")
else:
    print("Names don't match (case-sensitive)")

# Case-insensitive comparison
if name1.lower() == name3.lower():
    print("Names match (ignoring case)")

# Alphabetical comparison
if name1 < name2:
    print(f"{name1} comes before {name2} alphabetically")
```
 String comparisons are case-sensitive by default. Use `.lower()` or `.upper()` methods for case-insensitive comparisons.

**List and other comparisons:**
```python
list1 = [1, 2, 3]
list2 = [1, 2, 3]
list3 = [1, 2, 4]

if list1 == list2:
    print("Lists are identical")  # This will print

if list1 == list3:
    print("Lists match")
else:
    print("Lists are different")  # This will print

# Length comparison
if len(list1) > 2:
    print("List has more than 2 elements")

# Check if list is empty
my_list = []
if len(my_list) == 0:
    print("List is empty")
# or more pythonic:
if not my_list:
    print("List is empty")
```
 You can compare lists element by element, check lengths, or test for emptiness using various approaches.



## Logical Operators

Logical operators combine multiple conditions to create more complex conditional statements.

### And Operator

**Both conditions must be True:**
```python
age = 25
has_license = True

if age >= 18 and has_license:
    print("You can drive")
else:
    print("You cannot drive")

score1 = 85
score2 = 90
if score1 >= 80 and score2 >= 80:
    print("Both scores are excellent")

# Multiple conditions with and
temperature = 25
is_sunny = True
is_weekend = False

if temperature > 20 and is_sunny and is_weekend:
    print("Perfect day for a picnic!")
elif temperature > 20 and is_sunny:
    print("Nice day, but you might be working")
else:
    print("Not ideal outdoor weather")
```
 The `and` operator returns `True` only when all conditions are `True`. If any condition is `False`, the entire expression is `False`.

### Or Operator

**At least one condition must be True:**
```python
weather = "rainy"
has_umbrella = True

if weather == "sunny" or has_umbrella:
    print("You can go outside")
else:
    print("Better stay indoors")

day = "Saturday"
if day == "Saturday" or day == "Sunday":
    print("It's the weekend!")

# Emergency contact check
has_phone = False
has_email = True
has_address = True

if has_phone or has_email or has_address:
    print("We can contact you")
else:
    print("We need at least one contact method")
```
 The `or` operator returns `True` if any condition is `True`. It only returns `False` when all conditions are `False`.

### Not Operator

**Inverts the boolean value:**
```python
is_logged_in = False

if not is_logged_in:
    print("Please log in first")
else:
    print("Welcome back!")

password = ""
if not password:  # Empty string is falsy
    print("Password cannot be empty")

items = []
if not items:  # Empty list is falsy
    print("Shopping cart is empty")

# Double negative (avoid in practice)
is_not_admin = False
if not is_not_admin:  # Confusing - better to use positive logic
    print("User is admin")
```
 The `not` operator flips `True` to `False` and `False` to `True`. It's useful for checking negative conditions.

### Combining Logical Operators

**Complex conditions with precedence:**
```python
age = 20
has_job = True
has_savings = False
credit_score = 650

# Loan approval logic
if (age >= 18 and has_job) or (has_savings and credit_score > 600):
    print("Loan pre-approved")
else:
    print("Loan application needs review")

# User access control
is_admin = False
is_moderator = True
is_premium_user = True

if is_admin or (is_moderator and is_premium_user):
    print("Access to advanced features granted")
else:
    print("Basic access only")

# Complex validation
username = "user123"
password = "securepass"
two_factor_enabled = True
trusted_device = False

if (username and password) and (two_factor_enabled or trusted_device):
    print("Login successful")
else:
    print("Additional verification required")
```
 Use parentheses to control the order of evaluation. `and` has higher precedence than `or`, but explicit parentheses make intentions clearer.



## Truthiness and Falsiness

Python evaluates many values as `True` or `False` in boolean contexts, even if they're not explicitly boolean.

### Falsy Values

**Values that evaluate to False:**
```python
# These all evaluate to False in boolean context
falsy_values = [
    False,      # Boolean False
    0,          # Zero (integer)
    0.0,        # Zero (float)
    "",         # Empty string
    [],         # Empty list
    {},         # Empty dictionary
    set(),      # Empty set
    None        # None value
]

for value in falsy_values:
    if not value:
        print(f"{repr(value)} is falsy")

# Practical examples
user_input = ""
if user_input:
    print("User entered something")
else:
    print("User entered nothing")  # This prints

items_in_cart = 0
if items_in_cart:
    print("Cart has items")
else:
    print("Cart is empty")  # This prints
```
 These values are considered "falsy" and evaluate to `False` in boolean contexts. This allows for concise checks.

### Truthy Values

**Values that evaluate to True:**
```python
# These all evaluate to True in boolean context
truthy_values = [
    True,           # Boolean True
    1,              # Any non-zero number
    -1,             # Negative numbers
    "hello",        # Non-empty string
    " ",            # String with whitespace
    [1, 2, 3],      # Non-empty list
    {"a": 1},       # Non-empty dictionary
    {1, 2, 3}       # Non-empty set
]

for value in truthy_values:
    if value:
        print(f"{repr(value)} is truthy")

# Practical examples
user_name = "Alice"
if user_name:
    print(f"Hello, {user_name}!")  # This prints

numbers = [1, 2, 3, 4, 5]
if numbers:
    print(f"Processing {len(numbers)} numbers")  # This prints
```
 Any value that isn't falsy is considered "truthy" and evaluates to `True` in boolean contexts.

### Using Truthiness Effectively

```python
# Check for non-empty data
data = input("Enter some data: ")
if data:  # More concise than: if data != ""
    print("Processing your data...")
else:
    print("No data provided")

# Validate list has content
results = get_search_results()  # Assume this returns a list
if results:  # More concise than: if len(results) > 0
    print(f"Found {len(results)} results")
    for result in results:
        print(f"- {result}")
else:
    print("No results found")

# Default value pattern
username = user_input or "guest"  # Use "guest" if user_input is falsy
print(f"Welcome, {username}")

def get_search_results():
    # Placeholder function
    return ["result1", "result2"]
```
 Using truthiness makes code more concise and readable. The `or` operator can provide default values when the first value is falsy.



## Nested Conditions

You can nest if statements inside other if statements to create more complex decision logic.

### Basic Nested Conditions

**Conditions within conditions:**
```python
age = 25
has_license = True
has_car = False

if age >= 18:
    print("You are old enough to drive")
    if has_license:
        print("You have a license")
        if has_car:
            print("You can drive your own car")
        else:
            print("You need to borrow or rent a car")
    else:
        print("You need to get a license first")
else:
    print("You are too young to drive")
```
 Nested conditions allow you to check additional conditions only when outer conditions are met. Each level of nesting adds another layer of indentation.

### Complex Decision Trees

**Multi-level validation:**
```python
username = "admin"
password = "correct123"
is_account_active = True
login_attempts = 2

if username:
    print("Username provided")
    if password:
        print("Password provided")
        if username == "admin" and password == "correct123":
            print("Credentials are correct")
            if is_account_active:
                print("Account is active")
                if login_attempts < 3:
                    print("LOGIN SUCCESSFUL")
                    print("Welcome to the system!")
                else:
                    print("Too many failed attempts. Account locked.")
            else:
                print("Account is inactive. Contact administrator.")
        else:
            print("Invalid username or password")
    else:
        print("Password is required")
else:
    print("Username is required")
```
 Nested conditions create decision trees where each level depends on the previous level's outcome.

### Avoiding Deep Nesting

**Using early returns (in functions):**
```python
def check_access(username, password, is_active, attempts):
    if not username:
        return "Username is required"
    
    if not password:
        return "Password is required"
    
    if username != "admin" or password != "correct123":
        return "Invalid credentials"
    
    if not is_active:
        return "Account inactive"
    
    if attempts >= 3:
        return "Account locked"
    
    return "Access granted"

# Usage
result = check_access("admin", "correct123", True, 1)
print(result)
```
 Early returns can make deeply nested conditions more readable by handling error cases first.

**Using logical operators instead of nesting:**
```python
# Instead of deep nesting:
if age >= 18:
    if has_license:
        if has_insurance:
            print("Can drive")

# Use logical operators:
if age >= 18 and has_license and has_insurance:
    print("Can drive")

# Complex example
age = 25
has_license = True
has_insurance = True
car_is_working = True

if age >= 18 and has_license and has_insurance and car_is_working:
    print("Ready to drive!")
else:
    print("Cannot drive right now")
    if age < 18:
        print("- Too young")
    if not has_license:
        print("- No license")
    if not has_insurance:
        print("- No insurance")
    if not car_is_working:
        print("- Car needs repair")
```
 Logical operators can often replace nested conditions, making code more readable and maintainable.



## Ternary Operator (Conditional Expression)

The ternary operator provides a concise way to assign values based on conditions.

### Basic Ternary Syntax

```python
# Basic structure
variable = value_if_true if condition else value_if_false
```

### Simple Ternary Examples

**Basic value assignment:**
```python
age = 20
status = "adult" if age >= 18 else "minor"
print(f"You are a {status}")

score = 85
grade = "Pass" if score >= 60 else "Fail"
print(f"Result: {grade}")

temperature = 30
clothing = "shorts" if temperature > 25 else "pants"
print(f"Wear: {clothing}")

# Without ternary (more verbose):
if temperature > 25:
    clothing = "shorts"
else:
    clothing = "pants"
```
 The ternary operator evaluates the condition and returns the first value if `True`, otherwise the second value. It's a shorthand for simple if-else assignments.

### Ternary with Function Calls

```python
numbers = [1, 2, 3, 4, 5]
result = max(numbers) if numbers else 0
print(f"Maximum: {result}")

username = input("Enter username: ")
display_name = username.title() if username else "Guest"
print(f"Welcome, {display_name}!")

# Chained ternary (use sparingly)
score = 95
grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "F"
print(f"Grade: {grade}")
```
 Ternary operators can call functions and be chained, but chaining reduces readability and should be used sparingly.

### When to Use Ternary

**Good use cases:**
```python
# Simple value assignments
is_weekend = True
sleep_time = 10 if is_weekend else 7

# Default values
user_input = ""
name = user_input if user_input else "Anonymous"

# Simple mathematical operations
x = 10
abs_x = x if x >= 0 else -x

# List comprehensions with conditions
numbers = [1, -2, 3, -4, 5]
absolute_numbers = [n if n >= 0 else -n for n in numbers]
```

**Avoid for complex logic:**
```python
# Avoid - too complex for ternary
score = 85
# This is hard to read:
result = "Excellent" if score >= 90 else "Good" if score >= 80 else "Average" if score >= 70 else "Poor"

# Better as regular if-elif-else:
if score >= 90:
    result = "Excellent"
elif score >= 80:
    result = "Good"
elif score >= 70:
    result = "Average"
else:
    result = "Poor"
```
 Use ternary operators for simple conditions. For complex logic, regular if-elif-else statements are more readable.



## Common Patterns and Best Practices

### Input Validation Patterns

**Validating user input:**
```python
# Number validation
user_input = input("Enter a number: ")
if user_input.isdigit():
    number = int(user_input)
    if 1 <= number <= 100:
        print(f"Valid number: {number}")
    else:
        print("Number must be between 1 and 100")
else:
    print("Please enter a valid number")

# Email validation (simple)
email = input("Enter email: ")
if "@" in email and "." in email:
    if email.count("@") == 1:
        print("Email format looks valid")
    else:
        print("Email should contain exactly one @ symbol")
else:
    print("Email must contain @ and . symbols")

# Password strength
password = input("Enter password: ")
is_long_enough = len(password) >= 8
has_digit = any(char.isdigit() for char in password)
has_letter = any(char.isalpha() for char in password)

if is_long_enough and has_digit and has_letter:
    print("Strong password!")
elif is_long_enough:
    print("Password needs both letters and numbers")
else:
    print("Password must be at least 8 characters")
```
 Layer validation checks from basic to specific, providing helpful feedback for each failure case.

### Error Handling Patterns

**Graceful error handling:**
```python
# File processing with validation
filename = input("Enter filename: ")
if filename:
    if filename.endswith('.txt'):
        try:
            with open(filename, 'r') as file:
                content = file.read()
                if content:
                    print(f"File contains {len(content)} characters")
                else:
                    print("File is empty")
        except FileNotFoundError:
            print(f"File '{filename}' not found")
        except PermissionError:
            print("Permission denied to read file")
    else:
        print("Please provide a .txt file")
else:
    print("Filename cannot be empty")

# Division with error checking
dividend = float(input("Enter dividend: "))
divisor = float(input("Enter divisor: "))

if divisor != 0:
    result = dividend / divisor
    print(f"{dividend} ÷ {divisor} = {result}")
else:
    print("Cannot divide by zero!")
```
 Check for error conditions before attempting operations that might fail.

### Configuration and Settings

**Feature flags and configuration:**
```python
# Application configuration
DEBUG_MODE = True
MAINTENANCE_MODE = False
USER_ROLE = "admin"

# Feature availability
if DEBUG_MODE:
    print("Debug information: Application starting...")

if not MAINTENANCE_MODE:
    print("System is online and ready")
    
    if USER_ROLE == "admin":
        print("Admin panel available")
        print("User management enabled")
    elif USER_ROLE == "moderator":
        print("Moderation tools available")
    else:
        print("Standard user interface")
else:
    print("System is under maintenance")

# Environment-based settings
ENVIRONMENT = "development"  # Could be "development", "testing", "production"

if ENVIRONMENT == "development":
    database_url = "localhost:5432"
    log_level = "DEBUG"
    cache_enabled = False
elif ENVIRONMENT == "testing":
    database_url = "test-db:5432"
    log_level = "INFO"
    cache_enabled = True
elif ENVIRONMENT == "production":
    database_url = "prod-db:5432"
    log_level = "ERROR"
    cache_enabled = True
else:
    raise ValueError(f"Unknown environment: {ENVIRONMENT}")

print(f"Database: {database_url}")
print(f"Log level: {log_level}")
print(f"Cache enabled: {cache_enabled}")
```
 Use conditions to adapt program behavior based on configuration settings or environment variables.



## Performance and Best Practices

### Efficient Condition Checking

**Order conditions by likelihood:**
```python
# Put most likely conditions first
user_type = "regular"  # Most users are regular

if user_type == "regular":
    # Most common case first
    apply_regular_discount()
elif user_type == "premium":
    # Less common
    apply_premium_discount()
elif user_type == "vip":
    # Least common
    apply_vip_discount()

def apply_regular_discount():
    print("5% discount applied")

def apply_premium_discount():
    print("15% discount applied")

def apply_vip_discount():
    print("25% discount applied")
```
 Python evaluates conditions from top to bottom, so put the most likely conditions first for better performance.

**Short-circuit evaluation:**
```python
# Expensive operations last
def expensive_check():
    print("Performing expensive check...")
    return True

def quick_check():
    return False

# Good: quick_check() is False, so expensive_check() never runs
if quick_check() and expensive_check():
    print("Both checks passed")

# Also works with 'or'
if quick_check() or expensive_check():
    print("At least one check passed")
```
 Python uses short-circuit evaluation - it stops checking conditions as soon as the result is determined.

### Avoiding Common Mistakes

**Comparison mistakes:**
```python
# Wrong: assignment instead of comparison
x = 5
if x = 10:  # SyntaxError - should be ==
    print("x is 10")

# Correct:
if x == 10:
    print("x is 10")

# Wrong: comparing different types inconsistently
age_string = "25"
if age_string > 18:  # Comparing string to number
    print("Adult")

# Correct: convert types first
age = int(age_string)
if age > 18:
    print("Adult")
```

**Boolean comparison mistakes:**
```python
is_active = True

# Unnecessary - avoid
if is_active == True:
    print("Active")

# Better - direct boolean check
if is_active:
    print("Active")

# For false checks
if not is_active:
    print("Inactive")

# Not recommended
if is_active == False:
    print("Inactive")
```
 Don't compare boolean variables to `True` or `False` explicitly - just use the variable directly or with `not`.



## Quick Reference Summary

| Condition Type | Syntax | Use Case |
|-|--|-|
| **Simple if** | `if condition:` | Execute code when condition is true |
| **If-else** | `if condition: ... else:` | Choose between two options |
| **If-elif-else** | `if cond1: ... elif cond2: ... else:` | Choose from multiple options |
| **Nested if** | `if cond1: if cond2:` | Conditions within conditions |
| **Ternary** | `value_if_true if condition else value_if_false` | Concise conditional assignment |

### Comparison Operators
| Operator | Meaning | Example |
|-|||
| `==` | Equal to | `x == 5` |
| `!=` | Not equal to | `x != 5` |
| `<` | Less than | `x < 5` |
| `<=` | Less than or equal | `x <= 5` |
| `>` | Greater than | `x > 5` |
| `>=` | Greater than or equal | `x >= 5` |

### Logical Operators
| Operator | Meaning | Example |
|-|||
| `and` | Both conditions true | `x > 0 and x < 10` |
| `or` | At least one condition true | `x < 0 or x > 10` |
| `not` | Opposite of condition | `not is_empty` |

**Comprehensive example combining multiple concepts:**
```python
# Student Management System
def process_student_application(name, age, gpa, has_recommendation, financial_aid_needed):
    print(f"=== Processing Application for {name} ===")
    
    # Basic eligibility check
    if not name or not isinstance(age, int) or age <= 0:
        return "Invalid application data"
    
    # Age requirements
    if age < 16:
        return "Applicant too young (minimum age: 16)"
    elif age > 65:
        print("⚠️  Senior applicant - special review required")
    
    # Academic requirements
    if gpa < 0 or gpa > 4.0:
        return "Invalid GPA (must be 0.0-4.0)"
    
    # Determine admission status
    if gpa >= 3.5:
        admission_status = "Accepted"
        scholarship_eligible = True
        print("🌟 Excellent academic record!")
    elif gpa >= 3.0:
        admission_status = "Accepted" if has_recommendation else "Conditional"
        scholarship_eligible = has_recommendation
        print("👍 Good academic standing")
    elif gpa >= 2.5:
        if has_recommendation:
            admission_status = "Conditional"
            scholarship_eligible = False
            print("📋 Conditional acceptance with recommendation")
        else:
            admission_status = "Rejected"
            scholarship_eligible = False
            print("❌ Below minimum requirements without recommendation")
    else:
        admission_status = "Rejected"
        scholarship_eligible = False
        print("❌ GPA too low for admission")
    
    # Financial aid processing
    if admission_status == "Accepted" and financial_aid_needed:
        if gpa >= 3.8:
            aid_amount = 5000
            print("💰 Full scholarship awarded!")
        elif gpa >= 3.5:
            aid_amount = 3000
            print("💰 Partial scholarship awarded")
        elif scholarship_eligible:
            aid_amount = 1000
            print("💰 Need-based aid available")
        else:
            aid_amount = 0
            print("💸 No financial aid available")
    else:
        aid_amount = 0
    
    # Final summary
    result = {
        "status": admission_status,
        "scholarship_eligible": scholarship_eligible,
        "financial_aid": aid_amount,
        "message": "Welcome to our university!" if admission_status == "Accepted" else "Thank you for your application"
    }
    
    print(f"📋 Final Status: {admission_status}")
    if aid_amount > 0:
        print(f"💰 Financial Aid: ${aid_amount}")
    
    return result

# Example usage
students = [
    ("Alice Johnson", 18, 3.8, True, True),
    ("Bob Smith", 17, 2.9, False, False),
    ("Charlie Brown", 19, 3.2, True, True),
    ("Diana Prince", 16, 4.0, True, False),
    ("Eve Wilson", 20, 2.3, False, True)
]

print("🎓 UNIVERSITY ADMISSION PROCESSING 🎓\n")

for student_data in students:
    name, age, gpa, has_rec, needs_aid = student_data
    result = process_student_application(name, age, gpa, has_rec, needs_aid)
    print("-" * 50)

print("\n✅ All applications processed!")
```

### Falsy Values Quick Reference
```python
# All these evaluate to False in boolean context:
if not False:        print("False is falsy")
if not 0:           print("0 is falsy")
if not 0.0:         print("0.0 is falsy")  
if not "":          print("Empty string is falsy")
if not []:          print("Empty list is falsy")
if not {}:          print("Empty dict is falsy")
if not set():       print("Empty set is falsy")
if not None:        print("None is falsy")
```

### Common Condition Patterns
```python
# 1. Range checking
def categorize_age(age):
    if age < 0:
        return "Invalid age"
    elif age < 13:
        return "Child"
    elif age < 20:
        return "Teenager"  
    elif age < 65:
        return "Adult"
    else:
        return "Senior"

# 2. Multiple criteria validation
def validate_password(password):
    if not password:
        return "Password cannot be empty"
    
    issues = []
    if len(password) < 8:
        issues.append("must be at least 8 characters")
    if not any(c.isupper() for c in password):
        issues.append("must contain uppercase letter")
    if not any(c.islower() for c in password):
        issues.append("must contain lowercase letter")
    if not any(c.isdigit() for c in password):
        issues.append("must contain a number")
    
    if issues:
        return "Password " + ", ".join(issues)
    return "Password is valid"

# 3. State machine pattern
def process_order(current_state, action):
    if current_state == "cart":
        if action == "checkout":
            return "processing"
        elif action == "add_item":
            return "cart"
        elif action == "clear":
            return "empty"
    elif current_state == "processing":
        if action == "confirm":
            return "confirmed"
        elif action == "cancel":
            return "cancelled"
    elif current_state == "confirmed":
        if action == "ship":
            return "shipped"
        elif action == "cancel":
            return "cancelled"
    elif current_state == "shipped":
        if action == "deliver":
            return "delivered"
        elif action == "return":
            return "returned"
    
    return current_state  # No state change

# 4. Configuration-based behavior
def get_database_config(environment):
    base_config = {
        "timeout": 30,
        "pool_size": 10
    }
    
    if environment == "development":
        base_config.update({
            "host": "localhost",
            "debug": True,
            "ssl": False
        })
    elif environment == "testing":
        base_config.update({
            "host": "test-db.company.com",
            "debug": True,
            "ssl": True
        })
    elif environment == "production":
        base_config.update({
            "host": "prod-db.company.com",
            "debug": False,
            "ssl": True,
            "pool_size": 50
        })
    else:
        raise ValueError(f"Unknown environment: {environment}")
    
    return base_config

# 5. Input sanitization and validation
def process_user_input(raw_input):
    # Clean the input
    if not raw_input:
        return None, "Input cannot be empty"
    
    cleaned = raw_input.strip()
    if not cleaned:
        return None, "Input cannot be only whitespace"
    
    # Validate length
    if len(cleaned) > 100:
        return None, "Input too long (max 100 characters)"
    
    # Check for forbidden characters
    forbidden_chars = ['<', '>', '&', '"', "'"]
    if any(char in cleaned for char in forbidden_chars):
        return None, "Input contains forbidden characters"
    
    # Additional validation based on content
    if cleaned.lower() in ['admin', 'root', 'system']:
        return None, "Reserved word not allowed"
    
    return cleaned, "Valid input"

# Usage examples
print("=== CONDITION PATTERNS EXAMPLES ===\n")

# Age categorization
ages = [5, 15, 25, 70, -1]
for age in ages:
    category = categorize_age(age)
    print(f"Age {age}: {category}")

print()

# Password validation
passwords = ["weak", "StrongPass123", "nodigits", "NOLOWERCASE123"]
for pwd in passwords:
    result = validate_password(pwd)
    print(f"'{pwd}': {result}")

print()

# Order state machine
order_state = "cart"
actions = ["add_item", "checkout", "confirm", "ship", "deliver"]
for action in actions:
    new_state = process_order(order_state, action)
    print(f"State '{order_state}' + action '{action}' → '{new_state}'")
    order_state = new_state
```

This comprehensive guide covers all essential conditional concepts with detailed explanations, practical examples, performance considerations, and best practices for effective condition usage in Python!