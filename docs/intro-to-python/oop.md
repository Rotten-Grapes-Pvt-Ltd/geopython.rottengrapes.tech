# Object-Oriented Programming (OOP) in Python

Object-Oriented Programming (OOP) is one of the most powerful paradigms in Python.  
It allows you to model real-world entities as **objects**, which combine **data** (attributes) and **behavior** (methods).

---

## What is OOP?

OOP is a programming approach where the focus is on **objects** rather than functions or logic.  
Objects are created using **classes**, which serve as blueprints.  

For example:
- A class defines what attributes and behaviors an object will have.
- An object is an instance of that class with actual data.

---

## Key Building Blocks of OOP

- **Class** → Blueprint or template for creating objects.  
- **Object** → An instance of a class containing specific data.  
- **Attributes** → Variables that store data related to an object.  
- **Methods** → Functions that define the behavior of an object.  
- **Constructor (`__init__`)** → A special method used to initialize new objects.

---

## Example: Creating a Class and Object

```python
class Car:
    def __init__(self, brand, color):
        self.brand = brand
        self.color = color

    def drive(self):
        print(f"The {self.color} {self.brand} is driving!")

# Creating objects
car1 = Car("Toyota", "Red")
car2 = Car("Tesla", "Black")

car1.drive()
car2.drive()
```

Each object (`car1`, `car2`) has its own **state** (brand and color) but shares the same **behavior** (`drive()`).

---

## Principles of OOP

### **Encapsulation**
Encapsulation means keeping the internal details of an object hidden and only exposing what’s necessary.  
It helps prevent direct modification of internal data.

```python
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance   # private variable

    def deposit(self, amount):
        self.__balance += amount

    def get_balance(self):
        return self.__balance
```

Here, `__balance` cannot be accessed directly from outside the class.

---

### **Inheritance**
Inheritance allows a class to use properties and methods of another class.  
It encourages **code reusability**.

```python
class Animal:
    def speak(self):
        print("Animal speaks")

class Dog(Animal):
    def speak(self):
        print("Woof!")

dog = Dog()
dog.speak()
```

---

### **Polymorphism**
Polymorphism means using a common interface for different types of objects.  
It allows methods with the same name to behave differently depending on the class.

```python
class Cat:
    def speak(self):
        print("Meow!")

animals = [Dog(), Cat()]
for animal in animals:
    animal.speak()
```
Each class implements its own version of `speak()`.

---

### **Abstraction**
Abstraction hides the internal implementation and shows only essential features.  
In Python, it can be achieved using **abstract base classes**.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2
```

---

## **Special Methods (Dunder Methods)**

Python classes can define special (double underscore) methods that integrate with Python’s built-in operations.

Common examples:
- `__init__` → Constructor  
- `__str__` → String representation  
- `__len__` → Length of an object  
- `__add__` → Add two objects  

```python
class Book:
    def __init__(self, title, pages):
        self.title = title
        self.pages = pages

    def __str__(self):
        return f"Book: {self.title}"

    def __len__(self):
        return self.pages

book = Book("Python Guide", 350)
print(book)
print(len(book))
```

---

## **OOP in Practice**

- **Real-world analogy:**
  - **Class:** Blueprint of a house  
  - **Object:** Actual house built from the blueprint  
  - **Attributes:** Color, size, number of rooms  
  - **Methods:** Open door, close window  

- **Applications of OOP:**
  - GUI frameworks (Tkinter, PyQt)
  - Game development (player, enemy objects)
  - Web development (Django models)
  - Data analysis (Pandas DataFrames are objects)

---

## **Conclusion**

Object-Oriented Programming helps you organize your code into manageable, modular pieces.  
By applying concepts like **encapsulation**, **inheritance**, **polymorphism**, and **abstraction**, you can write cleaner, reusable, and scalable Python programs.
