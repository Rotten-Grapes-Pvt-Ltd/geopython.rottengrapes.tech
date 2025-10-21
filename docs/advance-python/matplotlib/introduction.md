# Introduction to Matplotlib

Matplotlib is the most widely used **data visualization library in Python**. It provides the foundation for many other plotting libraries such as Seaborn and Pandas’ built-in plotting functions. With Matplotlib, you can create everything from simple line charts to complex scientific visualizations.

---

## Why Learn Matplotlib?

- **Universal foundation**: Most other Python visualization tools (Seaborn, Plotly, Pandas plots) are built on top of Matplotlib.  
- **Customizable**: Almost every element of a plot can be changed — colors, line styles, fonts, legends, axes.  
- **Flexible**: Works with NumPy arrays, Pandas DataFrames, and even raw Python lists.  
- **Publication quality**: Used in academic papers, research, and professional reports.

---

## Installing Matplotlib

If you don’t have it installed:

```bash
pip install matplotlib
```

Importing conventionally:

```python
import matplotlib.pyplot as plt
```

---

## How Matplotlib Fits With NumPy & Pandas

- **NumPy**: Provides arrays of numbers that Matplotlib can visualize.  
- **Pandas**: Provides tabular data (rows and columns) that can be plotted directly with Matplotlib.  
- **Matplotlib**: Turns that numerical/tabular data into visualizations.

Example:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.random.randn(100).cumsum()
df = pd.DataFrame(data, columns=["values"])

plt.plot(df["values"])
plt.show()
```

---

## Anatomy of a Matplotlib Figure

Every Matplotlib visualization has a structure:

- **Figure**: The whole window or page where plots are drawn.  
- **Axes**: The area where the actual plot is drawn (can be multiple in one figure).  
- **Axis**: The x and y coordinate system of the plot.  
- **Artist**: Everything you see in the figure (titles, lines, text, legends).

Example:

```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [2, 4, 6])
ax.set_title("My First Plot")
plt.show()
```

---

## Interfaces: Pyplot vs Object-Oriented (OO)

- **Pyplot style (quick & easy)**: Works like MATLAB, good for small scripts.

```python
plt.plot([1, 2, 3], [2, 4, 6])
plt.title("Quick Plot")
plt.show()
```

- **Object-Oriented style (recommended)**: Gives more control and scales better.

```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [2, 4, 6])
ax.set_title("Better Plot")
plt.show()
```

---

## First Plot: Step by Step

```python
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.figure(figsize=(6,4))        # Set figure size
plt.plot(x, y, color="blue", marker="o", linestyle="--")
plt.title("Simple Line Chart")
plt.xlabel("X-axis Label")
plt.ylabel("Y-axis Label")
plt.grid(True)
plt.show()
```

---

## Adding Labels, Legends, and Colors

```python
x = [1, 2, 3, 4, 5]
y1 = [1, 4, 9, 16, 25]
y2 = [2, 4, 6, 8, 10]

plt.plot(x, y1, label="Squares", color="red")
plt.plot(x, y2, label="Doubles", color="green")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Multiple Lines with Legend")
plt.legend()
plt.show()
```

---

## Saving Plots

```python
plt.plot([1, 2, 3], [2, 4, 6])
plt.savefig("my_plot.png")   # Save as PNG
plt.savefig("my_plot.pdf")   # Save as PDF
```

---

## Best Practices & Tips

- Prefer **object-oriented interface** for bigger projects.  
- Use **labels and legends** to make plots understandable.  
- Always set **titles and axis labels**.  
- Choose **colors and markers** that improve readability.  
- Use `plt.savefig()` to preserve results for reports.  
