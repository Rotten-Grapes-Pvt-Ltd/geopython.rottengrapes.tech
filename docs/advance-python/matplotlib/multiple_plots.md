# Chapter 4 — Working with Multiple Plots and Subplots

When analyzing data, we often need to visualize multiple plots side by side or stacked together for comparison. Matplotlib makes this possible using **figures** and **subplots**.

---

## Figure and Axes: The Basics

- **Figure**: The entire window or canvas that holds your plots.  
- **Axes**: The actual plot area inside the figure (including x-axis, y-axis, labels, and data).  
- A single figure can contain **multiple Axes** (plots).

---

## Creating Multiple Plots with `subplot()`

The `plt.subplot(nrows, ncols, index)` function divides the figure into a grid.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
plt.plot(x, y1, 'b')
plt.title("Sine Wave")

plt.subplot(2, 1, 2)  # second plot
plt.plot(x, y2, 'r')
plt.title("Cosine Wave")

plt.tight_layout()
plt.show()
```

This creates two vertically stacked plots.

---

## Creating Grids of Plots with `subplots()`

The `plt.subplots()` method is more flexible and is the recommended approach.

```python
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

x = np.linspace(0, 5, 100)

axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title("Sine")

axes[0, 1].plot(x, np.cos(x))
axes[0, 1].set_title("Cosine")

axes[1, 0].plot(x, np.tan(x))
axes[1, 0].set_title("Tangent")

axes[1, 1].plot(x, np.exp(x))
axes[1, 1].set_title("Exponential")

plt.tight_layout()
plt.show()
```

Here we created a **2×2 grid** of subplots.

---

## Sharing Axes

You can make plots share the same x-axis or y-axis for easier comparison.

```python
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

t = np.linspace(0, 2*np.pi, 400)
axes[0].plot(t, np.sin(t), 'g')
axes[0].set_title("Sine")

axes[1].plot(t, np.sin(2*t), 'm')
axes[1].set_title("Sine (double frequency)")

plt.tight_layout()
plt.show()
```

---

## Different Plot Types in One Figure

You can mix multiple chart types in one figure using subplots.

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

x = np.linspace(0, 10, 100)

# Line plot
ax1.plot(x, np.sin(x), label="Sine")
ax1.set_title("Line Plot")
ax1.legend()

# Histogram
data = np.random.randn(1000)
ax2.hist(data, bins=30, color="skyblue", edgecolor="black")
ax2.set_title("Histogram")

plt.tight_layout()
plt.show()
```

---

## Advanced Layouts with `GridSpec`

For more control over subplot positioning, Matplotlib offers `GridSpec`.

```python
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(3, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, :])       # top row (spans all columns)
ax2 = fig.add_subplot(gs[1, :-1])     # middle row, first 2 columns
ax3 = fig.add_subplot(gs[1:, -1])     # right column (last 2 rows)
ax4 = fig.add_subplot(gs[2, 0])       # bottom-left
ax5 = fig.add_subplot(gs[2, 1])       # bottom-middle

ax1.plot(np.random.rand(10))
ax2.hist(np.random.randn(100))
ax3.scatter(np.random.rand(20), np.random.rand(20))
ax4.bar([1, 2, 3], [3, 2, 5])
ax5.plot(np.cumsum(np.random.randn(50)), 'r')

fig.suptitle("Complex Layout with GridSpec")
plt.tight_layout()
plt.show()
```

---

## Practical Example: Comparing Stock Trends

```python
days = np.arange(1, 11)
company_a = np.random.randint(50, 100, size=10)
company_b = np.random.randint(60, 110, size=10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(days, company_a, marker="o", label="Company A")
ax1.set_title("Company A Stock")
ax1.set_xlabel("Day")
ax1.set_ylabel("Price")
ax1.legend()

ax2.plot(days, company_b, marker="s", label="Company B", color="orange")
ax2.set_title("Company B Stock")
ax2.set_xlabel("Day")
ax2.set_ylabel("Price")
ax2.legend()

plt.suptitle("Stock Comparison")
plt.tight_layout()
plt.show()
```

---

## Key Takeaways

- Use `subplot()` for quick simple grids.  
- Use `subplots()` for flexibility and array-like access to Axes.  
- Use `sharex`/`sharey` to align scales.  
- Use `GridSpec` for complex layouts.  
- Always apply `tight_layout()` to fix overlapping text.
