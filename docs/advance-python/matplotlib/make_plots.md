# Basic Plots in Matplotlib

In this chapter, we will explore the most commonly used plots in Matplotlib. These basic plots form the foundation of data visualization in Python and will help you represent your data in multiple ways.

---

## Line Plot

A **line plot** is used to represent data points connected by straight lines. It’s useful for showing trends over time.

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y, color="blue", marker="o", linestyle="-")
plt.title("Line Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()
```

---

## Scatter Plot

A **scatter plot** shows individual data points on the graph. It’s useful for finding patterns, correlations, or outliers.

```python
x = [5, 7, 8, 7, 6, 9, 5, 6, 7, 8]
y = [99, 86, 87, 88, 100, 86, 103, 87, 94, 78]

plt.scatter(x, y, color="red")
plt.title("Scatter Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

---

## Bar Plot

A **bar plot** is used to compare quantities between categories.

```python
categories = ["A", "B", "C", "D"]
values = [3, 7, 2, 5]

plt.bar(categories, values, color="purple")
plt.title("Bar Plot Example")
plt.xlabel("Categories")
plt.ylabel("Values")
plt.show()
```

---

## Horizontal Bar Plot

You can also plot horizontal bars using `barh()`.

```python
categories = ["A", "B", "C", "D"]
values = [3, 7, 2, 5]

plt.barh(categories, values, color="green")
plt.title("Horizontal Bar Plot Example")
plt.xlabel("Values")
plt.ylabel("Categories")
plt.show()
```

---

## Histogram

A **histogram** shows the distribution of data by dividing it into intervals (bins).

```python
import numpy as np

data = np.random.randn(1000)

plt.hist(data, bins=30, color="skyblue", edgecolor="black")
plt.title("Histogram Example")
plt.xlabel("Bins")
plt.ylabel("Frequency")
plt.show()
```

---

## Pie Chart

A **pie chart** represents data as slices of a circle, useful for showing proportions.

```python
sizes = [20, 30, 25, 25]
labels = ["A", "B", "C", "D"]
colors = ["gold", "lightcoral", "lightskyblue", "yellowgreen"]

plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
plt.title("Pie Chart Example")
plt.show()
```

---

## Best Practices for Basic Plots

- Use line plots for **trends**.  
- Use scatter plots for **correlations**.  
- Use bar plots for **categorical comparisons**.  
- Use histograms for **distributions**.  
- Use pie charts carefully — they are less precise than bar plots.  

---

✅ You now know how to create the most common types of plots in Matplotlib. In the next chapter, we’ll learn how to **customize plots** to make them more informative and visually appealing.
