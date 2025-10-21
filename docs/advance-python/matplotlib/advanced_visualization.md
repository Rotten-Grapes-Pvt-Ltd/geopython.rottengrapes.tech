# Chapter 5 — Advanced Visualizations and Techniques in Matplotlib

## Learning Objectives

- Learn advanced plotting methods for deeper insights.
- Work with histograms, scatter plots, and boxplots.
- Visualize 2D data using heatmaps and imshow.
- Create stacked and grouped bar plots.
- Handle time series data effectively.
- Integrate Matplotlib with Pandas.
- Apply colormaps, annotations, and advanced styling.

---

## Histograms — Understanding Distributions

- Use histograms to see how values are distributed across bins.

```python
import matplotlib.pyplot as plt
import numpy as np

data = np.random.randn(1000)

plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title("Histogram of Random Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

- Adjust `bins` to control granularity.
- Use `density=True` for probability density.

---

## Boxplots — Detecting Outliers and Spread

- Boxplots show median, quartiles, and outliers.

```python
np.random.seed(42)
data = [np.random.normal(0, std, 100) for std in range(1, 4)]

plt.boxplot(data, vert=True, patch_artist=True)
plt.title("Boxplot of Different Distributions")
plt.xticks([1,2,3], ['Std=1', 'Std=2', 'Std=3'])
plt.show()
```

- Useful for comparing multiple groups.

---

## Scatter Plots — Relationships Between Variables

- Scatter plots are great for correlation and clustering.

```python
x = np.random.rand(100)
y = x*2 + np.random.randn(100)*0.2

plt.scatter(x, y, c='blue', alpha=0.6, edgecolor='black')
plt.title("Scatter Plot Example")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.show()
```

- Use `c` and `s` parameters to control colors and marker sizes.

---

## Heatmaps and Imshow — Visualizing 2D Data

- Use `imshow` to plot matrices, images, or heatmaps.

```python
matrix = np.random.rand(10,10)

plt.imshow(matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label="Intensity")
plt.title("Heatmap Example")
plt.show()
```

- Common in correlation matrices and image data.

---

## Stacked and Grouped Bar Plots

- Stacked bars show parts of a whole.

```python
labels = ['A','B','C','D']
x = np.arange(len(labels))

men = [20, 34, 30, 35]
women = [25, 32, 34, 20]

plt.bar(x, men, label='Men')
plt.bar(x, women, bottom=men, label='Women')
plt.xticks(x, labels)
plt.ylabel("Scores")
plt.title("Stacked Bar Plot")
plt.legend()
plt.show()
```

- Use side-by-side for comparisons:

```python
width = 0.35
plt.bar(x - width/2, men, width, label='Men')
plt.bar(x + width/2, women, width, label='Women')
plt.xticks(x, labels)
plt.title("Grouped Bar Plot")
plt.legend()
plt.show()
```

---

## Time Series Visualization

- Matplotlib works well with datetime objects.

```python
import pandas as pd

dates = pd.date_range(start="2023-01-01", periods=10)
values = np.random.randint(10, 100, size=10)

plt.plot(dates, values, marker='o')
plt.title("Time Series Plot")
plt.xlabel("Date")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.show()
```

- Pandas integrates directly with plotting: `df['col'].plot()`.

---

## Pandas + Matplotlib Integration

- Pandas has built-in Matplotlib support.

```python
df = pd.DataFrame({
    "A": np.random.randn(100).cumsum(),
    "B": np.random.randn(100).cumsum()
}, index=pd.date_range("2023-01-01", periods=100))

df.plot(title="Cumulative Random Walks", figsize=(8,4))
plt.show()
```

- Great for quick exploratory data analysis.

---

## Colormaps and Advanced Styling

- Use colormaps for better representation.

```python
x = np.random.rand(100)
y = np.random.rand(100)
colors = np.random.rand(100)

plt.scatter(x, y, c=colors, cmap='plasma', s=80, alpha=0.7)
plt.colorbar(label="Color Intensity")
plt.title("Scatter with Colormap")
plt.show()
```

---

## Annotations for Clarity

- Add text, arrows, and highlights to make plots more informative.

```python
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("Annotated Sine Curve")
plt.xlabel("X")
plt.ylabel("sin(X)")

# annotate max
max_idx = np.argmax(y)
plt.annotate("Max Point",
             xy=(x[max_idx], y[max_idx]),
             xytext=(x[max_idx]+1, y[max_idx]),
             arrowprops=dict(facecolor='red', shrink=0.05))
plt.show()
```

---

## Exercises

- Create a histogram with 50 bins and overlay a KDE (kernel density estimate).
- Plot a boxplot comparing test scores of 3 different subjects.
- Make a scatter plot where points are colored by a third variable.
- Create a heatmap of a correlation matrix for a random DataFrame.
- Plot stock price time series data with annotations for key events.

---
