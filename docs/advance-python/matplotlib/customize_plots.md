# Chapter 3 — Customizing Plots in Matplotlib

Creating plots is only the first step. To make them **clear, attractive, and meaningful**, you need to customize them. Matplotlib offers a wide range of options to change how your plots look.

---

## Changing Line Styles and Colors

You can change the appearance of lines using color codes, line styles, and markers.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x), color="blue", linestyle="--", linewidth=2, marker="o", label="Sine")
plt.plot(x, np.cos(x), color="red", linestyle="-.", linewidth=2, marker="s", label="Cosine")

plt.legend()
plt.title("Custom Line Styles")
plt.show()
```

- **Colors**: `"red"`, `"blue"`, `"green"`, or short codes like `"r"`, `"b"`, `"g"`.  
- **Line styles**: `"-"` (solid), `"--"` (dashed), `"-."` (dash-dot), `":"` (dotted).  
- **Markers**: `"o"` (circle), `"s"` (square), `"^"` (triangle), `"*"`, etc.

---

## Titles, Labels, and Legends

Adding titles and labels makes plots understandable.

```python
x = np.linspace(0, 5, 50)
y = x ** 2

plt.plot(x, y, label="y = x²")
plt.title("Quadratic Function")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.legend(loc="upper left")
plt.show()
```

- `plt.title("...")` adds a title.  
- `plt.xlabel()` / `plt.ylabel()` add axis labels.  
- `plt.legend()` shows labels defined in `label=`.  
- The `loc` parameter controls legend placement.

---

## Adjusting Figure Size and Resolution

```python
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4), dpi=120)
plt.plot(x, y)
plt.title("Custom Figure Size & Resolution")
plt.show()
```

- `figsize=(width, height)` is in inches.  
- `dpi` controls resolution (dots per inch). Higher values = sharper plots.

---

## Adding Gridlines

```python
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("Grid Example")
plt.grid(True, linestyle="--", color="gray", alpha=0.7)
plt.show()
```

Gridlines make plots easier to read, especially for precise values.

---

## Fonts and Text Customization

```python
plt.plot(x, y)
plt.title("Custom Fonts", fontsize=16, fontweight="bold", color="darkblue")
plt.xlabel("Time", fontsize=12, style="italic")
plt.ylabel("Amplitude", fontsize=12)
plt.show()
```

You can change font size, weight, style, and color for titles and labels.

---

## Colormaps

Colormaps (or "cmap") are used in heatmaps, scatter plots, and contour plots.

```python
np.random.seed(0)
x = np.random.rand(100)
y = np.random.rand(100)
colors = np.random.rand(100)

plt.scatter(x, y, c=colors, cmap="viridis", s=80)
plt.colorbar(label="Color Intensity")
plt.title("Scatter Plot with Colormap")
plt.show()
```

Popular colormaps: `"viridis"`, `"plasma"`, `"inferno"`, `"coolwarm"`, `"cividis"`.

---

## Multiple Customizations Together

```python
t = np.linspace(0, 2*np.pi, 200)
s = np.sin(t)

plt.figure(figsize=(8, 5), dpi=100)
plt.plot(t, s, color="purple", linestyle="--", linewidth=2, marker="o", label="sin(t)")

plt.title("Beautifully Customized Plot", fontsize=14, fontweight="bold")
plt.xlabel("Angle (radians)")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle=":", color="gray", alpha=0.6)

plt.show()
```

This combines **line style, markers, title, labels, legend, and grid**.

---

## Key Takeaways

- Use colors, line styles, and markers for clarity.  
- Always label axes and add legends where needed.  
- Adjust figure size and DPI for reports and presentations.  
- Gridlines and fonts make your plots more readable.  
- Use colormaps to represent additional data dimensions.  
