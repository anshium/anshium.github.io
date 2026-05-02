---
layout: default
title: Endsem Solutions - POIS
---
# POIS Endsem Solutions

Welcome to the endsem solutions for POIS.

## Problem 1: Quantum Mechanics

Calculate the energy levels of a particle in a 1D box.

**Solution:**

The time-independent Schrödinger equation is given by:

$$ \hat{H}\psi = E\psi $$

For a particle in a 1D box of length $L$, the potential $V(x) = 0$ inside the box and $\infty$ outside. The wavefunction is:

$$ \psi_n(x) = \sqrt{\frac{2}{L}} \sin\left(\frac{n\pi x}{L}\right) $$

The corresponding energy levels are:

$$ E_n = \frac{n^2 h^2}{8mL^2} $$

## Problem 2: Example provided

This is inline math: \( a^2 + b^2 = c^2 \)

Block math:

$$
\int_0^1 x^2 dx
$$

## Problem 3: Code Snippet

Here is a simple python solution for calculating this:

```python
def calculate_energy(n, h, m, L):
    """Returns the energy of the n-th state."""
    return (n**2 * h**2) / (8 * m * L**2)
```

## Problem 4: Data Visualization

> "A picture is worth a thousand words."

![Sample Plot](https://via.placeholder.com/600x300.png?text=Placeholder+Image+for+Plot)

**GIF Demo:**

![Rotating cube](https://media.giphy.com/media/xT9IgzoKnwFNmISR8I/giphy.gif)

*Note: Replace these images with your actual plots/gifs.*
