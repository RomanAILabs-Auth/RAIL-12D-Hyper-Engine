# RAIL-12D-Hyper-Engine
### The RomanAILabs 12D Adaptive Guidance System

![RomanAILabs](https://img.shields.io/badge/RomanAILabs-12D%20Engine-blue) ![Python](https://img.shields.io/badge/Python-3.12+-yellow) ![License](https://img.shields.io/badge/License-Proprietary%20%2F%20Reference-red)

**RAIL-12D-Hyper-Engine** is a single-file, zero-dependency reference implementation of a **12-Dimensional Dynamical Control System** designed for steering Large Language Models (LLMs) and complex trajectory engines.

Unlike static steering vectors, this engine is **self-regulating**. It uses the **RAIL-FPMF** (Flux-Potential-Memory Field) to monitor its own coherence and dynamically adjust its computational cost and steering strength in real-time.

---

## 🌌 The Breakthrough: Adaptive Hyper-Action

This engine moves beyond passive physics simulation into **Cybernetic Control**. It employs a composite invariant scalar ($H$) to couple geometry with coherence:

$$H(\sigma) = w_1 \cdot \text{FCES}(\sigma) + w_2 \cdot E_{mis}(\sigma) - w_3 \cdot \text{CI}(\sigma)$$

This allows the engine to autonomously shift between three compute tiers:

| Tier | Condition | Gradient Stride | Use Case |
| :--- | :--- | :--- | :--- |
| **LITE** | High Coherence ($CI \uparrow$) | Every 8 steps | Minimal CPU usage. Cruising mode. |
| **NORMAL** | Balanced State | Every 3 steps | Standard steering fidelity. |
| **FULL** | High Mismatch ($E_{mis} \uparrow$) | Every 1 step | Maximum compute. Emergency stabilization. |

---

## 🧠 Core Technologies

### 1. 12D Riemannian Geometry
A fully realized 12D manifold ($R^{12}$) separating **Spacetime Indices** (0-3) from **Internal Field Indices** (4-11). It calculates the **Riemann Curvature Tensor** ($R^i_{jkl}$) via finite differences to detect semantic loops and topological defects.

### 2. FCES (Field–Curvature Entanglement Scalar)
A diagnostic invariant that measures the "entanglement" between internal thought processes and output geometry.
> *Optimized "Fast Path" contraction avoids allocating the full $12^4$ tensor, making 12D curvature calculation negligible on consumer CPUs.*

### 3. White Hole Steering
A gravity-well attractor that uses the gradient of insight ($\nabla \text{FCES}$) to exert force on the model's trajectory, effectively "warping" it out of repetitive basins and toward higher-complexity states.

### 4. Warp Multiplier (FMPC Slot)
A "Time Dilation" mechanism. When the engine detects high complexity or confusion, it dynamically scales the effective time step ($dt$), allowing for higher-resolution processing of difficult tokens.

---

## 🚀 Usage

This is a pure Python implementation. No `pip install` required.

### Self-Test
Verify the physics engine and numerical stability:
```bash
python3 romanailabs_12d_engine_15of10_fpmf.py --selftest
