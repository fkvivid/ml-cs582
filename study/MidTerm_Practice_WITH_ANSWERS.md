# Sample MidTerm Questions — with answers

> From `MidTerm_Practice.docx`. Use for checking your work sheet.

---

## True or False — justify

**1. A perceptron will converge for a linearly separable problem**

**T** — a perceptron can learn any linearly separable function (convergence theorem).

**2. Perceptron can learn X-NOR Gate**

**F** — X-NOR is like XOR; it is **not** linearly separable.

**3. The Backpropagation method is mainly based on Gradient Descent**

**F** — BP **uses** gradient descent, but its **main idea** is to **propagate errors back** through the network.

---

## Neural nets

**4. Step-function net: multiply all weights and thresholds by a constant. Does behavior change?**

For a **hard step / threshold** unit, scaling weights **and** the threshold by the **same positive** constant leaves the comparison \(w\cdot x \gtrless \theta\) unchanged (both sides scale). So behavior **does not change** (for that positive scale). (Sign flip or zero scale would change things — assume same nonzero positive constant as intended in class.)

**5. Back Propagation of Errors** — network from handout #2; \(t_k=[0.7,\ 0.8]\); sigmoid.

Handout setup: inputs \(0.5\) (left), \(1.0\) (right). Ignore circled output values; recompute with sigmoid. \(\eta=0.25\).

### Forward pass (handout numbers)

- Left hidden: net \(= 0.5\times(-1)+1\times 0=-0.5\) →  
  \(\sigma(-0.5)=1/(1+e^{0.5})\approx 0.384\)
- 2nd hidden: \(\sigma(2)\approx 0.88\)
- 3rd hidden: \(\sigma(0)=0.50\)

Left output net \(\approx 2(0.384)+(-0.5)(0.88)+0.5=0.828\) → \(y\approx 0.69\) (target \(0.7\))

Right output net \(\approx -2(0.384)+(0.88)(1)+(0.5)(0.5)=0.362\) → \(y\approx 0.59\) (target \(0.8\))

### Backward pass

Output delta (sigmoid): \(\delta_k=(y_k-t_k)\,y_k(1-y_k)\)

- Left out: \((0.69-0.7)(0.69)(0.31)\approx -0.002\)
- Right out: \((0.59-0.8)(0.59)(0.41)\approx -0.05\) *(handout)*

Hidden (left unit, \(a_j=0.384\)), weights to outs \(2\) and \(-2\):

\[
\delta_j=a_j(1-a_j)\big[\delta_{\text{left}}(2)+\delta_{\text{right}}(-2)\big]
\approx 0.384(0.616)\big[(-0.002)(2)+(-0.05)(-2)\big]\approx 0.0236
\]

*(Handout then multiplies into weight update; their sample update for one \(W_{jk}\) mixes terms — treat the forward/delta structure as the exam pattern.)*

Sample weight update style in handout:
\[
W\leftarrow W-\eta\,\delta\,a
\]
e.g. one illustrated update ends near \(2-0.0059=1.9941\). Same approach for other weights.

---

## Why Machine Learning?

Key reasons (handout):

- Algorithms do **not** exist for many data-driven apps (e.g. self-driving).
- Algorithms may exist but are **slow / inaccurate** for complex apps.
- **Big data**: even if algorithms exist, they can be inefficient at that scale.
- **Automation** of programming / configuring / replacing blocks via ML.

---

## Topics to also be ready for (handout lists; answers in study Parts 1–5)

- **SVM** — margin, SVs, soft margin \(C\), dual, kernels  
- **Dimensionality reduction** — LDA vs PCA  
- **Training / validation / testing**, learning parameters, **VC dimension**

---

## Q & A

Use class Q&A + `study/04_…` and `study/07_…` question banks.
