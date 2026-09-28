# CS582 Machine Learning — Study Guide
## Lectures 1–3: Introduction · Perceptron · Multi-Layer Perceptron (MLP) + Labs 2 & 3

> Split from the ultimate study guide. See also the other parts in this `study/` folder.
>
> **Sources:** course slides, labs (incl. Lab 2 solutions), Final Practice, Marsland textbook Ch. 2–4.
> Every worked-example number was re-computed with code.

## How to use this part

1. Read all sections in order.
2. Each section ends with an **Exam angle** (in the original; check section ends).
3. After finishing Parts 1–3, use `04_QuestionBank_FormulaSheet_Traps.md` for practice + formula sheet + traps.

---

# PART 2 — LECTURE 2: NEURAL NETWORKS INTRO & THE PERCEPTRON

### 2.1 Artificial Neural Networks (ANN / NN)
- A **computational model based on the structure and functions of biological neural networks** in the brain (the most robust learning system we know). Also: ANN tries to *understand* natural biological systems through computational modeling.
- Information flowing through the network **changes the network's structure** — the network *learns* based on input and output.
- **Why ANN:**
  - **Massive parallelism** → computational efficiency.
  - **Distributed** representations (not "localist") → **robustness and graceful degradation**.
  - Intelligent behavior as an **emergent property** of many simple units — not explicit symbolic rules.
- ANN is a very popular, mostly-used ML approach, BUT there is a **large gap between ANNs and real neurons**.

### 2.2 The human brain (facts that can be asked)
- ~**1.4 kg** of water and "mush"; ~**10¹¹ neurons** and **10¹⁴ synapses** (average ~**10⁴ connections** per neuron).
- In computational terms: 10¹¹ simple, **slow** processors, but **massively parallel** and **fault tolerant**.
- **2 hemispheres** connected by the **corpus callosum** (bundle of nerve fibres).
- Lobes: **Occipital** = vision; **Frontal** = attention, short-term memory tasks, planning; **Temporal** = auditory, semantics, hippocampus; **Parietal** = integrates sensory info.
- Human info processing: i/o (visual, auditory, haptic/touch, movement), memory (sensory, short-term, long-term), processing (reasoning, problem solving, skill, error); emotion influences capabilities; each person is different.

### 2.3 Real neurons & neural communication
- Structure: **cell body (soma), dendrites (inputs), axon (output), synaptic terminals.**
- Electrical potential across the membrane exhibits **spikes = action potentials**.
- A spike originates in the cell body → travels down the axon → causes synaptic terminals to release **neurotransmitters** → chemicals diffuse across the **synapse** to dendrites of other neurons.
- Neurotransmitters can be **excitatory or inhibitory**.
- If the **net input is excitatory and exceeds a threshold**, the neuron **fires** an action potential.
- Action-potential graph: resting ≈ **−70 mV**, threshold ≈ **−55 mV**, peak ≈ **+40 mV**; depolarization → repolarization → **refractory period**; weak stimuli = "failed initiations."

### 2.4 How does the brain learn? — Hebb's Rule
- What changes? The **strength of synaptic connections**.
- **Hebb's rule:** *If two neurons connected by a synapse fire simultaneously, the synapse strengthens.* (Optionally: if they don't fire simultaneously, it weakens.)

### 2.5 Hodgkin–Huxley model
"One way to make an intelligent computer is to model neurons": take a real neuron (giant squid), measure chemical concentrations, monitor the membrane (action) potential, write down the differential equations — win the Nobel prize. (Too detailed to be practical for computing → we simplify.)

### 2.6 Speed constraints — why the brain must be parallel
- Neurons switch in **milliseconds**; computers in **nanoseconds**.
- Yet the brain does vision/speech understanding in **tenths of a second** → only time for **~100 serial steps**.
- ⇒ The brain **must exploit massive parallelism**.

### 2.7 History timeline
| Year | Event |
|---|---|
| 1943 | **McCulloch & Pitts** neuron (the slides say the perceptron based on it was "introduced in 1949") |
| 1950s (1958) | **Perceptron** (Rosenblatt) — learning for simple single-layer networks |
| 1962 | Rosenblatt's **Perceptron Convergence** proof |
| 1969 | **Minsky & Papert, "Perceptrons"** — showed what perceptrons can't learn (XOR) → NN research stalled ~20 years; symbolic AI dominated |
| 1986 | **Backpropagation** (Rumelhart, Hinton, Williams/McClelland) → multi-layer networks. "It took over 30 years to develop learning algorithms for multi-layer NN." |

### 2.8 The McCulloch–Pitts neuron (first artificial neuron)

```
 x1 ──w1──┐
 x2 ──w2──┤──► Σ ──► h ──► [threshold θ] ──► o
  ⋮       │
 xm ──wm──┘
```

$$h = \sum_{i=1}^{m} w_i x_i, \qquad o = \begin{cases}1 & h \ge \theta\\ 0 & h < \theta\end{cases}$$

- "Greatly simplified neuron": fires if the weighted sum is above the threshold.
- **Weights:** positive = **excitatory**, negative = **inhibitory**.
- **Simplifications** vs real neurons: only a **linear sum** of inputs; **no refractory period**; a **single output** instead of a pulse (**spike train**); works on the **computer clock**.
- Put lots of them together, connect as we like → assemblies are capable of **universal computation** (anything a normal computer can do). We just need the right **parameters (weights and thresholds)**.
- **A single neuron cannot do much** → we need networks.
- **Where is the learning?** Mostly in the **weights** — the model of the **synapse** — i.e., *between* the neurons, not in the neurons.

### 2.9 Perceptrons as logic gates (slides + Lab 2 Q1 & Q2)

Slide rules (threshold $T_j$, n inputs):
- **AND:** let all weights $= T_j/n$ (need all n inputs on to reach T).
- **OR:** let all weights $= T_j$ (any single input reaches T).
- **NOT:** threshold 0, single input with a **negative** weight.
- With such gates you can build arbitrary logic circuits, sequential machines, computers. **Given negated inputs, a two-layer network can compute ANY boolean function** (two-level AND-OR network).

#### ✅ Lab 2 Q1 (solution): $w_1 = 1, w_2 = 1, b = -1.5$, threshold activation

Output $= 1$ if $w_1x_1 + w_2x_2 + b > 0$:

| $x_1$ | $x_2$ | $h = x_1 + x_2 - 1.5$ | output |
|---|---|---|---|
| 0 | 0 | −1.5 | 0 |
| 0 | 1 | −0.5 | 0 |
| 1 | 0 | −0.5 | 0 |
| 1 | 1 | +0.5 | **1** |

- ⇒ It is an **AND gate**. (Instructor's note: *with a different bias it could be a different gate* — e.g., $b = -0.5$ gives OR.)
- **Discriminant function (decision boundary):** $w_1x_1 + w_2x_2 + b = 0$ → $x_1 + x_2 = 1.5$ → $\mathbf{x_2 = 1.5 - x_1}$.
- Drawing: a straight line with slope −1 crossing both axes at 1.5; only the point (1,1) lies above it.

```
x2
1.5 \
 1  o\    ●(1,1) → output 1
     \
 0  o  \o ──── x1        o = output 0
    0   1 1.5
```

#### ✅ Lab 2 Q2 (solution): NOT, NAND, NOR

| Gate | Weights | Bias | Check |
|---|---|---|---|
| **NOT** | $w = -1$ | $b = 0.5$ (or threshold 0 with "fires if ≥ 0") | x=0 → 0.5 > 0 → **1**; x=1 → −0.5 → **0** |
| **NAND** | $w_1 = w_2 = -1$ | $b = +1.5$ | (0,0)→1.5→1; (0,1),(1,0)→0.5→1; (1,1)→−0.5→**0** |
| **NOR** | $w_1 = w_2 = -1$ | $b = +0.5$ | (0,0)→0.5→**1**; (0,1),(1,0)→−0.5→0; (1,1)→−1.5→0 |

- Instructor's handwritten answer: NOT: $w_1 = -1$ with "1×(−1) < 0 → 0, 0×(−1) = 0 → 1" (i.e., fires when $h \ge 0$). **NAND (2 ways):** use **AND followed by a NOT**, or directly $w_1=w_2=-1$, bias $=1.5$. **NOR:** $w_1=w_2=-1$, bias $=0.5$.
- **Pattern to remember:** NAND/NOR = negate the AND/OR weights and bias. (AND: 1,1,−1.5 → NAND: −1,−1,+1.5. OR: 1,1,−0.5 → NOR: −1,−1,+0.5.)

### 2.10 The Perceptron network
- A Perceptron = **a collection of McCulloch–Pitts neurons + a set of inputs + weights connecting them.** Single layer of weights.
- **Input nodes are NOT neurons** — just a way of showing values fed in (the number of them = dimension of the input vector).
- **Neurons are independent of each other** — each decides using only its own weights and threshold; they share only the inputs.
- Notation: $w_{ij}$ = weight from **input i** to **neuron j** (e.g., $w_{32}$ connects input 3 to neuron 2). m inputs, n neurons. Output pattern is a vector of 0s and 1s, compared to the **target**.

### 2.11 Perceptron learning rule

Error (slides): $E = t - y$. Change weights to minimize it:
$$w_{ij} \leftarrow w_{ij} + \Delta w_{ij}, \qquad \boxed{\Delta w_{ij} = \eta\,(t_j - y_j)\,x_i}$$

Book's equivalent form: $\boxed{w_{ij} \leftarrow w_{ij} - \eta\,(y_j - t_j)\,x_i}$ — **same thing, just sign flipped twice.**

**Why this rule? (explain in words)**
- If the neuron fired when it shouldn't ($y=1, t=0$): $y - t = +1$ → weights are **too big** → subtract.
- If it didn't fire but should ($y=0,t=1$): $y - t = -1$ → weights **too small** → add.
- Multiply by $x_i$ because the input could be negative (which would flip what "increase" means), and if $x_i = 0$ that weight did not contribute, so it isn't changed.
- If correct, $y - t = 0$ → **no change**.

**Learning rate η:**
- Controls how much weights change. η = 1 (or missing) → weights change a lot → **unstable**, never settles.
- Small η → more stable, **resistant to noise**, but slower.
- Typical: **0.1 < η < 0.4**. (For the perceptron itself η doesn't change *whether* it converges — the convergence proof uses η=1 — but it matters a lot for other algorithms.)

### 2.12 The bias input (handling the threshold)
- Problem: if all inputs are 0, changing weights does nothing — only the **threshold** can decide. The threshold must be **adjustable/learnable**.
- **Trick:** fix the threshold at **0** and add an **extra input fixed at −1** (the **bias node**) with its own weight $w_{0j}$. That weight is learned like any other.
- The slide's diagram shows a "−1" node connected to all output neurons.
- Then: fire if $\sum_{i=0}^{m} w_{ij} x_i > 0$ where $x_0 = -1$. Equivalently $\sum_{i=1}^m w_{ij}x_i > w_{0j}$ → **$w_{0j}$ acts as the threshold θ**.
- In code: `inputs = np.concatenate((inputs, -np.ones((self.nData,1))), axis=1)`

### 2.13 The Perceptron Algorithm (write this from memory)

```
Initialisation:
    set all weights w_ij to small random numbers (positive and negative)
Training:
    for T iterations or until all outputs are correct:          # each pass = 1 EPOCH
        for each input vector:
            compute activation of each neuron j:
                y_j = g( Σ_{i=0..m} w_ij x_i ) = 1 if Σ w_ij x_i > 0 else 0
            update each weight:
                w_ij ← w_ij − η (y_j − t_j) x_i
Recall:
    y_j = 1 if Σ_i w_ij x_i > 0 else 0
```

- Slide wording: *Initialize weights to random values. Until outputs of all training examples are correct: for each training pair E, compute current output, compare to target, update weights and threshold using the learning rule.* **Each execution of the outer loop is an epoch.**
- Output as a sign function: $y_j = \text{sign}\left(\sum w_{ij}x_i\right)$ → fire if $\vec{w}\cdot\vec{x} \ge 0$.
- **Complexity:** recall $O(mn)$; training $O(Tmn)$.

### 2.14 Hand-worked example: learning OR (book §3.3.4) — PRACTISE THIS

Setup: bias input $x_0=-1$; initial $w_0=-0.05,\; w_1=-0.02,\; w_2=0.02$; $\eta = 0.25$; fire if $h > 0$.
Update: $w_i \leftarrow w_i - \eta(y-t)x_i$.

**Epoch 1**
| input | $h = -w_0 + w_1x_1 + w_2x_2$ | y | t | update? | new $(w_0, w_1, w_2)$ |
|---|---|---|---|---|---|
| (0,0) | $0.05$ | 1 | 0 | yes: $w_0 = -0.05 - 0.25(1)(-1) = 0.2$ | (0.20, −0.02, 0.02) |
| (0,1) | $-0.2 + 0.02 = -0.18$ | 0 | 1 | yes: $w_0 = 0.2 - 0.25(-1)(-1) = -0.05$; $w_2 = 0.02 + 0.25 = 0.27$ | (−0.05, −0.02, 0.27) |
| (1,0) | $0.05 - 0.02 = 0.03$ | 1 | 1 | no | (−0.05, −0.02, 0.27) |
| (1,1) | $0.05 - 0.02 + 0.27 = 0.30$ | 1 | 1 | no | (−0.05, −0.02, 0.27) |

**Epoch 2**
| input | h | y | t | new weights |
|---|---|---|---|---|
| (0,0) | 0.05 | 1 | 0 | (0.20, −0.02, 0.27) |
| (0,1) | 0.07 | 1 | 1 | — |
| (1,0) | −0.22 | 0 | 1 | (−0.05, 0.23, 0.27) |
| (1,1) | 0.55 | 1 | 1 | — |

**Epoch 3:** (0,0): h=0.05 → wrong → $w=(0.20, 0.23, 0.27)$; the rest are correct.
**Epoch 4:** (0,0): h = −0.2 → 0 ✓; (0,1): 0.07 ✓; (1,0): 0.03 ✓; (1,1): 0.30 ✓ → **no changes → converged.**

Final: $w_0 = 0.2$ (threshold), $w_1 = 0.23$, $w_2 = 0.27$. Decision boundary: $0.23x_1 + 0.27x_2 = 0.2$.

> Note: many weight sets solve the problem; which one you get depends on η, input order and initial weights. We only care that it works and **generalizes**.

### 2.15 Implementation in Python (NumPy) — slides + book

**Loop version (recall):**
```python
for data in range(nData):              # loop over input vectors
    for n in range(N):                 # loop over neurons
        activation[data][n] = 0
        for m in range(M+1):           # +1 for the bias node
            activation[data][n] += weight[m][n] * inputs[data][m]
        activation[data][n] = 1 if activation[data][n] > 0 else 0
```

**Matrix version** — shapes: inputs $N \times (m+1)$, weights $(m+1)\times n$, activations/targets $N \times n$:
```python
# Forward pass / recall
activations = np.dot(inputs, self.weights)
return np.where(activations > 0, 1, 0)

# Training (BATCH): transpose inputs to (m+1) x N
self.weights -= eta*np.dot(np.transpose(inputs), self.activations - targets)

# Add the bias inputs
inputs = np.concatenate((inputs, -np.ones((self.nData,1))), axis=1)

# Initial weights: small random in [-0.05, 0.05]
weights = np.random.rand(nIn+1, nOut)*0.1 - 0.05
```
- `np.dot` = matrix multiply (inner dims must match: $(m\times n)(n\times p)$). `np.where(cond, x, y)` = elementwise choose. `np.transpose` swaps rows/columns.
- Running OR: `p = pcn.pcn(inputs,targets); p.pcntrain(inputs,targets,0.25,6)` → weights stop changing after a few iterations, outputs [0,1,1,1].

#### ✅ Lab 2 Q4 (solution): convert batch → sequential and compare

**Batch** (book code): all inputs go forward, total error computed, weights updated **once per epoch**.
**Sequential** (the original algorithm): weights updated **after every single input**.

Instructor's approach: *minimal change — feed one row (1 × m) instead of N × m at a time.*

```python
def pcntrain_seq(self, inputs, targets, eta, nIterations):
    inputs = np.concatenate((inputs, -np.ones((self.nData,1))), axis=1)
    for n in range(nIterations):
        for m in range(self.nData):
            x = inputs[m:m+1, :]        # one input vector, shape (1, nIn+1)
            t = targets[m:m+1, :]       # its target,       shape (1, nOut)
            y = self.pcnfwd(x)
            self.weights -= eta*np.dot(np.transpose(x), y - t)
```
(Tested: both versions learn OR with 100% accuracy.)

**Comparison (what to write):**
| Batch | Sequential |
|---|---|
| One update per epoch, direction "most inputs want" | One update per input — weights "pulled around" by each input |
| More accurate estimate of the gradient; converges to (local) minimum faster; easy with matrices | Simpler to program with loops; **order matters** → shuffle inputs each epoch |
| Can get stuck in local minima | Noisier → may **escape local minima** |
| Final weights often differ between the two, but both solve linearly separable problems | |

### 2.16 Perceptron as a Linear Separator
- Because it uses a **linear threshold function**, the perceptron searches for a **linear separator** (line in 2D, plane in 3D, **hyperplane** in n-D) = the **decision boundary / discriminant function**.
- Neuron fires if $\mathbf{x}\cdot\mathbf{w}^T \ge 0$. Boundary: $\mathbf{x}\cdot\mathbf{w}^T = 0$.

**Proof that w is perpendicular to the boundary:** take two points $x_1, x_2$ on the boundary: $x_1\cdot w = 0 = x_2\cdot w \Rightarrow (x_1 - x_2)\cdot w = 0$. Since $a\cdot b = \|a\|\|b\|\cos\theta$ and neither vector is zero, $\cos\theta = 0$ → $\theta = 90°$. $x_1 - x_2$ lies **along** the boundary, so **w is perpendicular to the decision boundary.**

- Distance from a point $x'$ to the hyperplane $w^Tx + b = 0$: $\dfrac{|w^Tx' + b|}{\|w\|}$ (book problem 3.7).
- **Several output neurons → several lines**, each separating different parts of the space (Fig 3.8).
- **Linearly separable** = there exists a straight line/hyperplane separating the classes.

### 2.17 What the Perceptron CANNOT learn: XOR

| $x_1$ | $x_2$ | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

- **No single straight line** separates {(0,1),(1,0)} from {(0,0),(1,1)} → **not linearly separable**.
- Also can't learn **parity** functions in general.
- Running the code: weights **cycle** between two wrong solutions forever (iterations 11–14 alternate between the same two weight vectors, outputs all 0). Running longer doesn't help.

**Algebraic proof XOR is impossible (good exam answer):** need $w_0$ (threshold) such that
(0,0)→0: $0 \le w_0$ ; (1,1)→0: $w_1 + w_2 \le w_0$ ; (0,1)→1: $w_2 > w_0$ ; (1,0)→1: $w_1 > w_0$.
Adding the last two: $w_1 + w_2 > 2w_0 \ge w_0$ (since $w_0 \ge 0$) — contradicts $w_1 + w_2 \le w_0$. ∎

**Two solutions (slide):**
1. **Make the network more complicated** → add layers → **MLP** (Lecture 3).
2. **Make the input more complicated** → add a dimension.
   - Add $x_3$ that is 1 only for (0,0): inputs (0,0,1),(0,1,0),(1,0,0),(1,1,0) → now **a plane separates the classes in 3D** (Fig 3.10). The perceptron then learns outputs 0,1,1,0 ✓.
   - Or add $x_1 \times x_2$ as a third feature (Fig 3.11).
   - General insight: **it is always possible to separate two classes linearly if you project the data into the right set of dimensions** → basis of **kernel methods / SVMs** (later lecture).

### 2.18 Perceptron limits
- A system cannot learn concepts it **cannot represent**.
- **Minsky & Papert (1969)** analyzed the perceptron and showed many functions it can't learn → discouraged NN research; **symbolic AI** became dominant (~20 years).

### 2.19 Perceptron Convergence & Cycling Theorems
- **Convergence theorem:** *If the data is linearly separable* (so a consistent set of weights exists), the perceptron algorithm **will eventually converge** to a consistent set of weights — in a **finite** number of updates, bounded by $1/\gamma^2$ (with $\|x\|\le 1$), where **γ = the margin** = distance from the separating hyperplane to the closest data point.
- **Cycling theorem:** *If the data is NOT linearly separable*, the algorithm will eventually **repeat** a set of weights and threshold at the end of some epoch → **infinite loop**.
- ⇒ By checking for repeated weights, you can guarantee termination with a positive or negative answer.
- The perceptron stops as soon as all training data is correct → **no guarantee of the largest margin** (SVMs do that).

#### ✅ Lab 2 Q3 (solution): Prove convergence with $\|x\| \le R$ (instead of $\|x\| \le 1$)

**Setup.** Labels $y \in \{-1, +1\}$, η = 1. Since the data is linearly separable, there is a unit vector $w^*$ ($\|w^*\| = 1$) with $y\,(w^*\cdot x) \ge \gamma > 0$ for every data point (γ = margin). Start from $w^{(0)} = 0$. Goal: make $w$ as parallel to $w^*$ as possible → show (a) $w^*\cdot w$ grows fast, (b) $\|w\|$ doesn't grow too fast. (Because $w^*\cdot w = \|w^*\|\|w\|\cos\theta$, we want θ → 0 while $\|w\|$ stays controlled.)

Suppose at step t the network gets input x wrong: $y\,(w^{(t-1)}\cdot x) < 0$ (actually ≤ 0). Update: $w^{(t)} = w^{(t-1)} + yx$.

**(a) Lower bound.**
$$w^*\cdot w^{(t)} = w^*\cdot w^{(t-1)} + y\,(w^*\cdot x) \;\ge\; w^*\cdot w^{(t-1)} + \gamma$$
After t updates: $w^*\cdot w^{(t)} \ge t\gamma$. By Cauchy–Schwarz, $w^*\cdot w^{(t)} \le \|w^*\|\|w^{(t)}\| = \|w^{(t)}\|$, so
$$\|w^{(t)}\| \ge t\gamma.$$

**(b) Upper bound.**
$$\|w^{(t)}\|^2 = \|w^{(t-1)} + yx\|^2 = \|w^{(t-1)}\|^2 + y^2\|x\|^2 + 2y\,(w^{(t-1)}\cdot x)$$
- $y^2 = 1$
- $\|x\|^2 \le R^2$ ← **this is the only change from the book's proof**
- $2y\,(w^{(t-1)}\cdot x) \le 0$ because the point was **misclassified**

$$\Rightarrow \|w^{(t)}\|^2 \le \|w^{(t-1)}\|^2 + R^2 \;\Rightarrow\; \|w^{(t)}\|^2 \le tR^2 \;\Rightarrow\; \|w^{(t)}\| \le \sqrt{t}\,R$$

**(c) Combine.**
$$t\gamma \le \|w^{(t)}\| \le \sqrt{t}\,R \;\Rightarrow\; \sqrt{t} \le \frac{R}{\gamma} \;\Rightarrow\; \boxed{t \le \frac{R^2}{\gamma^2}}$$

So the perceptron makes **at most $R^2/\gamma^2$ updates** → it converges in finite time. With R = 1 this reduces to the book's $t \le 1/\gamma^2$.

Notes:
- The instructor's handwritten solution follows exactly these steps (he writes the upper bound as $\le \|w^{(t-1)}\|^2 + y^2R^2 = k$ and ends with $t \le 1/\gamma^2$ for the normalised case). He also notes: *"W(t−1) should be W(t), but since the difference between t and t−1 after long iterations is small, the proof is still correct."*
- **Interpretation:** bigger margin γ → faster convergence; larger data spread R → slower. Scaling the data scales R and γ together, so the ratio R/γ is what matters.
- The book says "the network made an error, so $w^{(t-1)}$ and $x$ are perpendicular" — the precise reason is that the cross term $2y\,w\cdot x$ is **≤ 0** because of the error.

### 2.20 Perceptron performance
- Linear threshold functions are **restrictive (high bias)** but still reasonably expressive — more general than **pure conjunctive** (AND of features), **pure disjunctive** (OR), and **M-of-N** (at least M of N features present).
- Converges **fairly quickly** for linearly separable data.
- Can use even **incompletely converged** results when only a few outliers are misclassified.
- Experimentally does quite well on many benchmark datasets.

### 2.21 ✅ Lab 2 Q5: Pima Indians dataset (book §3.4.4)

**Data:** 768 data points × 9 columns — 8 measurements of Pima Indian women in Arizona; column 8 = class (diabetes yes/no). (The lab notes Pima is no longer on UCI, so any UCI dataset can be used — the method is the same.)

**Code:**
```python
import numpy as np, pylab as pl, pcn
pima = np.loadtxt('pima-indians-diabetes.data', delimiter=',')
np.shape(pima)                       # (768, 9)

# plot 2 features, classes in different markers
indices0 = np.where(pima[:,8]==0); indices1 = np.where(pima[:,8]==1)
pl.plot(pima[indices0,0], pima[indices0,1], 'go')
pl.plot(pima[indices1,0], pima[indices1,1], 'rx'); pl.show()

p = pcn.pcn(pima[:,:8], pima[:,8:9])          # pima[:,:8] = all rows, cols 0..7 (inputs)
p.pcntrain(pima[:,:8], pima[:,8:9], 0.25, 100) # pima[:,8:9] = all rows, col 8 (target, kept 2-D)
p.confmat(pima[:,:8], pima[:,8:9])

# fairer: even rows train, odd rows test
trainin = pima[::2,:8];  testin  = pima[1::2,:8]
traintgt = pima[::2,8:9]; testtgt = pima[1::2,8:9]
```

**Preprocessing (§3.4.5) — this is what improves results:**
```python
# normalise inputs (zero mean, unit variance) BEFORE splitting into train/test
pima[:,:8] = (pima[:,:8] - pima[:,:8].mean(axis=0)) / pima[:,:8].var(axis=0)
# domain-based: cap pregnancies at 8, quantise age into ranges
pima[np.where(pima[:,0]>8),0] = 8
pima[np.where(pima[:,7]<=30),7] = 1
pima[np.where((pima[:,7]>30) & (pima[:,7]<=40)),7] = 2
# ... etc.
```

**Observations to write (expected results):**
- Plots of any 2 features: the classes **overlap heavily → not linearly separable**.
- Raw data: roughly **50–70% accuracy**, **unstable** between runs (sometimes ~30% — worse than chance).
- Testing on the training data is **unfair** → use a separate test set.
- **Normalisation + quantisation** (and **feature selection**: drop features one at a time and keep the drop if results improve) → much better, more stable results.
- Conclusion: a single-layer perceptron is limited on real non-linearly-separable data → motivates the MLP.
- Reminder: normalise the **whole dataset before splitting**, otherwise the same point would be scaled differently in train and test.

> **Exam angle (Lecture 2):** McCulloch–Pitts equation; perceptron learning rule + why it works; bias trick; hand-trace an epoch; design AND/OR/NOT/NAND/NOR; draw the decision boundary; why XOR fails and the two fixes; convergence vs cycling theorem; proof sketch with R; batch vs sequential.

---
