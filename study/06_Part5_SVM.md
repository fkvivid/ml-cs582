# CS582 Machine Learning — Ultimate Study Guide
## Lesson 5: Support Vector Machines (+ Lab 5)

> **Sources:** `5-Support Vector Machines.ppt`, `Lab_5_SVM.docx` (= Marsland Problems 8.1–style), Marsland Ch. 8.
>
> Lab 5 numbers (margin, support vectors, \(\Phi_1\), circle lift) were recomputed in code.
>
> **If you only read this file for Lesson 5, you should be able to answer the exam.**

---

## How to use

1. Read §§1–8 (intuition → margin math → dual → kernels → algorithm → multi-class → pros/cons).
2. Work Lab 5 solutions in §9 **by hand** once.
3. Memorize the formula box and traps.
4. Drill `07_L4L5_QuestionBank_FormulaSheet.md`.

Expect on the exam: **why SVM beats “any” Perceptron line, margin & support vectors, \(\min \tfrac12 w^Tw\) with \(t_i(w^Tx_i+b)\ge 1\), dual \(w^*=\sum\lambda_i t_i x_i\), kernel trick, name 3 kernels, advantages (global min, max margin, sparse SVs), disadvantages (kernel choice, scale, binary), multi-class one-vs-rest.**

---

# 1. Setup — link to Perceptron & VC

- Perceptron cannot separate **2D XOR**, but can if you **add a dimension** (raise effective capacity / change representation).
- SVM (and kernel methods) use that insight: **map data so classes become linearly separable**.
- Hard part: **which new dimensions?** → **kernels**.
- Introduced by **Vapnik (1992)**; popular because strong results on **reasonably sized** data.
- **Does not scale well** to huge training sets (QP cost grows badly with \(n\)).
- SVM also reformulates classification so we can say which of two “perfect training” lines is **better** → **maximum margin**.
- Built on **VC theory** ideas (capacity + margin → better generalization).

---

# 2. Optimal separation — margin & support vectors

### Many lines can “work”
Figure: three different lines all separate the same two classes. Perceptron may stop at any of them. Which is best?

**Prefer the line through the middle of the gap** — not hugging either class (Goldilocks).

### Margin \(M\)
- Measure distance **perpendicular** from the decision line until you hit a datapoint.
- Imagine a “no-man’s land” (strip / cylinder / hypercylinder) around the line with no training points inside.
- Largest such radius = **margin \(M\)**.
- Middle separator has the **largest margin** → **maximum margin (linear) classifier**.

### Support vectors
- Datapoints in each class **closest** to the boundary (on the margin edges).
- They alone determine the boundary.
- After training you can **throw away** all other training points (sparse model).

### Classifier equation
\[
y = w\cdot x + b = w^Tx + b
\]
- \(y > 0\) → one class (“+”); \(y < 0\) → other (“◦”); \(y=0\) → decision boundary.
- With margin: require \(|y|\) at least \(M\) for classified points (soft wording on slides).
- Pick support vector \(x_+\) on the “+” margin face: \(w^Tx_+ = M\).

### Key geometry
- \(w\) is **perpendicular** to the decision boundary.
- Width of margin \(\propto 1/\|w\|\) (with unit \(w/\|w\|\): \(\boxed{M = 1/\|w\|}\)).
- Some texts call “margin” the **full** gap between the two parallel faces (= \(2/\|w\|\)). Know which convention you use.

**Therefore:** maximize \(M\) ⇔ **minimize \(w^Tw\)** (or \(\tfrac12 w^Tw\)).

Two goals at once:
1. Classify correctly, and  
2. Make \(w^Tw\) as small as possible.

---

# 3. Constrained optimization (primal)

### Labels
Use targets \(t_i \in \{-1,+1\}\) (not 0/1). Then \(t_i y_i > 0\) means correct.

### Constraint
\[
t_i\,(w^T x_i + b) \ge 1 \quad\text{for all } i=1,\ldots,n
\]
(The “1” is a convenient scaling of the margin faces.)

### Primal problem (Eq. 8.1 — MEMORIZE)
\[
\boxed{\min_{w,b}\; \tfrac12\, w^T w \quad\text{subject to}\quad t_i(w^T x_i + b)\ge 1\;\forall i}
\]

- Quadratic objective + linear constraints → **convex quadratic program** → **unique global minimum** (unlike MLP local minima).
- Gradient descent struggles with constraints; use a **QP solver**.

### Soft margin (non-separable data)
Allow mistakes with slack \(\eta_i\ge 0\):
\[
t_i(w^Tx_i+b)\ge 1-\eta_i
\]
Minimize \(w^Tw + C\sum\eta_i\):
- **Small \(C\)** → prefer large margin (tolerate more errors)
- **Large \(C\)** → prefer few errors (smaller margin OK)

Dual then has box constraints \(0\le\lambda_i\le C\).

---

# 4. Dual solution (what you use in practice)

Using Lagrange multipliers \(\lambda_i\ge 0\) and KKT conditions:

\[
\boxed{w^* = \sum_{i=1}^{n}\lambda_i\, t_i\, x_i},\qquad \sum_{i=1}^{n}\lambda_i t_i = 0
\tag{8.8}
\]

- \(\lambda_i \neq 0\) **only for support vectors** (others can be discarded).
- Sparse representation of the data.

**Bias (average over \(N_s\) support vectors):**
\[
b^* = \frac{1}{N_s}\sum_{j\in\mathrm{SV}}\Biggl(t_j - \sum_{i=1}^{n}\lambda_i t_i\, x_i^T x_j\Biggr)
\tag{8.10}
\]

**Classify new point \(z\):**
\[
w^{*T}z + b^* = \Biggl(\sum_{i}\lambda_i t_i x_i\Biggr)^T z + b^*
\tag{8.11}
\]
⇔ only **inner products with support vectors**.

**Dual objective (maximize over \(\lambda\)):**
\[
\max_\lambda \sum_i\lambda_i - \tfrac12\sum_{i,j}\lambda_i\lambda_j t_i t_j\, x_i^T x_j
\]
with \(\lambda_i\ge 0\), \(\sum\lambda_i t_i=0\).

With kernel \(K\) and soft margin:
\[
\max_\lambda \sum_i\lambda_i - \tfrac12\lambda^T(tt^T\odot K)\lambda,\quad 0\le\lambda_i\le C,\quad\sum\lambda_i t_i=0
\tag{8.24–8.25}
\]

---

# 5. Kernels & the kernel trick (THE big idea)

### Linear SVM is not enough
Eqs 8.8–8.11 are for **linear** problems. Real data is often nonlinear → **kernel trick**.

### Same idea as XOR fix
Add / transform features so data become linearly separable in a new space (Fig. 8.5: wavy boundary in 2D → straight line after transform).

### Feature map \(\phi\)
Replace \(x\) by \(\phi(x)\) in the dual. Prediction becomes:
\[
\sum_i\lambda_i t_i\,\phi(x_i)^T\phi(z) + b
\tag{8.16}
\]

### Kernel trick
Computing \(\phi(x_i)^T\phi(x_j)\) in high dimension is expensive. A **kernel**
\[
\boxed{K(x_i,x_j) = \phi(x_i)^T\phi(x_j)}
\]
gives the **same inner product without building \(\phi\) explicitly**.

Example (polynomial degree 2): \(\Phi\) has \(\sim d^2/2\) features, but
\[
\Phi(x)^T\Phi(y) = (1 + x^Ty)^2
\]
— only \(O(d)\) work in the original space.

**Mercer:** symmetric **positive definite** functions are valid kernels; kernels can be combined.

### Three standard kernels (MEMORIZE)

| Name | Formula | Notes |
|------|---------|--------|
| **Polynomial** (deg \(s\)) | \(K(x,y)=(1+x^Ty)^s\) | \(s=1\) → **linear** |
| **Sigmoid** | \(K(x,y)=\tanh(\kappa x^Ty - \delta)\) | NN-like |
| **RBF / Gaussian** | \(K(x,y)=\exp\!\big(-(x-y)^2/(2\sigma^2)\big)\) | also used in RBF nets |

Choose kernel + parameters by **validation** (like MLP). VC theory exists but practice = experiment.

### Geometric example (slides): concentric circles
\[
T([x_1,x_2]) = [x_1,\, x_2,\, x_1^2+x_2^2]
\]
Inner ring stays low in \(z=x_1^2+x_2^2\); outer ring lifts up → **plane** separates in 3D.

### Book XOR with quadratic kernel
Targets ±1; dual solved algebraically → decision \(x_1 x_2 = 0\); margin \(\sqrt{2}\). Computations stay in 2D even if feature space is 6D.

---

# 6. SVM algorithm (high level)

1. **Init:** build kernel matrix \(K\) between training points  
   - linear: \(K=XX^T\)  
   - poly degree \(d\): \((1/\sigma\,K)^d\) style as in slides  
   - RBF: \(\exp(-\|x-x'\|^2/(2\sigma^2))\)
2. **Train:** assemble QP matrices \(P,q,G,h,A,b\) → `cvxopt.solvers.qp(...)`  
   Soft margin: \(0\le\lambda_i\le C\).
3. Keep points with \(\lambda_i>0\) (within tolerance) as **support vectors**; drop the rest.
4. Compute \(b^*\) (8.10).
5. **Classify \(z\):** \(\mathrm{sign}\!\big(\sum_i\lambda_i t_i K(x_i,z)+b^*\big)\) (hard) or raw value (soft).

Overfitting control: still minimizing \(w^Tw\) keeps weights small; soft-margin \(C\) trades margin vs errors.

---

# 7. Multi-class SVM

Standard SVM is **binary**.

**One-vs-rest:** for \(N\) classes, train \(N\) SVMs (class \(k\) vs all others).  
Predict with the SVM that gives the **strongest** (largest) decision value.

---

# 8. Advantages & disadvantages (exam lists)

### Advantages
1. **Global minimum** (convex QP).
2. **Maximum margin** classifier → strong generalization story (VC / margin).
3. **Sparse:** only **support vectors** matter; most training data can be discarded after training.

### Disadvantages
1. Naturally **binary** (multi-class needs tricks).
2. **Kernel choice** is hard / critical.
3. **Speed and size** — training and sometimes testing; poor on **very large** \(n\).
4. Optimal multi-class design still research-ish.
5. Scalability issues (some progress exists).

---

# 9. Lab 5 — full worked solutions

## Q1. Optimal separating line, SVs, margin

**Data (book Problem 8.1 / Lab images):**

| Class +1 | Class −1 |
|----------|----------|
| \((1,1),\ (1,2),\ (2,1)\) | \((0,0),\ (1,0),\ (0,1)\) |

**Plot:** + points in the upper-right; − points near the origin / axes.

**Optimal hard-margin SVM** (verified):
\[
w^* = (2,\,2),\quad b^* = -3
\]
Decision: \(\mathrm{sign}(2x_1 + 2x_2 - 3)\), i.e. line
\[
\boxed{x_1 + x_2 = 1.5}
\]
(parallel margin faces: \(x_1+x_2=2\) through \((1,1)\), and \(x_1+x_2=1\) through \((1,0)\) and \((0,1)\)).

**Support vectors:** \(\boxed{(1,1),\ (1,0),\ (0,1)}\)  
(Not SVs: \((1,2),\ (2,1),\ (0,0)\) — farther from the boundary.)

**Margin:**
\[
\|w\| = \sqrt{8} = 2\sqrt{2},\qquad M = \frac{1}{\|w\|} = \frac{1}{2\sqrt{2}} = \frac{\sqrt{2}}{4} \approx 0.354
\]
Full gap between parallel faces \(= 2M = 1/\sqrt{2} \approx 0.707\).

Check constraints \(t_i(w^Tx_i+b)\ge 1\):  
\((1,1)\to 1\); \((1,0)\to 1\); \((0,1)\to 1\); others \(>1\). ✓

---

## Q2. Perceptron on same data vs SVM

Train Perceptron (bias input −1, η = 0.5, targets 1/0). One run converges in a few epochs to e.g.
\[
w_{\text{bias}}\approx 1,\quad w_1\approx 0.5,\quad w_2\approx 1
\]
Decision roughly: \(0.5\,x_1 + x_2 > 1\) (line \(x_2 = 1 - 0.5x_1\)), **not** the max-margin line.

**Differences to write:**

| | Perceptron | SVM |
|--|------------|-----|
| Objective | Any separating line that gets training right | **Maximum margin** line |
| Which line? | Depends on init, η, order | Unique (for hard margin, separable) optimum |
| Margin \(M\) | Often **smaller** (line can hug data) | **Largest** possible \(M=1/\|w\|\) |
| Uses all points? | Updates from mistakes; final weights use history | Final boundary uses **only SVs** |
| Guarantees | Convergence if linearly separable | Global QP optimum + better generalization story |

---

## Q3. Nonlinear data + \(\Phi_1\) transform

**Positive labels:** \(\{(2,2),\ (2,-2),\ (-2,-2),\ (-2,2)\}\)  
**Negative labels:** \(\{(1,1),\ (1,-1),\ (-1,-1),\ (-1,1)\}\)

**Transform (Lab equation):**
\[
\Phi_1\begin{pmatrix}x_1\\x_2\end{pmatrix}
=
\begin{cases}
\begin{pmatrix}4-x_2+|x_1-x_2|\\ 4-x_1+|x_1-x_2|\end{pmatrix}
& \text{if }\sqrt{x_1^2+x_2^2}>2\\[6pt]
\begin{pmatrix}x_1\\x_2\end{pmatrix}
& \text{otherwise}
\end{cases}
\]

**Radii:** positives \(r=\sqrt{8}\approx 2.83>2\) → transform; negatives \(r=\sqrt{2}\approx 1.41\le 2\) → stay.

**Transformed points:**

| Original | Class | \(\Phi_1\) |
|----------|-------|------------|
| (2,2) | + | (2, 2) |
| (2,−2) | + | (10, 6) |
| (−2,−2) | + | (6, 6) |
| (−2,2) | + | (6, 10) |
| (1,1),(1,−1),(−1,−1),(−1,1) | − | unchanged |

Negatives stay in a small diamond about the origin; three positives are pushed to large first-quadrant coords.  
**Linear separator example:** \(x_1 + x_2 = 3\)
- (2,2): 4 > 3 → +
- (10,6),(6,6),(6,10): ≫ 3 → +
- all negatives: sum ≤ 2 → −

So after \(\Phi_1\), a **straight line** separates the classes (kernel / feature-map idea by hand).

---

## Q4. Two circles + polynomial kernel → 3D

**Setup:** outer circle (e.g. radius 3) and inner circle (radius 1), ~10 points each — **not** linearly separable in 2D.

**Polynomial-style lift (slides):**
\[
\phi(x_1,x_2) = (x_1,\ x_2,\ x_1^2 + x_2^2)
\]
(or use kernel \(K(x,y)=(1+x^Ty)^s\) without building \(\phi\)).

**What happens:** third coordinate \(z = x_1^2+x_2^2\) = squared radius.
- Inner points: \(z \approx 1\)
- Outer points: \(z \approx 9\)

A plane such as \(\boxed{z = 5}\) (i.e. \(0\cdot x_1 + 0\cdot x_2 + 1\cdot z - 5 = 0\)) separates them in 3D.

**Lab steps:** sample 10 points on each circle → apply \(\phi\) / polynomial \(K\) → plot (2D before, 3D after) → optional: run SVM with poly kernel on the data.

---

# 10. Exam angle — Lesson 5

1. Why one separating line is better than another → **margin**.
2. Define **margin** and **support vector**.
3. Write primal \(\min \tfrac12 w^Tw\) s.t. \(t_i(w^Tx_i+b)\ge 1\).
4. \(M=1/\|w\|\); max margin ⇔ min \(\|w\|\).
5. Dual: \(w^*=\sum\lambda_i t_i x_i\); only SVs have \(\lambda_i>0\).
6. Classify with \(\sum\lambda_i t_i K(x_i,z)+b^*\).
7. Explain **kernel trick** in one paragraph + name **3 kernels**.
8. Soft margin / role of \(C\).
9. Pros/cons list; multi-class one-vs-rest.
10. Contrast **Perceptron vs SVM** on the same separable data (Lab Q2).
11. Connect to Lesson 4: **VC / capacity**; higher dim via kernels like the XOR trick.

---

# 11. Formula box (Lesson 5)

| Topic | Formula |
|-------|---------|
| Decision | \(y=w^Tx+b\) |
| Margin | \(M=1/\|w\|\) (half-width); full gap \(2/\|w\|\) |
| Primal | \(\min\tfrac12 w^Tw\) s.t. \(t_i(w^Tx_i+b)\ge 1\) |
| Soft | \(t_i(w^Tx_i+b)\ge 1-\eta_i\); \(\min w^Tw+C\sum\eta_i\); \(0\le\lambda_i\le C\) |
| Dual \(w\) | \(w^*=\sum_i\lambda_i t_i x_i\), \(\sum\lambda_i t_i=0\) |
| Bias | \(b^*=\frac1{N_s}\sum_{j\in SV}(t_j-\sum_i\lambda_i t_i x_i^Tx_j)\) |
| Predict | \(\sum_i\lambda_i t_i K(x_i,z)+b^*\) |
| Kernel | \(K(x,y)=\phi(x)^T\phi(y)\) |
| Poly | \((1+x^Ty)^s\) |
| Sigmoid | \(\tanh(\kappa x^Ty-\delta)\) |
| RBF | \(\exp(-(x-y)^2/(2\sigma^2))\) |
| Circle lift | \(\phi=(x_1,x_2,x_1^2+x_2^2)\) |
| Lab Q1 | line \(x_1+x_2=1.5\); SVs \((1,1),(1,0),(0,1)\); \(M=1/(2\sqrt{2})\) |

---

# 12. Traps

1. Targets for SVM theory are **±1**, not 0/1.
2. Margin: say whether you mean **half** (\(1/\|w\|\)) or **full** (\(2/\|w\|\)).
3. Perceptron “correct” ≠ SVM “optimal.”
4. Kernel trick: you do **not** need to compute \(\phi\) if you have \(K\).
5. Soft-margin \(C\): small \(C\) = larger margin / more slack; large \(C\) = fewer errors.
6. \(\lambda_i=0\) points are **not** support vectors — discard after training.
7. SVM = **global** min; MLP = often **local** min.
8. More features via \(\phi\) can help separability but kernels avoid explicit cost — still choose kernel carefully (overfitting / validation).

---

*Lesson 5 complete. Use `07_L4L5_QuestionBank_FormulaSheet.md` for drills.*
