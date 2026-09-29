# CS582 Machine Learning — Ultimate Study Guide
## Lesson 4: Dimensionality Reduction & VC Dimension (+ Lab 4)

> **Sources:** `4-Dimensionality_Reduction_VC_Dimension.ppt`, `Lab_4_Dim_Reduction_MLP.docx`, Marsland textbook §§2.1.2, Ch. 6 (LDA/PCA), lecture notes on VC dimension.
>
> Every number that can be checked (hypersphere volumes, LDA/PCA formulas, VC for 2D lines) was verified against the book and slides.
>
> **If you only read this file for Lesson 4, you should be able to answer the exam.**

---

## How to use

1. Read §§1–8 in order (concepts → math → algorithms → VC).
2. Memorize the **Exam angles** and the **formula box** at the end.
3. Do Lab 4 answers in §9 out loud.
4. Drill companion file `07_L4L5_QuestionBank_FormulaSheet.md`.

Instructor style: short “why / what / how,” plus hand calculations. For this lecture expect: **curse of dimensionality (volume table + intuition), 3 ways to reduce dimensions, LDA vs PCA, maximize \(S_B/S_W\), PCA steps (center → covariance → eigenvectors), VC dimension / shatter / perceptron VC = 3 in 2D, link to SVM.**

---

# 1. Why reduce dimensions?

When features (dimensions) grow:

| Problem | Why |
|--------|-----|
| **More neurons / weights** | More inputs → more input (and usually hidden) neurons → more weights |
| **More training data needed** | Curse of dimensionality (below) |
| **More computation** | Cost of many algorithms scales with dimension |
| **Hard to visualize** | Humans stop at 2–3D; 2D is best for viewing |

**Bottom line (slides):** reducing dimensions is key for **data need, speed, visualization, noise removal, and easier interpretation.**

Book adds: DR can **remove noise**, improve learning results, make data easier to work with; in extreme cases (e.g. SOM later) reduce to ≤3D for plotting.

---

# 2. Curse of dimensionality (MUST KNOW)

### Definition
As the number of dimensions increases, the **volume of the unit hypersphere does not keep increasing** — it eventually **shrinks toward zero**.

**Unit hypersphere:** all points at distance **1** from the origin.
- 2D → unit **circle**
- 3D → unit **sphere**
- higher D → **hypersphere**

### Volume table (slides / book — memorize peak)

| Dimension \(n\) | Volume |
|-----------------|--------|
| 1 | 2.0000 |
| 2 | 3.1416 |
| 3 | 4.1888 |
| 4 | 4.9348 |
| **5** | **5.2636** ← **maximum** |
| 6 | 5.1677 |
| 7 | 4.7248 |
| 8 | 4.0587 |
| 9 | 3.2985 |
| 10 | 2.5502 |

- Formula: \(\boxed{v_n = \dfrac{2\pi}{n}\, v_{n-2}}\)
- As soon as \(n > 2\pi \approx 6.28\), volume starts to shrink.
- Above about **20 dimensions**, volume is **effectively zero**.
- Slides: “Considering all factors, we are better off if dimensions do not exceed **3**” (for visualization / practical comfort — not a hard math law).

### Intuitive reason (Lab 4 Q1 — write this on the exam)

Put the unit ball inside a box of width 2 (coordinates from −1 to 1 on each axis).

1. In **2D**, the circle fills **most** of the square; only the **corners** are outside.
2. In **3D**, more of the cube’s volume is in the **corners** outside the sphere.
3. In high D, follow the diagonal to a corner of the hypercube: you hit the sphere’s surface when every coordinate is still small (e.g. ~0.1 in the book’s 100-D story). **Most of the line (and volume) toward the corner is outside the ball.**
4. Recurrence \(v_n = (2\pi/n)v_{n-2}\): when \(n > 2\pi\), each step **shrinks** volume.

So the ball’s volume **peaks around 5**, then **decreases** — data become sparse; “nearest neighbors” and density estimates break; you need **far more samples** to fill the space.

### ML consequence
Algorithms separate classes using features → **more features ⇒ more datapoints needed to generalize**. Blindly adding features hurts. Choose / derive useful features carefully.

---

# 3. Three broad ways to reduce dimensionality

| Method | Idea |
|--------|------|
| **1. Feature selection** | Keep useful features (correlated with the output); drop useless ones |
| **2. Feature derivation / extraction** | Create new features by **transforms** (move/rotate axes); combine features; keep the useful new axes |
| **3. Clustering** | Group similar points; see if fewer descriptors suffice |

### Correct representation is key (Fig. 6.1)
Same 4 points:
- As numbers: hard to see pattern
- As 2D plot: maybe a rotated rectangle
- **Right answer:** points on a **circle** → describe by **one angle** \(\pi/6,\,4\pi/6,\,7\pi/6,\,11\pi/6\) → **2D → 1D**

Choosing the right representation can make the problem trivial.

### Feature selection notes
- Later: Decision Trees do constructive feature selection / pruning.
- Feature selection is a **search** over subsets: for \(d\) features there are \(\mathbf{2^d - 1}\) nonempty subsets → expensive.
- Practice: **greedy** search (and sometimes backtracking).
- Course covers DR with **both supervised (LDA)** and **unsupervised (PCA)** learning.

---

# 4. LDA — Linear Discriminant Analysis (supervised)

### Idea
Use **class labels**, means, and covariances to find a projection that **separates classes well**:
- **Within-class scatter \(S_W\)** small (each class tight)
- **Between-class scatter \(S_B\)** large (class means far apart)
- Maximize \(\boxed{\dfrac{S_B}{S_W}}\) (equivalently maximize \(S_B/S_W\) after projection)

**Requires labeled data** (supervised).

### Scatter formulas (memorize)

**Within-class scatter:**
\[
S_W = \sum_{\text{classes }c}\sum_{j\in c} p_c\,(\mathbf{x}_j - \boldsymbol{\mu}_c)(\mathbf{x}_j - \boldsymbol{\mu}_c)^T
\tag{6.1}
\]
\(p_c\) = class probability (fraction of points in class \(c\)), \(\boldsymbol{\mu}_c\) = class mean.

**Between-class scatter:**
\[
S_B = \sum_{\text{classes }c}(\boldsymbol{\mu}_c - \boldsymbol{\mu})(\boldsymbol{\mu}_c - \boldsymbol{\mu})^T
\tag{6.2}
\]
\(\boldsymbol{\mu}\) = global mean.

### Projection onto a line \(\mathbf{w}\)
Any line = vector \(\mathbf{w}\). Projection of a point:
\[
z = \mathbf{w}^T \mathbf{x}
\]
(scalar = distance along \(\mathbf{w}\)).

After projecting every point, scatters become:
\[
\mathbf{w}^T S_W \mathbf{w}\qquad\text{and}\qquad \mathbf{w}^T S_B \mathbf{w}
\tag{6.3–6.4}
\]
Ratio: \(\dfrac{\mathbf{w}^T S_B \mathbf{w}}{\mathbf{w}^T S_W \mathbf{w}}\).

Differentiate w.r.t. \(\mathbf{w}\), set to 0 → solve **generalized eigenvector** problem for \(S_W^{-1}S_B\) (assuming \(S_W^{-1}\) exists).

**Two-class shortcut:** \(\mathbf{w}\) is in the direction of \(S_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)\).

Result: data from 2D → **1D** (or to `redDim` dimensions) **and** classes easier to separate.

**Picture check:** a bad projection line mixes classes; a good one separates them (Fig. 6.4).

### Python sketch (slides / book)
```python
# Sw = sum of class covariances weighted by class size
# Sb = C - Sw   (C = total covariance)   # book/slide trick
evals, evecs = la.eig(Sw, Sb)            # generalized eigenproblem
# sort eigenvalues descending; take top redDim eigenvectors as w
newData = np.dot(data, w)
```

### Exam one-liner
> LDA is **supervised** DR: project to maximize **between-class / within-class** scatter using labels.

---

# 5. PCA — Principal Components Analysis (unsupervised)

### Idea
Find new axes that **capture maximum variance**, then drop axes with little variance → dimensionality reduction.

**No labels needed** (unsupervised).

### Geometric story (Fig. 6.6)
1. Data ellipse tilted at 45° to the axes.
2. PCA **rotates + translates** so data runs along the new \(x'\)-axis and is **centered at the origin**.
3. Variation along \(y'\) is small → **ignore \(y'\)** → reduce dimensions.

### Algorithm (memorize the steps)

1. **Center** the data: subtract the mean.
2. Choose the direction of **largest variation** → first principal axis.
3. Among directions **orthogonal** to the first, take the one with most remaining variance → second axis.
4. Repeat until you run out of axes (or keep only top \(k\)).

**End result:** variation lies along the new axes; **covariance of transformed data is diagonal** → new variables are **uncorrelated**.

### Matrix form
Data matrix \(X\). Want rotation \(Y = P^T X\) so \(\mathrm{cov}(Y)\) is diagonal:
\[
\mathrm{cov}(Y) = \mathrm{diag}(\lambda_1,\lambda_2,\ldots,\lambda_N)
\]
\(\lambda_i\) = eigenvalues = variance along each PC.  
**\(P\)** = eigenvectors of the **covariance of \(X\)**.

### Python sketch
```python
data -= data.mean(axis=0)           # center
C = np.cov(np.transpose(data))
evals, evecs = np.linalg.eig(C)
# sort by eigenvalue size (descending); keep top nRedDim
x = np.dot(np.transpose(evecs), np.transpose(data))   # project
# reconstruct (optional): y = evecs @ x  then add mean back
```

### Final notes (slides — exam favorites)
1. **PCA is linear** — only rotate and translate. Cannot fix strongly nonlinear structure by itself.
2. Nonlinear cases → **kernel trick / Kernel PCA** (SVM lecture).
3. **Link to MLP autoencoder:** bottleneck hidden layer compresses input ≈ what PCA does (linear autoencoder ≈ PCA).

### Exam one-liner
> PCA is **unsupervised** DR: new orthogonal axes ordered by **variance**; drop small-variance axes.

### LDA vs PCA (comparison table — learn this)

| | **LDA** | **PCA** |
|--|---------|---------|
| Labels? | **Yes** (supervised) | **No** (unsupervised) |
| Goal | Maximize class **separability** \(S_B/S_W\) | Maximize **variance** / decorrelate |
| Uses | Means + within/between scatter | Covariance eigenvectors |
| Best when | Classification, labels available | Compression, visualization, unlabeled data |
| Slide remark | Optimal via generalized eigenvectors | “Usually works better” even though unsupervised |

---

# 6. VC Dimension — learning capacity

### Motivation
- How many things can a model classify? → **capacity**
- How well can it **generalize**?
- **VC dimension** (Vapnik–Chervonenkis, early 90s) measures **learning capacity**.

### Definitions (memorize)

**Shatter:** A model \(f\) **shatters** a set of points if **for every possible labeling** of those points, there exist parameters so \(f\) classifies them with **zero error**.

**VC dimension:** cardinality of the **largest** set of points that the hypothesis class can shatter.  
Formally: largest \(D\) such that **some** set of \(D\) points can be shattered.

Set-family version (slides): \(H\) shatters \(C\) if \(H\cap C\) contains **all subsets** of \(C\) (\(H\cap C \supseteq 2^C\)).

### Classic example: lines in 2D (perceptron)

- **3 points** (not collinear): every labeling can be separated by a straight line → **shattered**.
- **4 points** in a square with diagonal labels (XOR pattern): **no** single line works → cannot shatter 4.
- ⇒ For linear classifiers in 2D: \(\boxed{\mathrm{VC} = 3}\)

(Only 3 of the \(2^3=8\) labelings shown on the slide — enough to illustrate.)

### Neural nets (slides — results, not full derivation)

**Perceptron**
- Capacity \(C = 2^N\) if \(N < m\); otherwise less than \(2^N\).
- \(\mathrm{VCd} = m\) = dimension of the input vector (= number of weights \(W\) for a single neuron with bias counted appropriately).
- \(\mathrm{VCd}\) is \(\mathbf{O}(W)\).
- Slide warning: if you use **more training examples than VCd**, learning/generalization behavior changes (capacity vs sample size).

**MLP**
\[
\mathrm{VCd} \le 2\,(W_{\text{input layer}} + W_{\text{hidden layer}})\,\log V
\]
where \(V\) = number of neurons.

- Training-pattern need (order-of-magnitude, perceptron): about \(m\log m\).
- Full generalization theory postponed; **SVM is built on VC ideas** (next lecture: **maximum margin**).

### Exam one-liner
> VC dimension = size of the largest set the model can **shatter**. 2D linear separator: **VC = 3**. Larger VC → more capacity, higher overfitting risk unless you have enough data / regularization (SVM maximizes margin).

---

# 7. Exam angle — Lesson 4

Be ready to:

1. **Define** the curse of dimensionality and **explain** why hypersphere volume falls after dim ≈ 5 (corners of the box + \(v_n=(2\pi/n)v_{n-2}\)).
2. Recite the **volume table** peak at 5.
3. List **3 DR methods** (selection, derivation, clustering).
4. **LDA:** \(S_W\), \(S_B\), maximize ratio, supervised, projection \(z=w^Tx\).
5. **PCA:** center → cov → eigenvectors → keep top \(k\); unsupervised; diagonal cov; link to autoencoder.
6. **LDA vs PCA** table.
7. **VC:** shatter + VC definition; prove/sketch why line in 2D has VC = 3; perceptron VCd = \(m\).
8. Say **SVM uses VC / margin ideas**.

---

# 8. Formula box (Lesson 4)

| Topic | Formula / fact |
|-------|----------------|
| Hypersphere volume | \(v_n=(2\pi/n)v_{n-2}\); peaks at \(n=5\); ~0 for \(n\gtrsim 20\) |
| Feature subsets | \(2^d-1\) nonempty subsets |
| Within scatter | \(S_W=\sum_c\sum_{j\in c}p_c(x_j-\mu_c)(x_j-\mu_c)^T\) |
| Between scatter | \(S_B=\sum_c(\mu_c-\mu)(\mu_c-\mu)^T\) |
| Projected scatters | \(w^TS_Ww\), \(w^TS_Bw\); maximize ratio |
| Projection | \(z=w^Tx\) |
| PCA | \(Y=P^TX\); \(P\) = eigenvectors of \(\mathrm{cov}(X)\); \(\mathrm{cov}(Y)=\mathrm{diag}(\lambda)\) |
| VC (2D line) | 3 |
| Perceptron VCd | \(m\) (input dim) / \(O(W)\) |
| MLP VCd | \(\le 2(W_{\mathrm{in}}+W_{\mathrm{hid}})\log V\) |

---

# 9. Lab 4 — full answers

### Q1. Intuitive reason unit hypersphere volume decreases after dim > 5

**Answer structure:**

1. Unit hypersphere = ball of radius 1 about the origin; enclose it in a hypercube of side 2.
2. In low D the ball fills most of the box; as D grows, **volume concentrates in the corners** of the box — most of the box is **outside** the ball.
3. Along a diagonal toward a corner, you leave the ball long before you reach the corner → fraction of volume inside the ball **shrinks**.
4. Recurrence \(v_n=(2\pi/n)v_{n-2}\) forces shrinkage once \(n>2\pi\); table peaks at **5** then falls (by ~20, ≈0).

### Q2. Short summary of the first paper  
*(Gadekallu et al., “Analysis of Dimensionality Reduction Techniques on Big Data,” IEEE Access, 2020, doi:10.1109/ACCESS.2020.2980942)*

**Summary (use in your own words):**

Big datasets often have many attributes; some are **irrelevant** or redundant and burden ML algorithms (**curse of dimensionality**). The paper analyzes how **dimensionality reduction** helps Big Data ML. It focuses on prominent techniques such as **PCA** and related **feature extraction**, and studies their effect when combined with classifiers (e.g. **decision tree, random forest, Naive Bayes, SVM**), including application settings such as **intrusion detection**. Main takeaway: reducing dimensions by removing / transforming attributes can **cut computation and storage**, improve classifier performance, and mitigate the curse of dimensionality — but the **choice of DR method and how it interacts with the classifier** matters, so you should **compare** techniques on your data rather than assuming one method always wins.

*(Second paper — Cunningham & Ghahramani, JMLR survey on linear DR — is heavier math; lab only requires a short summary of the **first** paper.)*

### Q3. Key points for your project / future Big Data

Write something like:

- If my project later has **many features**, I will apply **PCA** (unsupervised compression / noise reduction) and/or **LDA** (if labeled classes) before training.
- I will check **feature selection** (correlation with target, greedy subset search) so I am not training on useless dimensions.
- DR helps **training speed**, **memory**, **visualization**, and often **generalization** when \(N\) is not huge compared to \(d\).
- I will validate DR choice with a **validation set** (as with MLP early stopping) — wrong reduction can throw away signal.

---

*Lesson 4 complete. Next: `06_Part5_SVM.md`.*
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
# CS582 — Lessons 4 & 5 Companion
## Question Bank · Combined Formula Sheet · Traps

> Use after reading `05_Part4_DimReduction_VC.md` and `06_Part5_SVM.md`.
> Answer out loud with books closed. Check yourself against the part files.

---

# PART A — Question bank with short answers

## Lesson 4 — Dimensionality reduction & VC

1. **Why reduce dimensions?**  
   Fewer weights/neurons, less data needed, less compute, better visualization, noise reduction, easier interpretation.

2. **What is the curse of dimensionality?**  
   As dimension grows, unit hypersphere volume eventually shrinks → space is sparse → need far more samples to generalize.

3. **Why does hypersphere volume fall after dim ≈ 5?**  
   Volume concentrates in hypercube corners outside the ball; \(v_n=(2\pi/n)v_{n-2}\) shrinks for \(n>2\pi\). Peaks at n=5 (~5.26), ~0 by n≈20.

4. **Three ways to do DR?**  
   Feature selection; feature derivation/extraction (transforms); clustering.

5. **LDA: supervised or unsupervised? Goal?**  
   Supervised. Maximize \(S_B/S_W\) (between / within class scatter) via projection \(z=w^Tx\).

6. **Write \(S_W\) and \(S_B\).**  
   \(S_W=\sum_c\sum_{j\in c}p_c(x_j-\mu_c)(x_j-\mu_c)^T\);  
   \(S_B=\sum_c(\mu_c-\mu)(\mu_c-\mu)^T\).

7. **PCA: supervised or unsupervised? Goal?**  
   Unsupervised. Orthogonal axes of max variance; drop low-variance axes; cov becomes diagonal.

8. **PCA algorithm steps?**  
   Center → covariance → eigenvectors/eigenvalues → sort → keep top k → project \(Y=P^TX\).

9. **LDA vs PCA in one line each.**  
   LDA uses labels to separate classes; PCA ignores labels and keeps variance.

10. **Link PCA ↔ MLP?**  
    Linear autoencoder bottleneck ≈ PCA compression.

11. **Define shatter and VC dimension.**  
    Shatter = all \(2^{|S|}\) labelings realizable with zero error. VC = size of largest shatterable set.

12. **VC of a line in 2D?**  
    3 (can shatter 3 points; not all 4-point XOR patterns).

13. **Perceptron VCd?**  
    \(m\) = input dimension; order \(O(W)\).

14. **Why mention VC before SVM?**  
    SVM built on VC / margin ideas: max margin → better capacity control / generalization.

15. **Feature selection complexity?**  
    \(2^d-1\) subsets → usually greedy search.

---

## Lesson 5 — SVM

16. **Why prefer the middle separating line?**  
    Largest margin → less sensitive to new points near the boundary.

17. **Define margin and support vector.**  
    Margin = max empty strip half-width around the boundary. SVs = points on the margin edges that define the boundary.

18. **Primal SVM problem?**  
    \(\min\tfrac12 w^Tw\) s.t. \(t_i(w^Tx_i+b)\ge 1\).

19. **Relation margin ↔ \(\|w\|\)?**  
    \(M=1/\|w\|\). Max margin ⇔ min \(\|w\|\).

20. **Why QP / convex?**  
    Quadratic convex objective + linear constraints → unique global minimum.

21. **Dual expression for \(w^*\)?**  
    \(w^*=\sum\lambda_i t_i x_i\); \(\sum\lambda_i t_i=0\); \(\lambda_i>0\) only on SVs.

22. **How classify a new \(z\)?**  
    \(\mathrm{sign}(\sum\lambda_i t_i K(x_i,z)+b^*)\).

23. **What is the kernel trick?**  
    Compute \(\phi(x)^T\phi(y)\) via \(K(x,y)\) without building \(\phi\); enables nonlinear SVM in original space cost.

24. **Name three kernels.**  
    Poly \((1+x^Ty)^s\); sigmoid \(\tanh(\kappa x^Ty-\delta)\); RBF \(\exp(-\|x-y\|^2/(2\sigma^2))\).

25. **Soft margin: role of \(C\)?**  
    Small \(C\): larger margin, more slack OK. Large \(C\): fewer errors, smaller margin.

26. **Multi-class how?**  
    One-vs-rest: \(N\) binary SVMs; pick strongest score.

27. **Three advantages of SVM.**  
    Global min; max margin; sparse (SVs only).

28. **Three disadvantages.**  
    Binary by nature; kernel choice hard; slow/large-scale limits.

29. **Perceptron vs SVM on same separable data?**  
    Both separate; SVM unique max-margin; Perceptron any separator / smaller margin / depends on training path.

30. **Circles → 3D with what map?**  
    \(\phi=(x_1,x_2,x_1^2+x_2^2)\); separate with a plane on the third coord (radius²).

31. **Lab Q1: line, SVs, \(M\)?**  
    \(x_1+x_2=1.5\); SVs \((1,1),(1,0),(0,1)\); \(M=1/(2\sqrt{2})\).

32. **Can SVM and Perceptron give different lines when both “correct”?**  
    Yes — that’s the whole point of max margin.

---

# PART B — Combined formula sheet (L4 + L5)

| Topic | Formula / fact |
|-------|----------------|
| Hypersphere | \(v_n=(2\pi/n)v_{n-2}\); peak \(n=5\); ~0 for \(n\gtrsim20\) |
| \(S_W\) | \(\sum_c\sum_{j\in c}p_c(x_j-\mu_c)(x_j-\mu_c)^T\) |
| \(S_B\) | \(\sum_c(\mu_c-\mu)(\mu_c-\mu)^T\) |
| LDA goal | max \(w^TS_Bw / w^TS_Ww\); \(z=w^Tx\) |
| PCA | center; eig(\(\mathrm{cov}\)); \(Y=P^TX\); keep top \(\lambda\) |
| VC (2D line) | 3 |
| Perceptron VCd | \(m\) / \(O(W)\) |
| SVM primal | \(\min\tfrac12 w^Tw\) s.t. \(t_i(w^Tx_i+b)\ge1\) |
| Margin | \(M=1/\|w\|\) |
| Dual \(w\) | \(w^*=\sum\lambda_i t_i x_i\) |
| Predict | \(\sum\lambda_i t_i K(x_i,z)+b^*\) |
| Soft | \(0\le\lambda_i\le C\) |
| Poly / RBF / tanh | \((1+x^Ty)^s\) / \(\exp(-\|x-y\|^2/2\sigma^2)\) / \(\tanh(\kappa x^Ty-\delta)\) |
| Circle lift | \((x_1,x_2,x_1^2+x_2^2)\) |
| Lab5 Q1 | line \(x+y=1.5\); SVs (1,1)(1,0)(0,1); \(M=1/(2\sqrt2)\) |

**Key numbers:** volume peak dim **5**; VC line **3**; Vapnik SVM **1992**; hypersphere ~dead by dim **20**.

---

# PART C — Traps (L4 + L5)

1. Curse ≠ “volume always decreases” — it **increases then decreases** (peak at 5).
2. LDA needs **labels**; PCA does **not**.
3. PCA does **not** optimize class separation (may mix classes).
4. VC: to claim VC ≥ D find **one** shatterable set of size D; to claim VC = D show **no** set of size D+1 shatters.
5. SVM targets **±1**.
6. Margin half vs full width — state convention.
7. Kernel trick ≠ “always go to infinite dimensions blindly” — still validate kernel/params.
8. Soft-margin \(C\) direction: small C → soft/large margin.
9. After SVM training, non-SV training points are **redundant**.
10. “SVM can’t do nonlinear” is **false** — kernels make nonlinear decision boundaries in input space.

---

# PART D — 60-second oral drills

**Drill 1:** Explain curse → why DR → LDA vs PCA in under 60 seconds.  
**Drill 2:** Draw margin, mark SVs, write primal, say \(M=1/\|w\|\).  
**Drill 3:** Kernel trick in 3 sentences + list 3 kernels.  
**Drill 4:** Lab Q1 numbers from memory.  
**Drill 5:** VC shatter definition + why 2D line has VC = 3.

If you can do all five without notes, Lessons 4–5 are exam-ready.
