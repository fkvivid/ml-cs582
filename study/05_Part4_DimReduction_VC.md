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
