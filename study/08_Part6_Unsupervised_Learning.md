# CS582 Machine Learning — Ultimate Study Guide
## Lesson 6: Unsupervised Learning (+ Lab 6)

> Everything you need for Lesson 6 is in **this document**. Lab 6 k-means numbers were recomputed in code. Competitive-learning and SOM update rules match the course slides / Marsland Ch. 14.

---

# 1. Supervised vs unsupervised

| | Supervised | Unsupervised |
|---|---|---|
| Labels | Given (\(t_i\)) | **None** |
| Goal | Predict class / value | Find **structure**: clusters, manifolds, codes |
| Feedback | Error vs target | Similarity / reconstruction / competition |
| Course examples | Perceptron, MLP, SVM, trees | **k-means**, **competitive learning**, **SOM** |

Unsupervised question: *which inputs are alike?* Group them so similar points share a cluster (or a nearby neuron on a map).

---

# 2. k-Means clustering

### Idea
Pick \(k\) cluster centres (centroids). Assign every point to its **nearest** centre. Move each centre to the **mean** of its assigned points. Repeat until assignments stop changing.

### Algorithm (MEMORIZE)

1. **Initialize** \(k\) centres \(\mathbf{c}_1,\ldots,\mathbf{c}_k\) (random points, or distant points, or \(k\)-means++).
2. **Assign:** for each data point \(\mathbf{x}\),
   \[
   \text{cluster}(\mathbf{x}) = \arg\min_j \|\mathbf{x}-\mathbf{c}_j\|_2
   \]
   (Euclidean; slides also use squared Euclidean — same argmin).
3. **Update:** for each cluster \(j\),
   \[
   \mathbf{c}_j \leftarrow \frac{1}{|C_j|}\sum_{\mathbf{x}\in C_j}\mathbf{x}
   \]
   (mean of members; skip empty clusters carefully).
4. Repeat 2–3 until centres / assignments converge (or max iterations).

### Objective (SSE)
\[
J = \sum_{j=1}^{k}\sum_{\mathbf{x}\in C_j}\|\mathbf{x}-\mathbf{c}_j\|^2
\]
Each step does not increase \(J\). Stops at a **local** optimum (depends on init). Global optimum is hard (combinatorial).

### Complexity
\(O(t\,k\,n\,d)\) — \(t\) iterations, \(k\) centres, \(n\) points, \(d\) dims. With small \(t,k\) → effectively **linear** in \(n\).

### Strengths
- Simple, fast, intuitive.
- Works well when clusters are compact, roughly spherical, similar size.

### Weaknesses (exam favorites)
- Must choose **\(k\)**.
- Needs a **mean** (categorical → **k-modes**).
- **Local minima** → run many random inits; pick best \(J\).
- **Outliers** pull centres → remove distant points after several iters, use **median** (k-medians), or cluster a random sample then assign the rest.
- Too large \(k\) → **overfitting** (tiny meaningless clusters).

### Choosing \(k\)
Run many \(k\); plot SSE vs \(k\) (elbow). Too high \(k\) overfits.

---

# 3. Competitive learning (neural clustering)

### Setup
Input \(\mathbf{x}\). Several output neurons, each with weight vector \(\mathbf{w}_i\) (same dim as \(\mathbf{x}\)).

Activation (dot product):
\[
h_i = \mathbf{w}_i\cdot\mathbf{x} = \sum_j w_{ij}\,x_j
\]
**Winner-take-all:** neuron with **highest** \(h_i\) (or **smallest** Euclidean distance \(\|\mathbf{x}-\mathbf{w}_i\|\)) fires; others ignored.

Interpretation: weight vector = position of that neuron in input space = a **cluster prototype**.

### Why normalize (CRITICAL)
Example from slides — input \((0.2,0.2,-0.1)\):

| Weights | Dot product |
|---|---|
| \((0.2,0.2,-0.1)\) perfect match | \(0.09\) |
| \((0.15,-0.15,0.1)\) | \(-0.01\) |
| \((10,10,10)\) huge | \(3\) ← **wins wrongly** |

Large \(\|\mathbf{w}\|\) dominates. **Normalize weights** onto the unit hypersphere (\(\|\mathbf{w}_i\|=1\)); normalize inputs too. Then comparing activations is fair. Also stops weights growing unboundedly.

### Learning equation (Final exam favorite)

**Only the winner updates:**
\[
\boxed{\Delta w_{ij} = \eta\,(x_j - w_{ij})}
\quad\Leftrightarrow\quad
\mathbf{w}_{\text{win}} \leftarrow \mathbf{w}_{\text{win}} + \eta\,(\mathbf{x}-\mathbf{w}_{\text{win}})
\]

Moves the winning prototype **toward** the current input. Same geometric idea as moving a k-means centre — online / neural version.

That is the **unsupervised NN learning equation** the Final asks for.

### Vector quantization (VQ)
Codebook = set of weight vectors (prototypes). Encode input by **index of nearest codeword**. Competitive learning / k-means learn the codebook (speech MFCC features → discrete codes).

---

# 4. Self-Organizing Map (SOM / Kohonen map)

### Extra idea beyond competitive learning
Winner updates **and** neurons **near the winner on the map grid** also update (weaker). Neighborhood → **topology preservation**: similar inputs activate nearby map cells.

Mexican-hat style lateral interaction: excite neighbors, inhibit far neurons (course diagram).

### Algorithm (MEMORIZE)

**Init:** choose map size & dimension \(d\) (often 2D grid). Random distinct weights, **or** place along first \(d\) PCA directions.

**Train** — for each input \(\mathbf{x}\):
1. Best-matching unit (BMU) \(n_b\):
   \[
   n_b = \arg\min_i \|\mathbf{x}-\mathbf{w}_i\|
   \]
2. Update BMU:
   \[
   \mathbf{w}_{n_b} \leftarrow \mathbf{w}_{n_b} + \eta(t)\,(\mathbf{x}-\mathbf{w}_{n_b})
   \]
3. Update neighbors:
   \[
   \mathbf{w}_i \leftarrow \mathbf{w}_i + \eta_n(t)\,h(n_b,i,t)\,(\mathbf{x}-\mathbf{w}_i)
   \]
   \(h=1\) for neighbors, \(0\) otherwise (or a smooth Gaussian neighborhood).

**Anneal** learning rates and neighborhood size, e.g.
\[
\eta(t+1)=\alpha\,\eta(t)^{k/k_{\max}},\quad 0\le\alpha\le 1
\]
Same schedule idea for \(\eta_n\) and neighborhood width. Early: **large** neighborhood (global ordering). Late: **small** neighborhood (local fine-tuning).

Stop when map stabilizes or max iters hit.

**Use:** for a test point, find BMU by min Euclidean distance.

### Neighborhood function — purpose (Lab 6 Q1 / Final)

**Purpose:** decide **which neurons besides the winner** get pulled toward \(\mathbf{x}\).

**How it changes learning:**
- Large \(h\) early → whole region moves together → **global topological order** forms (map unfolds).
- Shrink \(h\) over time → only local tweaks → fine discrimination without destroying order.
- Without neighborhood: pure competitive learning (VQ); **no** map topology.

### Boundaries
Hard edges OK for ordered scales (pitch). Else wrap edges (1D line→circle, 2D→torus) so every neuron has a full neighborhood.

### Topology caveat
Projecting 1D/3D structure onto a 2D grid cannot preserve all distances perfectly (line bends; cube gets tangled). Relative order is approximate.

---

# 5. Lab 6 — full answers

## Q1. Neighborhood function in SOM
See §4 above: includes neighbors of the BMU in the weight update; large→small schedule creates then refines topological ordering. Without it, SOM collapses to ordinary competitive learning.

## Q2. SOM for network intruder detection
Features: (i) login time of day, (ii) session length, (iii) program types (encode categorically / multi-hot), (iv) program count.

**Preprocess:** scale continuous features to similar ranges (z-score or [0,1]); encode categoricals; optionally PCA if very wide.

**Data:** many normal sessions per user / role (hundreds–thousands). Intruders are rare — treat as **novelty**: train SOM mostly on normal traffic; flag sessions whose BMU has high quantization error or hits a rarely used map region.

**Map size:** start modest, e.g. \(10\times10\)–\(20\times20\); grow if many distinct normal profiles. Too big → sparse, unstable; too small → merges unlike users.

**Would it work?** Partially: catches gross pattern breaks (odd hours + odd tools). Misses mimicry attacks and needs careful thresholds. Better as one sensor in a larger IDS, not alone.

## Q3. Competitive learning for credit-card fraud
- Encode each transaction: amount (scaled), shop (embedding / one-hot / frequency features), time-of-day, day-of-week, maybe Δ from last txn.
- Train competitive prototypes **per cardholder** (or per peer group) on historical spend → “normal behavior codebook.”
- New txn: distance to nearest prototype (or unusual winning unit). Large distance → alert.

**Imbalance:** almost all data is legitimate → prototypes model **normal** well; fraud examples too few to form their own stable clusters (and you often **shouldn’t** need them).

**What to do:** unsupervised / one-class mindset (distance to normal); oversample fraud only if doing supervised follow-up; cost-sensitive threshold; maybe separate rare “travel” modes carefully so they aren’t false alarms.

**How well?** Good at sudden pattern breaks; weak if thief spends like the owner. Combine with rules + supervised model when labels exist.

## Q4. k-Means on 7 subjects (WORKED — verified)

| Subject | A | B |
|---|---|---|
| 1 | 1.0 | 1.0 |
| 2 | 1.5 | 2.0 |
| 3 | 3.0 | 4.0 |
| 4 | 5.0 | 7.0 |
| 5 | 3.5 | 5.0 |
| 6 | 4.5 | 5.0 |
| 7 | 3.5 | 4.5 |

**Init (lab):** two individuals **furthest apart** → subjects **1** and **4**.

| Group | Individual(s) | Mean vector (centroid) |
|---|---|---|
| Group 1 | 1 | \((1.0,\ 1.0)\) |
| Group 2 | 4 | \((5.0,\ 7.0)\) |

### Iteration 0 — assign all 7 points

Distances to \(\mathbf{c}_1=(1,1)\) and \(\mathbf{c}_2=(5,7)\):

| Subj | Point | \(d_1\) | \(d_2\) | Assign |
|---|---|---|---|---|
| 1 | (1,1) | 0 | 7.211 | G1 |
| 2 | (1.5,2) | 1.118 | 6.103 | G1 |
| 3 | (3,4) | **3.606** | **3.606** | tie → G1 (either OK) |
| 4 | (5,7) | 7.211 | 0 | G2 |
| 5 | (3.5,5) | 4.717 | 2.500 | G2 |
| 6 | (4.5,5) | 5.315 | 2.062 | G2 |
| 7 | (3.5,4.5) | 4.301 | 2.915 | G2 |

**New centres** (with 3→G1):
\[
\mathbf{c}_1=\frac{(1,1)+(1.5,2)+(3,4)}{3}=(1.833,\,2.333),\quad
\mathbf{c}_2=\frac{(5,7)+(3.5,5)+(4.5,5)+(3.5,4.5)}{4}=(4.125,\,5.375)
\]

*(If you put the tie 3→G2, you jump straight to the final partition below.)*

### Iteration 1

Reassign with new centres → **G1 = {1,2}**, **G2 = {3,4,5,6,7}**.

\[
\mathbf{c}_1=(1.25,\ 1.5),\qquad \mathbf{c}_2=(3.9,\ 5.1)
\]

### Iteration 2

Same assignment → **converged**.

**Final clusters**

| Cluster | Members | Centroid |
|---|---|---|
| 1 | subjects 1, 2 | \((1.25,\ 1.5)\) |
| 2 | subjects 3, 4, 5, 6, 7 | \((3.9,\ 5.1)\) |

---

# 6. Memorize box

| Item | Formula / fact |
|---|---|
| k-means assign | nearest centre (Euclidean) |
| k-means update | centre = mean of members |
| Competitive / unsupervised NN rule | \(\Delta\mathbf{w}=\eta(\mathbf{x}-\mathbf{w})\) **winner only** |
| Must | normalize weights (unit sphere) + inputs |
| SOM BMU | \(\arg\min_i\|\mathbf{x}-\mathbf{w}_i\|\) |
| SOM extra | neighborhood \(h\) updates nearby map neurons |
| Anneal | shrink \(\eta\) and neighborhood over time |
| VQ | nearest codebook index encodes the vector |

---

# 7. Exam traps

1. **Unsupervised learning equation** = competitive \(\eta(x-w)\), **not** backprop \(\eta\,t\,x\) / \(\eta\,\delta\,x\).
2. Forgetting **normalization** → large weights always “win.”
3. SOM without explaining **neighborhood schedule** = incomplete answer.
4. k-means **\(k\)** and **init** change the answer; local optimum ≠ global.
5. Fraud / intrusion: rare class → model **normal**, score novelty; don’t expect balanced clusters.
6. Equidistant point in k-means: either cluster OK; final Lab 6 answer still \(\{1,2\}\) vs \(\{3..7\}\).
