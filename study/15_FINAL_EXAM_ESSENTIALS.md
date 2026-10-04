# CS582 — REAL FINAL EXAM ESSENTIALS
## Super-important facts + formulas (predicted from Midterm + Final Practice)

> Built by mapping **what Midterm Practice and Final Practice actually ask**, plus Labs 2–8.  
> Night-before doc: memorize boxes + one-line answers. Details live in Parts 1–12 if you blank.

---

# How the real final is likely shaped

| Source | What it signals for the real exam |
|--------|-----------------------------------|
| **Midterm practice** | T/F on perceptron/XOR family; **hand-compute backprop**; why ML; SVM + LDA/PCA + VC concepts |
| **Final practice** | Short conceptual blasts (unsup / SOM / RL / DL / Agents) + **worked GA** + PEAS |
| **Labs** | Perceptron, MLP, DR, SVM margin, k-means/SOM, GA + PEAS + DFS |

**Predicted mix:** ~40% Final-style short answers · ~25% Mid-style NN/BP/SVM/DR · ~20% GA or PEAS work · ~15% unsup / trees / search.

**Skip unless asked:** Future tech / Semantic Engine / SARSA deep dive / KB agents (Final marks “not covered”).

---

# TIER 0 — Must recite cold (appeared on Mid or Final practice)

## One-line answers (Final favorites)

| # | Question | Answer |
|---|----------|--------|
| 1 | Unsupervised NN learning equation? | Winner only: \(\Delta\mathbf{w}=\eta(\mathbf{x}-\mathbf{w})\). Normalize weights. |
| 2 | SOM neighborhood function? | Updates **neighbors of BMU** too; large→small builds then refines **topology**. |
| 3 | Sparsity term (DL)? | Penalize avg hidden firing toward small \(\rho\) (often KL) → few units on → specialized features. |
| 4 | Policy vs Value (RL)? | Policy = which action; Value = expected **discounted** return from a state (or \(Q(s,a)\)). |
| 5 | Explore + Exploit? | Need both; explore alone never uses what works. \(\varepsilon\)-greedy. |
| 6 | RL ↔ BPN? | Action↔output; policy↔weights map; reward↔error/target signal driving updates. |
| 7 | Pooling? | Downsample maps; translation tolerance; less compute; keep strong responses (max). |
| 8 | FC in CNN? | Mix extracted features into class/regression scores. |
| 9 | Why not sigmoid in DL? | Vanishing gradients. Use **ReLU** \(\max(0,z)\). |
| 10 | What DL does / how? | Hierarchical feature learning via stacked layers; class **and** regression (change head/loss). |

## Midterm favorites

| # | Question | Answer |
|---|----------|--------|
| 11 | Perceptron converges if linearly separable? | **T** |
| 12 | Perceptron learns X-NOR / XOR? | **F** (not linearly separable) |
| 13 | BP “mainly” gradient descent? | **F** — uses GD; **main idea = back-propagate errors** |
| 14 | Scale all weights **and** thresholds by same \(+c\) (step net)? | Behavior **unchanged** |
| 15 | Why ML? | No/poor algorithms; big data; automate programming |
| 16 | LDA vs PCA? | LDA **supervised** (separate classes); PCA **unsupervised** (max variance) |
| 17 | VC? | Vapnik–Chervonenkis capacity; largest shatterable set size; **2D line = 3** |

## Agents (Final — memorize)

- **Agent:** sensors in, actuators out  
- **Agent function:** \(f:\mathcal{P}^*\to\mathcal{A}\)  
- **Rational:** best expected outcome · **Autonomous:** handles unforeseen / bad priors  
- **Reflex / model / goal / utility / learning** — know one-line each  
- **PEAS** for robot soccer + internet book-shopping (and Mars / theorem if Lab 8 asked)  
- Letter \(30\times30\): states \(2^{900}\); maps \(26^{2^{900}}\); use **features**, not raw table

---

# TIER 1 — Formula sheet (write these from memory)

## Perceptron & MLP (Mid will stress compute)

\[
y=\begin{cases}1& w\cdot x>0\\0&\text{else}\end{cases}
\quad(x_0=-1\text{ bias})
\]

\[
\boxed{w\leftarrow w-\eta(y-t)x}\quad\equiv\quad\Delta w=\eta(t-y)x
\]

\[
g(h)=\frac{1}{1+e^{-h}},\quad g'=g(1-g)
\]

\[
E=\tfrac12\sum_k(y_k-t_k)^2
\]

**Output delta (sigmoid + SSE):**
\[
\boxed{\delta_k=(y_k-t_k)\,y_k(1-y_k)}
\]

**Hidden delta:**
\[
\boxed{\delta_j=a_j(1-a_j)\sum_k w_{jk}\,\delta_k}
\]

**Updates:**
\[
w_{jk}\leftarrow w_{jk}-\eta\,\delta_k\,a_j,\qquad
v_{ij}\leftarrow v_{ij}-\eta\,\delta_j\,x_i
\]

**Exam drill:** forward → \(\delta\) out → \(\delta\) hidden (use **old** \(w\)) → update.

## SVM

\[
\boxed{\min\tfrac12\|w\|^2\ \text{s.t.}\ t_i(w^Tx_i+b)\ge 1},\quad
M=\frac{1}{\|w\|},\quad t_i\in\{\pm1\}
\]

\[
w^*=\sum_i\lambda_i t_i x_i,\quad
\hat y=\sum_i\lambda_i t_i K(x_i,z)+b
\]

Soft margin: \(0\le\lambda_i\le C\) (small \(C\) → softer / larger margin).

Kernels: poly \((1+x^Ty)^s\) · RBF \(e^{-\|x-y\|^2/2\sigma^2}\) · tanh.

Lab5 memory: line \(x+y=1.5\); SVs \((1,1),(1,0),(0,1)\); \(M=1/(2\sqrt{2})\).

## DR / VC

- Curse: high-\(d\) space sparse; hypersphere volume peaks ~\(d=5\), tiny by ~20  
- PCA: center → eig(cov) → keep top components  
- LDA: max \(S_B/S_W\) with labels  
- Shatter = all \(2^{|S|}\) labelings; VC = max shatterable size

## Unsupervised

\[
\boxed{\Delta\mathbf{w}=\eta(\mathbf{x}-\mathbf{w})\ \text{(winner only)}}
\]

k-means: assign nearest Euclidean · centre ← mean · local optimum · choose \(k\).

SOM BMU: \(n_b=\arg\min_i\|\mathbf{x}-\mathbf{w}_i\|\); also update neighbors via \(h(\cdot)\); anneal \(\eta\) + neighborhood size.

## Search / Agents

| Algo | Notes |
|------|--------|
| BFS | Optimal if uniform step cost |
| DFS | Not optimal; low memory |
| A* | \(f=g+h\); **admissible** \(h\) ⇒ optimal |

## GA (Final worked problem — practice until automatic)

Goal: \(a+2b+3c+4d=30\). Residual \(r=\ldots-30\). Fitness \(1/|r|\).

Loop: **init → fitness → select (top-2 probs) → crossover → mutate → replace → repeat**.

Correct init residuals: P1=**171**, P2=**92**, P3=**181**, P4=**60** → parents **P4 & P2**.

## RL

\[
G_t=\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}
\]

\[
Q(s,a)\leftarrow Q(s,a)+\alpha\big[R+\gamma\max_{a'}Q(s',a')-Q(s,a)\big]
\]

\(\varepsilon\)-greedy: random action w.p. \(\varepsilon\), else \(\arg\max Q\).

## Trees / Bayes

\[
H=-\sum_c p_c\log_2 p_c,\qquad
IG=H_{\text{parent}}-\sum_v\frac{|S_v|}{|S|}H(S_v)
\]

\[
P(C|X)=\frac{P(X|C)P(C)}{P(X)},\qquad
\hat C=\arg\max_C P(C)\prod_i P(x_i|C)
\]

(RF: bagging + random features + vote.)

## Deep Learning

- AE: encode→bottleneck→decode; + **sparsity**  
- CNN: Conv → ReLU → Pool → … → **FC** → softmax/linear  
- ReLU \(=\max(0,z)\) · Pool downsamples · Class **and** regression OK

---

# TIER 2 — Likely supporting (labs / lectures; shorter answers OK)

| Topic | Essentials |
|-------|------------|
| k-means Lab 6 | Furthest-init; iterate assign/mean until stable |
| Fraud / IDS with CL/SOM | Model **normal**; score novelty; imbalance → don’t expect fraud clusters |
| MP3/CD GA (Lab 8) | Binary genes which files on which CD; fitness = fill / waste; multi-CD encoding |
| DFS order | Preorder deep-left |
| Decision tree | Greedy max IG; prune overfitting |
| Softmax | Multi-class output; with CE, \(\delta=y-t\) |

---

# 15-minute oral drill (night before)

Say out loud without notes:

1. Unsupervised \(\Delta w\) + why normalize  
2. SOM neighborhood purpose  
3. Policy / Value / explore+exploit / RL↔BPN  
4. Sparsity · Pool · FC · ReLU vs sigmoid · DL class+reg  
5. Agent + PEAS soccer in 30 seconds  
6. Perceptron T/F trio + XOR not separable  
7. Backprop: write \(\delta_k\), \(\delta_j\), one update  
8. SVM primal + \(M=1/\|w\|\) + what is an SV  
9. LDA vs PCA + VC = 3 for a line  
10. GA: fitness \(1/|r|\), one full iteration sketch  
11. A*: \(f=g+h\) admissible  
12. Entropy + IG one sentence each  

If all 12 are clean → you are final-ready on essentials.

---

# Deadly traps (lose points here)

1. Unsupervised rule ≠ perceptron \(\eta(t-y)x\)  
2. BP main idea = **error backprop**, not “just GD”  
3. X-NOR / XOR **not** linearly separable  
4. Softmax/sigmoid **saturate** in deep nets → ReLU  
5. Explore **alone** fails  
6. Policy ≠ value  
7. Pooling ≠ convolution; sparsity ≠ dropout  
8. GA: wrong \(a+2b+3c+4d\) arithmetic (don’t do handout’s \(6\times10\) mistake)  
9. Hidden \(\delta\) uses **old** output weights  
10. Never tune on **test** set  
11. LDA needs labels; PCA does not  
12. Small \(C\) in SVM = softer margin  

---

# What to open if you blank one topic

| Blank on… | Open |
|-----------|------|
| Mid topics | `MidTerm_Practice_WITH_ANSWERS.md` · Parts 1–5 |
| Final shorts | `Final_Practice_WITH_ANSWERS.md` · `14_Final_…` |
| Formulas L1–3 | `04_QuestionBank_…` Part 5 |
| Formulas L4–5 | `07_L4L5_…` Part B |
| Unsup / GA / RL / DL | Parts 8 / 10 / 11 / 13 |

**Default night-before path:** this file only → then Final WITH_ANSWERS → then formula parts of 04 / 07 / 14.
