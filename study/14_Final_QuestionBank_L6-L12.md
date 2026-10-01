# CS582 — Final Practice Question Bank + Formula Sheet (L6–L12)

> Answers are **self-contained** here. Built from Final Practice + Labs 6 & 8 + Lectures 6–12.

---

# A. Final Practice — short answers

### 1. Learning equation for unsupervised learning in an NN?
Winner-take-all competitive rule (only winner updates):
\[
\Delta w_{ij}=\eta(x_j-w_{ij})\quad\text{i.e.}\quad\mathbf{w}\leftarrow\mathbf{w}+\eta(\mathbf{x}-\mathbf{w})
\]
Normalize weights (and inputs) so magnitude does not fake a “win.”

### 2. Purpose of the sparsity term in Deep Learning? How does it work?
In autoencoders / sparse coding: add a penalty so **hidden units are rarely active** (most near 0). Forces a compact code — each unit specializes. Commonly KL divergence between target sparsity \(\rho\) and average activation \(\hat\rho_j\), added to reconstruction loss. Result: better features, less “always-on” redundancy.

### 3. Relate reward and actions in RL to BPN?
| RL | Backprop net |
|---|---|
| Action | Network **output** |
| Policy \(\pi(s)\) | Mapping **input→output** (weights) |
| Reward / return | Like a **target / error signal** |
| Improve policy | Like **weight update** to raise future reward |

TD / policy-gradient methods play the role of the learning rule; reward replaces labeled \(t\).

### 4. What is Policy and Value in RL?
- **Policy** \(\pi(s)\) or \(\pi(a|s)\): which action to take in state \(s\).
- **Value** \(V^\pi(s)\): expected **discounted return** from \(s\) following \(\pi\).  
  Action-value \(Q^\pi(s,a)\): expected return from \(s\) after taking \(a\), then \(\pi\).

### 5. Why Exploitation **and** Exploration? Will Exploration alone work?
- **Exploit:** use best-known actions (maximize immediate estimated reward).
- **Explore:** try lesser-known actions to improve estimates.
- Exploration alone: never consistently use what works → poor cumulative reward. Exploitation alone: stuck in suboptimal actions forever. Need both (e.g. \(\varepsilon\)-greedy).

### 6. Purpose of Pooling in DL / CNN?
Downsample feature maps (max/avg): **smaller** spatial size, some **translation tolerance**, less compute/params for later layers, keeps strongest responses (max-pool).

### 7. Why FC layer in CNN?
After conv/pool extract spatial features, **flatten → fully connected** layers mix them for final **classification / regression** (global decision). Softmax/linear head sits on FC (or global pool + linear).

### 8. Why not Sigmoid in deep nets? Good alternative?
Sigmoid saturates → **vanishing gradients** in deep stacks; not zero-centered. **ReLU** \(f(z)=\max(0,z)\): sparse, non-saturating for \(z>0\), fast. (Variants: Leaky ReLU, etc.)

### 9. Role of neighborhood function in SOM?
Selects which map neurons **besides the BMU** move toward the input. Large early → global topology; shrink later → local refinement. Without it → ordinary competitive learning (no topology).

---

# B. Agents & PEAS

### Definitions (memorize)
- **Agent:** perceives environment via **sensors**, acts via **actuators**.
- **Agent function:** \(f: \mathcal{P}^* \to \mathcal{A}\) (percept history → action).
- **Agent program:** implements/approximates that function.
- **Rational:** acts for best expected outcome given percepts/knowledge.
- **Autonomy:** handles unforeseen cases; compensates for partial/wrong prior knowledge.
- **Reflex:** action from **current** percept only.
- **Model-based:** uses world model.
- **Goal-based:** chooses actions to achieve goals.
- **Utility-based:** maximizes a **utility** performance measure.
- **Learning:** improves performance over time.

### PEAS tables

| | Robot soccer | Internet book-shopping |
|---|---|---|
| **P** | Score, mistakes, follow rules, … | Min cost/time; find interesting books |
| **E** | Ball, field, teammates, opponents | Internet, browsers, customer |
| **A** | Legs, arms, head, speakers | Display, place order, speech |
| **S** | Camera, mic, touch, teammate links | Keyboard, mouse, mic, pages, clicks |

| | Autonomous Mars rover | Theorem-proving assistant |
|---|---|---|
| **P** | Science return, distance, survival, energy | Proofs found, validity, time, lemmas reused |
| **E** | Mars surface, rocks, weather, Earth link | Axioms, conjecture, proof state, CPU |
| **A** | Wheels, arm, drill, cameras, radio | Add/remove steps, tactic calls, display |
| **S** | Cameras, IMU, spectrometers, telemetry | Keyboard, formula AST, checker feedback |

### Letter agent (\(30\times30\) binary)
- **States:** \(2^{900}\) images.
- **Functions** image→26 letters: \(26^{2^{900}}\) (astronomical).
- **Practical features:** strokes, holes, projections, HOG/CNN features — **not** raw \(2^{900}\) table lookup.

---

# C. GA Final Practice (corrected)

Solve \(a+2b+3c+4d=30\), genes in \([0,30]\), pop size 4, show 2 iterations.

**Fitness:** minimize residual \(r=a+2b+3c+4d-30\); use \(1/|r|\) (or \(1/(|r|+\varepsilon)\)).

**Init:** P1\((1,10,20,30)\), P2\((5,6,7,21)\), P3\((10,11,21,29)\), P4\((7,8,9,10)\).

| ID | \(r\) | \(1/|r|\) |
|---|---|---|
| P1 | **171** | 0.00585 |
| P2 | **92** | 0.01087 |
| P3 | **181** | 0.00552 |
| P4 | **60** | **0.01667** |

*(Course handout arithmetic mixed up coefficients — treat handout numbers as illustrative; use this table on the exam unless told to copy the handout.)*

**Selection (top 2 fitness):** **P4, P2**.

**Iteration 1 (corrected path):** Select parents **P4, P2**. One-point crossover after gene 2:
\[
(7,8,9,10)\times(5,6,7,21)\;\Rightarrow\;(7,8,7,21),\;(5,6,9,10)
\]
Example mutation: \((7,8,7,15)\), \((5,12,9,10)\). Recompute \(r\) and \(1/|r|\) for all four (two parents kept or replaced by policy). Keep the two highest-fitness chromosomes with the rest of the pool as the next population of 4.

**Iteration 2:** Recompute fitness on the new four; again pick top-2 probabilities as parents; crossover + mutate; stop (exam only asks for two iterations — solution need not hit \(r=0\)).

---

# D. Quick formula sheet (L6–L12)

**Unsupervised / SOM**
\[
\Delta\mathbf{w}=\eta(\mathbf{x}-\mathbf{w})\quad\text{(winner);}\quad
n_b=\arg\min_i\|\mathbf{x}-\mathbf{w}_i\|
\]

**k-means:** assign nearest; centre ← mean.

**Search:** BFS optimal if step cost=1; DFS not optimal; A*: \(f=g+h\), admissible \(h\) ⇒ optimal.

**GA:** encode → fitness → select → crossover → mutate → replace.

**RL**
\[
G_t=\sum_{k=0}^\infty\gamma^k R_{t+k+1},\quad
Q(s,a)\leftarrow Q(s,a)+\alpha\bigl[R+\gamma\max_{a'}Q(s',a')-Q(s,a)\bigr]
\]
ε-greedy: explore with prob \(\varepsilon\).

**Trees**
\[
H=-\sum_c p_c\log_2 p_c,\quad
IG=H(\text{parent})-\sum_v\frac{|S_v|}{|S|}H(S_v)
\]

**Bayes / NB**
\[
P(C|X)=\frac{P(X|C)P(C)}{P(X)},\quad
\hat C=\arg\max_C P(C)\prod_i P(x_i|C)
\]

**DL / CNN:** ReLU \(\max(0,z)\); pool downsamples; FC for final decision; AE sparsity penalizes average hidden activity.

---

# E. Night-before trap list

1. Unsupervised NN rule ≠ perceptron/backprop rule.  
2. Normalize competitive weights.  
3. SOM neighborhood schedule.  
4. Exploration **and** exploitation.  
5. Policy ≠ value.  
6. Sigmoid dies in deep nets → ReLU.  
7. Pooling ≠ convolution; FC ≠ filter.  
8. Sparsity ≠ dropout (related idea, different mechanism).  
9. A* needs **admissible** heuristic for optimality claim.  
10. GA fitness for minimization is **inverse** error, not raw \(r\).  
11. IG uses **weighted** child entropies.  
12. Naive Bayes: independence assumption; smooth zero counts.
