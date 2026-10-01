# CS582 — Combined Ultimate Study Guide (Lessons 6–12)

> Single file binding of Parts 6–12 + Final question bank. Each section is self-contained.

---


<!-- ===== 08_Part6_Unsupervised_Learning.md ===== -->

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


---


<!-- ===== 09_Part7_Optimization_Search_Agents.md ===== -->

# CS582 Machine Learning — Ultimate Study Guide
## Lesson 7: Optimization, Search & Agents (+ Lab PEAS)

> Everything you need for Lesson 7 is in **this document**. Agents, PEAS tables, search properties, Romania walkthroughs, letter-recognition state-space counts, and Lab 8 DFS order are all here.

---

# 1. Optimization vs Search

### Continuous optimization (what prior lectures did)
- Algorithms so far (perceptron, MLP, SVM, …) optimize a **continuous** objective, e.g. minimize
\[
E = \tfrac12\sum_i (y_i - t_i)^2
\]
by choosing a weight vector \(W^*\).
- **Gradient descent** works because continuous functions have **derivatives**.

### Discrete optimization = Search
- Many problems are **discrete** (cities, puzzle tiles, letter labels, moves).
- Discrete systems **do not have derivatives** → GD / calculus-based methods do not apply.
- For discrete systems, **optimization is called search**: find a sequence of actions / a path in a combinatorial space.
- AI has studied search since the **early 1960s**. Discrete optimization is central in theoretical CS; hard instances are often **NP-complete** (e.g. **TSP**).
- **Problem-solving agents** (incl. intelligent agents) are the usual frame for search.

**Exam one-liner:** continuous → minimize error with gradients; discrete → **search** a state space for a goal path.

---

# 2. Agents — definitions (Final Practice wording)

| Term | Definition (memorize) |
|------|------------------------|
| **Agent** | Anything that can be viewed as **perceiving** its environment through **sensors** and **acting** upon that environment through **actuators**. |
| **Agent function** | Maps from **percept histories** to actions: \(f: P^* \to A\). |
| **Agent program** | Program implementing / approximating the agent function. |
| **Agent =** | **architecture + program**. |
| **Rationality** | A rational agent acts to achieve the **best outcome**, or under uncertainty the **best expected outcome**. |
| **Autonomy** | Ability to handle **unforeseen** circumstances — compensates for **partial or incorrect** prior knowledge. |

Human sensors: eyes, ears, …; actuators: hands, legs, mouth, ….  
Robot sensors: cameras, IR range finders, …; actuators: motors, ….

### Environment assumptions for simple problem-solving agents
| Property | Meaning in this lecture |
|----------|-------------------------|
| **Static** | Formulate & solve without tracking ongoing environment change |
| **Observable** | Initial state known |
| **Discrete** | Actions at each state can be enumerated |
| **Deterministic** | Each action leads to a unique next state |

### Agent types (Final Practice + lecture architectures)

| Type | Core idea |
|------|-----------|
| **(Simple) Reflex** | Action from **current percept only**; ignores rest of history. Condition–action rules. |
| **Model-based reflex** | Maintains internal **state** + model of **how the world evolves** and **what actions do**; still chooses via condition–action rules. |
| **Goal-based** | Tracks state + **goals**; chooses actions that lead toward goal states (search / planning). Problem-solving agents are a kind of goal-based agent. |
| **Utility-based** | Performance given by a **utility function** (“how happy” in a state); pick action maximizing expected utility (handles trade-offs, not just binary goal). |
| **Learning** | Improves performance over time. Architecture: **performance element** (acts) + **critic** (vs performance standard) + **learning element** (makes changes) + **problem generator** (suggests exploratory actions). |

Hierarchy of power: reflex ⊂ model-based ⊂ goal-based ⊂ utility-based; learning can wrap any of them.

### Applications (lecture)
- **Software:** RPA (robotic process automation), workflow, mail, e-commerce, e-learning, …
- **Hardware/software:** manufacturing, welding, material handling, packaging, medical, …

---

# 3. PEAS

**PEAS** = **P**erformance measure, **E**nvironment, **A**ctuators, **S**ensors.

Lecture warm-up: automated taxi — fill all four before designing the agent.

### Robot soccer player (Final Practice answer)

| PEAS | Answer |
|------|--------|
| **Performance** | Score, number of mistakes, following the rules, … |
| **Environment** | Ball, playground, team members, competitors |
| **Actuators** | Legs, arms, head, speakers |
| **Sensors** | Video camera, microphone, touch sensors, communication sensors |

### Internet book-shopping agent (Final Practice answer)

| PEAS | Answer |
|------|--------|
| **Performance** | Minimizing cost, time; information about interesting books |
| **Environment** | The Internet, browsers, customer |
| **Actuators** | Speakers, display, add a new order |
| **Sensors** | Keyboard, mouse, microphone, web pages, buttons / hyperlinks clicked by users |

### Autonomous Mars rover (Lab 8 — sensible complete PEAS)

| PEAS | Answer |
|------|--------|
| **Performance** | Science return (samples/photos of interest), survival (power, thermal), mission lifetime, avoid hazards / stuck, communication reliability |
| **Environment** | Martian surface (terrain, rocks, dust, temperature extremes), limited solar power, delayed / intermittent Earth link, weather / dust storms |
| **Actuators** | Wheels / locomotion, arm / drill / scoops, cameras (pan/tilt), antennas, heaters, sample storage |
| **Sensors** | Cameras, spectrometers, IMU / wheel encoders, temperature / power sensors, hazard / proximity sensors, radio receiver |

### Mathematician’s theorem-proving assistant (Lab 8 — sensible complete PEAS)

| PEAS | Answer |
|------|--------|
| **Performance** | Correct proofs; fewer steps / shorter proofs; coverage of theorems attempted; time; usefulness to mathematician (hints accepted) |
| **Environment** | Formal logic language, axiom/lemma library, user’s conjectures and partial proofs, computational resources |
| **Actuators** | Display / write proof steps, apply inference rules, query lemma database, suggest lemmas or counterexamples |
| **Sensors** | Keyboard / mouse / UI; parse statements and proof scripts; read axiom/lemma store; receive user accept/reject feedback |

**Exam tip:** PEAS is always **four columns**. Do not mix sensors into actuators. Performance = **how success is scored**, not “what the agent does.”

---

# 4. Well-defined search problems

A problem is specified by:

1. **Initial state** — current configuration.  
2. **Successor function** / actions — for a state, legal `(action, next-state)` pairs.  
3. **State space** — all states reachable from the initial state via successors. Representable as a **graph**: vertices = states, edges = actions.  
4. **Goal test** — is this state a goal? Goal states ⊆ state space.  
5. **Path cost** — numeric cost of a path (often sum of step costs). Reflects the agent’s performance measure.  

- A **path** = sequence of states linked by actions.  
- A **solution** = path from initial state to a goal.  
- **Optimal solution** = solution with **lowest path cost**.

### Holiday example (lecture)
- Goal: be in Miami after 4 days.  
- States: cities. Actions: drive between cities.  
- Solution: e.g. San Francisco → LA → Phoenix → Dallas → Miami.

### 8-puzzle (lecture)
| Component | Instantiation |
|-----------|----------------|
| States | Locations of the 8 tiles + blank |
| Actions | Move blank L / R / U / D |
| Goal test | Match given goal configuration |
| Path cost | 1 per move |

Optimal \(n\)-puzzle is **NP-hard**.

**State vs node**
- **State** = physical / logical configuration.  
- **Node** = search-tree record: state, parent, action, path cost \(g\), depth.  
- `Expand` creates child nodes via the successor function.

---

# 5. State-space size & branching factor

\[
b = \text{branching factor (avg.\ \# children per node)}
\]

| Domain | Facts from lecture |
|--------|--------------------|
| **8-puzzle** | \(b \approx 2.13\); depth-20 tree \(\sim 3.7\times 10^6\) nodes; only \(9!/2 = 181{,}440\) distinct states |
| **Rubik’s cube** | \(b \approx 13.34\); \(\sim 9.01\times 10^{17}\) states; avg solution depth \(\sim 18\) |
| **Chess** | \(b \approx 35\) legal moves per turn on average |

### Letter recognition (Final Practice — class question)

Agent sees a **binary** \(30\times 30\) image → \(900\) pixels, each on/off. Action = which letter the image is.

1. **How many states?**  
   \[
   |S| = 2^{900}
   \]
   (each pixel independent on/off).

2. **How many agent functions** (maps from image-state to character)?  
   If there are \(26\) letter actions:
   \[
   |A|^{|S|} = 26^{2^{900}}
   \]
   (huge — cannot store a lookup table over all images).

3. **Practical feature representation?**  
   Do **not** treat raw \(900\) bits as the feature vector for every possible image. Extract **features** (stroke detectors, edge/region stats, moments, learned embeddings, …) so the agent works in a much smaller feature space and generalizes.

**Exam angle:** raw percept space is exponential; intelligent agents need **features** / learning, not enumeration of \(f:P^*\to A\).

---

# 6. Search tree framework

**Idea:** explore by **expanding** states — generate successors of already-explored states.

**Fringe (frontier):** nodes generated but **not yet expanded** (leaf candidates). Strategy = order of expansion.

### General tree-search (lecture pseudocode)
```
function TREE-SEARCH(problem, fringe) returns solution or failure
  fringe ← Insert(Make-Node(Initial-State), fringe)
  loop
    if fringe empty → failure
    node ← Remove-Front(fringe)
    if Goal-Test(State[node]) → return Solution(node)
    fringe ← InsertAll(Expand(node, problem), fringe)
```

`Expand` fills parent, action, state,  
\(g(\text{child}) = g(\text{parent}) + \text{step-cost}\), depth \(+1\).

### Evaluating strategies
| Criterion | Question |
|-----------|----------|
| **Completeness** | Always find a solution if one exists? |
| **Time** | \# nodes generated |
| **Space** | Max nodes in memory |
| **Optimality** | Always least-cost solution? |

Complexity in terms of:
- \(b\) = max branching factor  
- \(d\) = depth of **shallowest / least-cost** solution (context-dependent)  
- \(m\) = max depth of state space (may be \(\infty\))  
- \(l\) = depth limit (DLS)  
- \(C^*\) = optimal solution cost; \(\varepsilon\) = min step cost (UCS)

---

# 7. Uninformed (blind) search

Uses only the problem definition: generate successors + recognize goals. **No** estimate of “how close” a non-goal is.

Algorithms in the lecture list: **BFS, UCS, DFS, DLS, IDS, bidirectional**.

---

## 7.1 Breadth-First Search (BFS)

- Fringe = **FIFO** queue; new successors at the **end**.  
- Expand all depth-\(i\) nodes before depth \(i+1\).

| Property | Value |
|----------|--------|
| Complete? | **Yes** (if \(b\) finite) |
| Optimal? | **Yes** if **all step costs equal** |
| Time | \(O(b^{d+1})\) |
| Space | \(O(b^{d+1})\) — **space is the bigger problem** |

Illustration (\(b=10\), lecture table — order of magnitude): depth 10 already \(\sim 10^{10}\) nodes / days of time / terabytes of memory. Memory dies first.

---

## 7.2 Uniform-Cost Search (UCS)

- Fringe = **priority queue** ordered by path cost \(g(n)\).  
- Always expand the **cheapest path so far** (not necessarily shallowest).  
- Same idea as Dijkstra on trees/graphs for shortest path.  
- Lecture note: **A\* = UCS + heuristic** (when \(h\equiv 0\), A\* degenerates to UCS).

| Property | Value |
|----------|--------|
| Complete? | **Yes** (under standard positive-cost assumptions) |
| Optimal? | **Yes** |
| Time / Space | \(O\!\left(b^{\lceil C^*/\varepsilon\rceil}\right)\) |

---

## 7.3 Depth-First Search (DFS)

- Fringe = **LIFO** stack; put successors at the **front**.  
- Expand **deepest** unexpanded node (ties: left → right).

| Property | Value |
|----------|--------|
| Complete? | **No** in infinite-depth / looping spaces; **Yes** on finite trees with no loops (or if you avoid repeated states along a path) |
| Optimal? | **No** |
| Time | \(O(b^m)\) — terrible if \(m \gg d\); can be fast if solutions are dense |
| Space | \(O(bm)\) — **linear in depth** (big win vs BFS) |

### Lab 8 — DFS visit order (method + answer)

**Method (memorize):** preorder, **deep left-first** — visit a node, then recursively its children **left → right**.

Lab tree structure (nodes numbered 1–17 in BFS order on the worksheet figure):

```
            1
     /------|------\
    2       3       4
   / \     / \     / \
  5   6   7   8   9  10
  |       |
 11      12
 / \     / \
13 14   15 16
    |
   17
```

**DFS visit order:**
\[
1,\; 2,\; 5,\; 11,\; 13,\; 14,\; 17,\; 6,\; 3,\; 7,\; 12,\; 15,\; 16,\; 8,\; 4,\; 9,\; 10
\]

(Contrast: BFS would be \(1,2,3,4,5,\ldots,17\) — the printed numbers.)

---

## 7.4 Depth-Limited Search (DLS)

- DFS with a hard **depth limit** \(l\): do not expand beyond depth \(l\).

| Property | Value |
|----------|--------|
| Complete? | **No** (goal may be deeper than \(l\)) |
| Optimal? | **No** |
| Time | \(O(b^l)\) |
| Space | \(O(bl)\) |

---

## 7.5 Iterative Deepening Search (IDS)

- Run DLS with \(l = 0,1,2,\ldots\) until a goal is found.  
- Combines **BFS-like completeness / optimality** (for unit costs) with **DFS-like space**.

| Property | Value |
|----------|--------|
| Complete? | **Yes** |
| Optimal? | **Yes** (equal step costs) |
| Time | \(O(b^d)\) — regenerates shallow levels, but still \(O(b^d)\) |
| Space | \(O(bd)\) |

**Exam tip:** IDS is often the **default uninformed** choice when you need completeness + low memory.

---

## 7.6 Bidirectional search (named in lecture)

Search forward from start **and** backward from goal; stop when frontiers meet. Can cut effective depth roughly in half when applicable (goal must allow reverse actions). Know it exists; details less emphasized than BFS/DFS/IDS/A\*.

---

## 7.7 Summary table (lecture slide — MEMORIZE)

| | BFS | UCS | DFS | DLS | IDS |
|--|-----|-----|-----|-----|-----|
| **Complete?** | Yes | Yes | No | No | Yes |
| **Time** | \(O(b^{d+1})\) | \(O(b^{\lceil C^*/\varepsilon\rceil})\) | \(O(b^m)\) | \(O(b^l)\) | \(O(b^d)\) |
| **Space** | \(O(b^{d+1})\) | \(O(b^{\lceil C^*/\varepsilon\rceil})\) | \(O(bm)\) | \(O(bl)\) | \(O(bd)\) |
| **Optimal?** | Yes\* | Yes | No | No | Yes\* |

\*BFS / IDS optimal when step costs are **equal**. UCS optimal for general non-negative step costs (with standard assumptions).

---

# 8. Informed (heuristic) search

Uses extra knowledge: a heuristic \(h(n)\) that estimates cost from \(n\) to the **nearest goal**. Must have \(h(\text{goal})=0\).

Example: **straight-line distance (SLD)** to Bucharest on the Romania map.

Many AI search problems are **NP-complete** → worst-case still exponential; a good heuristic can solve **typical** instances fast or find a **good** (not always optimal) solution fast.

### Best-first search
Order fringe by an evaluation \(f(n)\) (“desirability”). Expand most desirable first.  
Special cases: **greedy** (\(f=h\)) and **A\*** (\(f=g+h\)).

---

## 8.1 Greedy best-first search

\[
\boxed{f(n) = h(n)}
\]
Expand the node that **looks** closest to the goal.

**Romania trace (Arad → Bucharest), \(h =\) SLD:**

1. Expand Arad → fringe: Sibiu \(253\), Timisoara \(329\), Zerind \(374\) → pick **Sibiu**.  
2. Expand Sibiu → among fringe, **Fagaras** \(176\) beats Rimnicu Vilcea \(193\), Timisoara \(329\), … → pick **Fagaras**.  
3. Expand Fagaras → **Bucharest** \(h=0\).

**Path found:** Arad → Sibiu → Fagaras → Bucharest.  
**Not optimal** (cheaper route exists via Rimnicu Vilcea → Pitesti).

| Property | Value |
|----------|--------|
| Complete? | **No** (can loop, e.g. Iasi ⇄ Neamt) |
| Optimal? | **No** |
| Time / Space | \(O(b^m)\) worst case; good \(h\) helps a lot |

---

## 8.2 A\* search

\[
\boxed{f(n) = g(n) + h(n)}
\]
- \(g(n)\) = **exact** cost from start to \(n\)  
- \(h(n)\) = estimated cost \(n\) → goal  
- \(f(n)\) = estimated cost of **cheapest solution through \(n\)**

Avoid expanding paths that are already expensive. Same tree-search skeleton; only the **queuing / priority** changes.  
Lecture: **A\* = UCS + BFS-style guidance** (UCS when \(h=0\); perfect \(h\) → march straight to goal).

**Romania A\* start (worked numbers from slides):**

| Node | \(g\) | \(h\) | \(f=g+h\) |
|------|-------|-------|-----------|
| Arad | 0 | 366 | \(366\) |
| Sibiu | 140 | 253 | \(393\) ← expand next |
| Timisoara | 118 | 329 | \(447\) |
| Zerind | 75 | 374 | \(449\) |

After expanding Sibiu, fringe includes (among others):

| Node | \(f = g + h\) |
|------|----------------|
| Rimnicu Vilcea | \(413 = 220 + 193\) ← expand (beats Fagaras \(415\)) |
| Fagaras | \(415 = 239 + 176\) |
| Timisoara | \(447\) |
| Zerind | \(449\) |

**Key contrast with greedy:** greedy picked Fagaras (\(h=176\)); A\* prefers Rimnicu Vilcea because **\(g+h\)** is smaller (\(413 < 415\)), steering toward the **optimal** Arad–Sibiu–RV–Pitesti–Bucharest route.

| Property (lecture) | Value |
|--------------------|--------|
| Complete? | **Yes** (finite \(b\); every operator adds cost ≥ some \(\varepsilon>0\)) |
| Optimal? | **Yes** (with admissible / consistent \(h\) — see below) |
| Time / Space | \(O(b^m)\) worst case; quality of \(h\) dominates practice |

**How fast?** For a fixed heuristic, **no other algorithm expands fewer nodes than A\*** (lecture claim). Useless \(h\equiv 0\) → UCS; perfect \(h\) → no real search.

---

## 8.3 Admissibility & consistency of heuristics

Needed to justify **A\* optimality** (lecture skips the full proof but uses the result).

### Admissible
\[
\boxed{h(n) \le h^*(n)}\quad\text{for all }n
\]
\(h^*\) = true cheapest cost from \(n\) to a goal.  
**Never overestimates.**  
Example: **SLD** never overestimates road distance → admissible for Romania.  
Also: \(h\equiv 0\) is admissible (but uninformative).

**Theorem (tree-search A\*):** if \(h\) is admissible → A\* is **optimal**.

### Consistent (monotonic)
\[
\boxed{h(n) \le c(n,a,n') + h(n')}
\]
for every action \(a\) taking \(n\to n'\) with step cost \(c\).  
Implies admissibility (by induction to the goal).  
With consistency, \(f\) is nondecreasing along paths; graph-search A\* is optimal without reopening subtleties.

**Exam phrases**
- Admissible = optimistic (never too large).  
- Overestimating heuristic → A\* may **miss** the optimal path.  
- Greedy ignores \(g\) → not optimal even with admissible \(h\).

### Classic 8-puzzle heuristics (standard companions to this lecture)
- \(h_1\) = \# misplaced tiles (excl. blank) — admissible.  
- \(h_2\) = sum of Manhattan distances of tiles to goals — admissible, usually tighter than \(h_1\).  
Tighter admissible \(h\) → fewer expansions (still optimal).

---

# 9. Complexity classes (lecture “food for thought”)

Not the core of the exam, but slides include:

| Class | Idea |
|-------|------|
| **P** | Decision problems solvable in **polynomial** time (deterministic) |
| **NP** | “Yes” answers **verifiable** in polynomial time (or solvable in poly time on a nondeterministic machine) |
| **NP-complete** | In NP, and **every** NP problem reduces to it in poly time (e.g. TSP decision, Hamiltonian cycle) |
| **NP-hard** | At least as hard as NP-complete; not necessarily in NP |

Open question: **P = NP?** (#1 Millennium problem).  
**AI takeaway:** many discrete search problems are NP-complete → need heuristics / GA / approximation for large instances.

**AI / ML discrete apps listed:** route finding, planning, TSP, robot navigation, games, speech, resource allocation.

---

# 10. Other algorithms named (not detailed here)

Lecture lists then **skips** details for: **Hill Climbing**, **Simulated Annealing**.  
Next course topic: **Evolutionary / Genetic Algorithms (GA)** — Lab 8 also has GA/knapsack/TSP; those belong with the GA lesson materials.

---

# 11. Exam angle — Lesson 7

1. Continuous optimization vs discrete **search**.  
2. Agent / agent function / rationality / autonomy definitions.  
3. Name & contrast the **5 agent types**.  
4. Fill a **PEAS** table (soccer, book-shopping, Mars rover, theorem prover).  
5. Components of a well-defined problem; 8-puzzle instantiation.  
6. Letter agent: \(2^{900}\) states; \(26^{2^{900}}\) functions; use **features**.  
7. BFS vs DFS vs IDS vs UCS properties table.  
8. Hand-trace DFS (Lab tree) and/or greedy vs A\* on Romania.  
9. Write \(f=h\) (greedy) and \(f=g+h\) (A\*).  
10. Define **admissible** / **consistent** heuristics; SLD example.  
11. Why BFS memory explodes; why IDS is attractive.  
12. TSP / 8-puzzle as hard search → motivates heuristics / GA.

---

# 12. Formula / memorize box (Lesson 7)

| Topic | Formula / fact |
|-------|----------------|
| Agent function | \(f: P^* \to A\) |
| Agent | architecture + program |
| PEAS | Performance, Environment, Actuators, Sensors |
| Letter states | \(2^{900}\) |
| Letter agent fns | \(26^{2^{900}}\) (26 letters) |
| BFS fringe | FIFO |
| DFS fringe | LIFO |
| UCS order | increasing \(g(n)\) |
| Greedy | \(f(n)=h(n)\) |
| A\* | \(f(n)=g(n)+h(n)\) |
| Admissible | \(h(n)\le h^*(n)\) |
| Consistent | \(h(n)\le c(n,a,n')+h(n')\) |
| IDS | DLS with \(l=0,1,2,\ldots\) |
| A\* degenerates | \(h\equiv 0\) → UCS |
| 8-puzzle states | \(9!/2=181{,}440\) |
| Chess \(b\) | \(\approx 35\) |
| Lab DFS order | \(1,2,5,11,13,14,17,6,3,7,12,15,16,8,4,9,10\) |

**Uninformed summary (compact):**

| | Complete | Optimal | Time | Space |
|--|----------|---------|------|-------|
| BFS | Y | Y\* | \(O(b^{d+1})\) | \(O(b^{d+1})\) |
| UCS | Y | Y | \(O(b^{\lceil C^*/\varepsilon\rceil})\) | same |
| DFS | N† | N | \(O(b^m)\) | \(O(bm)\) |
| DLS | N | N | \(O(b^l)\) | \(O(bl)\) |
| IDS | Y | Y\* | \(O(b^d)\) | \(O(bd)\) |

\*equal step costs; †yes on finite loop-free trees.

---

# 13. Traps

1. **Optimization ≠ always GD** — discrete problems need **search**.  
2. BFS optimal **only if step costs equal**; otherwise use **UCS / A\***.  
3. DFS is **not** complete in infinite / loopy spaces; **not** optimal.  
4. DFS space \(O(bm)\) is small; BFS space \(O(b^{d+1})\) is the killer — don’t say “BFS is always better.”  
5. Greedy \(f=h\) can find a path fast and still be **wrong** (Romania via Fagaras).  
6. A\* needs an **admissible** (and for graph-search, **consistent**) heuristic for the optimality claim; overestimate → broken optimality.  
7. \(h=0\) is admissible but turns A\* into **UCS**, not magic.  
8. PEAS: **Performance** is the scorecard, not an actuator. Sensors ≠ actuators.  
9. Reflex ≠ model-based: model-based keeps **internal state**. Goal-based ≠ utility-based: goals are yes/no; utility grades trade-offs.  
10. State space size \(2^{900}\) ≠ number of **agent functions** \(26^{2^{900}}\).  
11. IDS **re-expands** shallow nodes — that is intentional; space stays \(O(bd)\).  
12. Lab DFS: use **preorder left-first**, not the BFS numbering printed on nodes.  
13. Rationality ≠ omniscience; **autonomy** is about coping with incomplete priors.  
14. Do not invent hill-climbing / SA details for this lesson — slides skip them; go to GA next.


---


<!-- ===== 10_Part8_Genetic_Algorithms.md ===== -->

# CS582 Machine Learning — Ultimate Study Guide
## Lesson 8: Genetic Algorithms / Evolutionary Learning (+ Lab 8)

> Everything you need for Lesson 8 is in **this document**. Final-practice GA numbers were recomputed in code; the course sample has arithmetic/coefficient errata (called out below).

---

# 1. Evolutionary inspiration

- Conventional search often fails on huge discrete / combinatorial spaces (scheduling, design, TSP, knapsack).
- Same idea as NNs from neuroscience: **steal useful structure from nature** — here, **Darwinian evolution**.
- View evolution as **search** on a **fitness landscape**: individuals that live / mate more leave more offspring → population drifts toward fitter regions.
- The **genetic algorithm (GA)** is a computational model of that genetic process.
- GA is a **randomized heuristic search**: maintain a **population** of candidate solutions; improve it with **selection + crossover + mutation**.
- Popular when you have **no closed-form method** and the search space is intractable for exhaustive / simple local search.
- Parameters (pop size, crossover/mutation rates, selection) matter a lot and are often hard to set.

### Biological mapping (what we keep)

| Biology | GA |
|---------|-----|
| Chromosome / DNA string | Encoded candidate solution |
| Gene | One decision variable / allele block |
| Population | Set of \(N\) candidates |
| Fitness (survive & reproduce) | Fitness function \(F(\cdot)\) |
| Mating / recombination | Crossover |
| Copying error | Mutation |
| Generations | Outer GA loop |

Real genetics is richer; GA keeps only the operators that help search.

### When GA works / fails

- **Good:** continuous-ish fitness landscape (nearby encodings → similar fitness); scheduling; design; knapsack; map coloring; TSP with careful encoding.
- **Bad:** finding large primes (fitness is nearly discontinuous — 1 bit from a prime ≈ same as anything else).
- **Often overkill:** small 2D pathfinding (use A*).
- Needs a fitness signal that roughly says “how close” — otherwise selection has nothing to exploit.

---

# 2. Chromosome encoding

**Goal:** encode candidates so **mutation and crossover are easy** and **meaningful**.

### Typical encodings

1. **Binary string** (classic GA): bits for flags / discretized parameters.
2. **Integer / real vector**: genes \(=\) variables \(a,b,c,\ldots\) directly (common on exams).
3. **Permutation** (TSP): ordered list of cities — need **order-preserving** crossover (see §9).
4. **Symbolic alphabet**: e.g. map coloring with \(\{b,d,l\}\) for black/dark/light.

### Encoding checklist (exam)

1. What is one **gene**?
2. What is a full **chromosome** length / alphabet?
3. How do you **decode** → real solution?
4. Do crossover/mutation produce **valid** chromosomes? If not, repair or use specialized operators.

### Example — classification rule as bits

Attributes: Make \(\in\{B,C,N,G\}\), Tires \(\in\{K,T\}\), Handlebars \(\in\{S,C\}\), Water \(\in\{Y,N\}\) → **10 bits**, one per value (1 = accepted).

Rule “Bridgestone or Cannondale, treaded tires, straight bars (water anything)”:
\[
1100\;01\;10\;11
\]

### Example — 3-color map

Alphabet \(\{b,d,l\}\). Six regions → string \(\alpha=\{bdblbb\}\): region 1 black, 2 dark, …

### Example — equation solve \(a+2b+3c+4d=30\)

Chromosome \(=\{a,b,c,d\}\) with each gene \(\in[0,30]\) (numeric encoding).

---

# 3. Population, fitness, solution test

### Population
- Size \(N\) (exam often \(N=4\)).
- Initialize **random** (or seeded / “blank”).
- Diversity early → exploration; convergence later → exploitation.

### Fitness function \(F\)
- Heuristic: “how good is this candidate?”
- Higher fitness ⇒ more likely to be selected (for maximization-style selection).
- Should be **consistent** when possible (better solutions score higher), but GA is still probabilistic — **no optimality guarantee**.

**If the true goal is to minimize a cost \(f\):** invert it, e.g.
\[
\boxed{F = \frac{1}{|f|+\varepsilon}\quad\text{or course-style}\quad F=\frac{1}{f}\text{ when }f>0}
\]
Use \(\varepsilon>0\) (e.g. \(10^{-6}\)) to avoid division by zero when \(f=0\).

### Solution test
- Optional hard check: “is this feasible / exact?”
- Or: stop after \(G\) generations and return **best-so-far**.

### Parameters to name on an exam
Population size \(N\), generation limit \(G\), crossover rate \(p_c\), mutation rate \(p_m\), selection method, elitism yes/no.

---

# 4. Selection

Selection biases reproduction toward fitter individuals (**exploitation**).

### 4.1 Roulette (fitness-proportionate)
\[
p_i = \frac{F_i}{\sum_{j=1}^{N} F_j}
\]
Spin a weighted wheel; higher \(F_i\) ⇒ larger slice.  
**Trap:** if one individual dominates fitness, roulette → premature convergence. Raw fitness scale matters a lot.

### 4.2 Rank selection
Sort by fitness; assign selection probability from **rank** (not raw \(F\)). Softens domination; still prefers better individuals.

### 4.3 Tournament
Pick \(k\) individuals at random; winner \(=\) best in the subset (or probabilistic). Easy, scalable, selection pressure controlled by \(k\).

### 4.4 Elitism
Copy the **best \(e\)** individuals unchanged into the next generation so the best-so-far never dies by bad luck.  
Often combined with any of the above.

### Exam selection variants
- “Pick **2 parents with top probabilities**” (Final Practice style).
- After producing offspring: from \(\{\text{2 parents}+\text{2 offspring}\}\) keep **top 2 by fitness** (a mini tournament / \(\mu+\lambda\) style).

**N-queens slide idea:** fitness \(=\) # non-attacking pairs (max \(28=8\cdot7/2\)); selection shares e.g. \(24/(24+23+20+11)=31\%\), etc.

---

# 5. Crossover (recombination)

**Primary** search operator: combine building blocks from two parents (**exploit** good partial solutions).

### 5.1 One-point
Choose cut index \(k\); swap tails.
\[
\begin{align*}
P_1 &= 11\,|\,011 \\
P_2 &= 01\,|\,010 \\
&\Rightarrow\; C_1=11010,\; C_2=01011
\end{align*}
\]

### 5.2 Two-point
Choose two cuts; swap the middle segment.

### 5.3 Uniform
For each gene independently, child takes allele from \(P_1\) or \(P_2\) with probability \(1/2\) (or biased).

### Notes
- Crossover rate \(p_c\): probability a selected pair actually recombines (else copy).
- For **permutations**, naive 1-point yields duplicates/missing cities → use OX / PMX / cycle crossover (Lab TSP).

---

# 6. Mutation

**Secondary** operator: random local change (**explore**); preserves diversity; escapes local optima that pure crossover cannot reach.

- Binary: flip bit with small \(p_m\).
- Integer/real: replace gene with a new random value in range, or add noise.
- Permutation: swap two cities, inversion, etc.

Example: \(01010 \xrightarrow{\text{flip pos 1}} 11010\).

Typical \(p_m\) is **small** (per gene); too high ⇒ random search.

---

# 7. Schema idea (high level — memorize the slogan)

A **schema** is a template over the alphabet with “don’t care” \(\ast\), e.g. \(1\!*\!0\!*\!\).

- **Order** \(o(H)\): number of fixed positions.
- **Defining length** \(\delta(H)\): distance between outermost fixed positions.

**Schema theorem (Holland) — exam slogan:**
> Short, low-order, above-average schemas receive exponentially increasing trials in successive generations (under fitness-proportionate selection + crossover + mutation).

Intuition: selection favors good patterns; short patterns survive crossover more often; GA works by combining **building blocks**.  
You do **not** need the full formula for this course — know the intuition and the three schema descriptors.

---

# 8. GA algorithm loop (MEMORIZE)

\[
\boxed{\begin{aligned}
&\textbf{1. Init:} &&\text{random population of }N\text{ chromosomes}\\
&\textbf{2. Eval:} &&\text{compute }F\text{ for each}\\
&\textbf{3. Loop until stop:} &&\\
&\quad\textbf{(a) Select} &&\text{parents (roulette / rank / tournament / top-}p\text{)}\\
&\quad\textbf{(b) Crossover} &&\text{produce offspring}\\
&\quad\textbf{(c) Mutate} &&\text{offspring (small }p_m\text{)}\\
&\quad\textbf{(d) Eval} &&\text{new individuals}\\
&\quad\textbf{(e) Replace} &&\text{form next population (optional elitism)}\\
&\textbf{4. Return} &&\text{best chromosome found}
\end{aligned}}
\]

**Stop when:** exact solution found, max generations, fitness plateau, or time budget.

**Replacement options (slides):**
- Insert both offspring into new pop until size \(N\); or  
- Tournament among \(\{\text{2 parents}, \text{2 offspring}\}\) keep 2.

### Pros / cons

| Pros | Cons |
|------|------|
| Handles huge combinatorial spaces | Randomized — not optimal / complete |
| Little analytic model needed if encoding+fitness OK | Can stick in local maxima (mutation/crossover help) |
| Lower memory than full search tree | Encoding design can be hard |
| Parallel population search | Many sensitive parameters |

---

# 9. TSP encoding ideas (Lab 8)

**TSP:** visit each city once, return to start; minimize tour length. NP-complete.

### Encoding
- Chromosome \(=\) **permutation** of city IDs, e.g. \([3,1,4,2]\) means tour \(3\to1\to4\to2\to3\).
- Fitness \(F = 1/(\text{tour length})\) or \(F = -\,\text{length}\) (if selection handles negatives), or rank on length.

### Operators (must keep valid permutations)
- **Mutation:** swap two cities; reverse a segment (2-opt style).
- **Crossover:** Order Crossover (OX), PMX, Cycle Crossover — **not** naive 1-point on the list.
- **Selection:** roulette on \(1/\text{length}\), tournament, elitism of best tour.

### Why GA?
Exhaustive \((n-1)!/2\) tours explode; GA samples promising regions via recombination of good subtours (building blocks ≈ contiguous city sequences).

---

# 10. Applications from lecture

### Map coloring
- Encode colors as alphabet string.
- Fitness \(=\) # of boundaries with different colors (maximize), or minimize # conflicts.
- Example: 16/26 boundaries correct → fitness 16.
- Standard mutation (recolor one region) + crossover.

### Knapsack
- Binary chromosome: bit \(i=1\) iff item \(i\) is taken.
- Fitness: total value if weight \(\le W\), else 0 or penalty.
- Slides: GA quickly reaches near-optimum (\(\approx 499.94\) vs global \(499.98\)).

### Aircraft surface design
Huge continuous/discrete design space — classic “no good closed form” GA use case.

### Tiny binary walkthrough (slides)
Target alternating length-4 string. Start \(C_1=1000\), \(C_2=0011\) → crossover → mutate → get \(1010\) = solution.

### Find target bitstring \(11010010\)
Init 5 random 8-bit strings; fitness \(=\) negative Hamming distance to target (e.g. \(-3,-3,-5,-4,-2\)); select fitter parents; recombine; mutate; repeat.

---

# 11. Lab 8 — CD backup / multi-knapsack design (FULL ANSWER)

**Problem:** 5000 MP3s; backup to CDs; **minimize number of CDs** by packing each CD as full as possible. Design a GA (encoding, operators, multi-CD handling).

### 11.1 Problem type
Multi-bin packing / **multiple knapsacks** with identical capacity \(C\) (CD size). Related to the knapsack GA paper: binary packing + fitness for fill / value under capacity.

### 11.2 Encoding options (pick one and defend it)

**Option A — Per-CD binary packing (sequential / multi-stage)**  
- Chromosome length \(= n\) (files still unplaced): bit \(i=1\) ⇒ put file \(i\) on **this** CD.
- Run GA to fill CD 1 well; remove packed files; repeat for CD 2, … until none left.
- Simple; reuses single-knapsack GA. May be suboptimal globally (greedy across CDs).

**Option B — Group / assignment encoding (holistic)**  
- Chromosome \(=\) integer string of length \(n\): gene \(i \in \{1,\ldots,K_{\max}\}\) = which CD file \(i\) goes to (\(K_{\max}\) upper bound on #CDs, e.g. \(\lceil W_{\mathrm{tot}}/C\rceil +\) slack).
- One individual encodes the **entire** packing.

**Option C — Permutation + first-fit decoder**  
- Chromosome \(=\) order of files; decoder packs in that order into CDs (first-fit / best-fit).
- Fitness from resulting #CDs / waste. Crossover must be permutation-safe (OX).

### 11.3 Fitness (for one CD stage or whole packing)

**Single CD (Option A):** maximize fill without overflow:
\[
F = 
\begin{cases}
\sum_{i:\,x_i=1} s_i & \text{if }\sum s_i \le C\\
0 \text{ or } C - \mathrm{penalty}\cdot\mathrm{overflow} & \text{otherwise}
\end{cases}
\]
Equivalently maximize \(C - \text{wasted space}\) among feasible packings.

**Whole multi-CD (B/C):** minimize number of CDs used, then minimize total wasted space:
\[
F = \frac{1}{K + \lambda\cdot \frac{W_{\mathrm{waste}}}{C}}
\quad\text{or}\quad
F = -K - \lambda\cdot\mathrm{waste}
\]
with \(\lambda\) small so **fewer CDs** dominates.

Invalid (overfull CD) ⇒ heavy penalty or repair (drop lowest-priority / random bits until feasible).

### 11.4 Genetic operators
- **Selection:** tournament or roulette on \(F\); **elitism** keep best packing.
- **Crossover:** 1-point / 2-point / uniform on binary or integer genes; OX if permutation.
- **Mutation:** flip bits; reassign a file to another CD; swap two files’ CD IDs.
- **Repair:** after variation, if a CD exceeds \(C\), move/remove files until feasible.

### 11.5 Dealing with multiple CDs (must say this)
1. **Sequential:** optimize CD \(k\), freeze it, remove files, repeat (simple exam answer).  
2. **Simultaneous:** one chromosome assigns every file to a CD index; fitness on global \(K\) + waste.  
3. **Variable \(K\):** start with large \(K_{\max}\); fitness rewards unused empty CDs (or shrink \(K\)).

### 11.6 Exam-ready short paragraph
> Encode each packing as a binary string (include/exclude on current CD) or as CD-assignment integers. Fitness \(=\) filled bytes if \(\le C\), else penalize. Use fitness-proportionate or tournament selection, 1-point/uniform crossover, bit-flip or reassignment mutation, and repair overflows. For many CDs: either run the knapsack GA repeatedly (pack CD, remove files, repeat) or use a multi-bin chromosome whose fitness minimizes the number of CDs and residual waste.

*(Lab also asks PEAS / agents / DFS / TSP implement — exam focus for GA is the MP3 design + TSP encoding + operators.)*

---

# 12. Final Practice — solve \(a+2b+3c+4d=30\) (FULL WORKED SOLUTION)

**Prompt (course):** find \(a,b,c,d\in[0,30]\) with population size 4; show **2 iterations**. Fitness should favor solutions of \(a+2b+3c+4d=30\). Select **2 parents with top probabilities**; after mating/mutation, from the 4 relevant chromosomes keep **top 2 by fitness**; form next population; repeat.

### 12.0 Goal & fitness (MEMORIZE)

True residual (can be negative):
\[
f(a,b,c,d)=a+2b+3c+4d-30
\]
**Goal:** minimize \(|f|\) (ideally \(0\)).

**Recommended fitness:**
\[
\boxed{F=\frac{1}{|f|+\varepsilon}}
\]
**Course sample wording:** \(F=1/f\) (works when all \(f>0\); same ranking as \(1/|f|\) then).

Given init population:
\[
\begin{align*}
P_1&=\{1,10,20,30\},&
P_2&=\{5,6,7,21\},\\
P_3&=\{10,11,21,29\},&
P_4&=\{7,8,9,10\}.
\end{align*}
\]

---

## 12.1 Course sample solution (AS WRITTEN) + ERRATA

> Below is the sample’s walkthrough. **Do not memorize its arithmetic as truth** — see §12.2 corrections. Include it so you can recognize the official writeup.

**Sample claim:** minimize \(f=a+2b+3c+4d-30\); fitness \(1/f\).

**Sample evaluation:**

| ID | Sample expansion (their writing) | Sample \(f\) | Sample \(1/f\) |
|----|-----------------------------------|-------------|----------------|
| \(P_1\) | \(1+2\cdot10+3\cdot20+4\cdot30-30=1+20+60+120-30\) | **71** | \(0.014\) |
| \(P_2\) | \(5+6\cdot10+7\cdot20+21\cdot30-30=5+60+140+630-30\) | **805** | \(0.0012\) |
| \(P_3\) | \(10+11\cdot10+21\cdot20+29\cdot30-30=\ldots\) | **1380** | \(0.000072\) |
| \(P_4\) | \(7+8\cdot10+9\cdot20+10\cdot30-30=\ldots\) | **537** | \(0.0018\) |

**Sample probabilities:**  
\(\sum F \approx 0.014+0.0012+0.00072+0.0018=0.01772\)  
\(p(P_1)=0.014/0.01772\approx 0.79\) → top two **\(P_1,P_4\)**.

**Sample crossover** at **3rd gene** (swap from gene 3 onward):
\[
\begin{align*}
P_1&=\{1,10,20,30\},& P_4&=\{7,8,9,10\}\\
P_1'&=\{1,10,9,10\},& P_2'&=\{7,8,20,30\}
\end{align*}
\]

**Sample mutation:** gene 4 of \(P_1'\) → \(20\); gene 2 of \(P_2'\) → \(19\):
\[
\{1,10,9,20\},\qquad \{7,19,20,30\}
\]
Then evaluate these 4 (parents + mutated offspring), keep top 2 as \(Q_1,Q_2\).

**Sample 2nd iteration setup:** new population \(\{P_2,P_3,Q_1,Q_2\}\); repeat fitness → select top-2 parents → crossover → mutation → select.

### ERRATA (critical)

1. **\(P_1\) arithmetic:** expansion \(1+20+60+120-30\) equals **\(171\)**, not \(71\). Sample summed wrong.
2. **\(P_2,P_3,P_4\) coefficients:** sample computed as if
   \[
   a + b\cdot 10 + c\cdot 20 + d\cdot 30 - 30
   \]
   instead of
   \[
   a + 2b + 3c + 4d - 30.
   \]
   Explicitly, sample wrote for \(P_2\): \(5+6\times10+7\times20+21\times30-30\) — **WRONG**.
3. Correct \(P_2\): \(5+2\cdot6+3\cdot7+4\cdot21-30\).
4. Because of (1)–(2), sample’s ranking \(P_1\gg P_4\) is **not** what correct fitness gives (correct top parents are **\(P_4,P_2\)**).

---

## 12.2 CORRECT evaluation of the initial population

\[
\begin{align*}
P_1:&\; 1+2(10)+3(20)+4(30)-30=1+20+60+120-30=\mathbf{171},&
F_1&=\frac1{171}\approx 0.005848\\
P_2:&\; 5+2(6)+3(7)+4(21)-30=5+12+21+84-30=\mathbf{92},&
F_2&=\frac1{92}\approx 0.010870\\
P_3:&\; 10+2(11)+3(21)+4(29)-30=10+22+63+116-30=\mathbf{181},&
F_3&=\frac1{181}\approx 0.005525\\
P_4:&\; 7+2(8)+3(9)+4(10)-30=7+16+27+40-30=\mathbf{60},&
F_4&=\frac1{60}\approx 0.016667
\end{align*}
\]

Sum of fitnesses:
\[
\sum F = F_1+F_2+F_3+F_4 \approx 0.038909
\]
\[
\begin{align*}
p_1&\approx 0.1503,&
p_2&\approx 0.2794,&
p_3&\approx 0.1420,&
p_4&\approx 0.4283.
\end{align*}
\]

**Top 2 probabilities → parents \(P_4\) and \(P_2\)** (not \(P_1,P_4\)).

| | Sample (errata) | Correct |
|--|-----------------|---------|
| Best parents | \(P_1,P_4\) | \(P_4,P_2\) |
| Best \(F\) | \(P_1\) (wrong) | \(P_4\) (\(|f|=60\)) |

---

## 12.3 CORRECTED full 2-iteration walkthrough

Use \(F=1/|f|\) (all \(f>0\) here ⇒ same as \(1/f\)). Crossover cut after gene 2 (“3rd position” as in sample). Replacement: from \(\{\text{2 parents}+\text{2 mutated children}\}\) keep top 2 as \(Q_1,Q_2\); next population \(=\) the two **non-parents** \(+\) \(Q_1,Q_2\).

### Iteration 1

**Parents:** \(P_4=\{7,8,9,10\}\), \(P_2=\{5,6,7,21\}\).

**Crossover:**
\[
\begin{align*}
O_1&=\{7,8,7,21\},& f&=7+16+21+84-30=98,& F&=1/98\\
O_2&=\{5,6,9,10\},& f&=5+12+27+40-30=54,& F&=1/54
\end{align*}
\]

**Mutation (stated choices for a complete exam answer):**  
gene 4 of \(O_1\): \(21\to 8\); gene 2 of \(O_2\): \(6\to 3\):
\[
\begin{align*}
M_1&=\{7,8,7,8\},& f&=7+16+21+32-30=\mathbf{46},& F&=1/46\approx 0.02174\\
M_2&=\{5,3,9,10\},& f&=5+6+27+40-30=\mathbf{48},& F&=1/48\approx 0.02083
\end{align*}
\]

**Pool \(\{P_4,P_2,M_1,M_2\}\):**

| Chrom | \(f\) | \(F=1/|f|\) |
|-------|------|-------------|
| \(M_1=\{7,8,7,8\}\) | 46 | **0.02174** |
| \(M_2=\{5,3,9,10\}\) | 48 | **0.02083** |
| \(P_4=\{7,8,9,10\}\) | 60 | 0.01667 |
| \(P_2=\{5,6,7,21\}\) | 92 | 0.01087 |

**Select:** \(Q_1=M_1=\{7,8,7,8\}\), \(Q_2=M_2=\{5,3,9,10\}\).

**New population (non-parents + Q’s):**
\[
\{P_1,\; P_3,\; Q_1,\; Q_2\}
=
\{\{1,10,20,30\},\;\{10,11,21,29\},\;\{7,8,7,8\},\;\{5,3,9,10\}\}
\]

### Iteration 2

| ID | Chromosome | \(f\) | \(F\) | \(p_i\) |
|----|------------|------|------|--------|
| \(P_1\) | \(\{1,10,20,30\}\) | 171 | 0.005848 | 0.1084 |
| \(P_3\) | \(\{10,11,21,29\}\) | 181 | 0.005525 | 0.1024 |
| \(Q_1\) | \(\{7,8,7,8\}\) | 46 | 0.021739 | **0.4030** |
| \(Q_2\) | \(\{5,3,9,10\}\) | 48 | 0.020833 | **0.3862** |

**Parents:** \(Q_1,Q_2\).

**Crossover** (cut after gene 2):
\[
\begin{align*}
C_1&=\{7,8,9,10\},& f&=60\\
C_2&=\{5,3,7,8\},& f&=5+6+21+32-30=\mathbf{34}
\end{align*}
\]

**Mutation:** gene 4 of \(C_1\): \(10\to 5\); gene 2 of \(C_2\): \(3\to 1\):
\[
\begin{align*}
D_1&=\{7,8,9,5\},& f&=7+16+27+20-30=\mathbf{40},& F&=1/40=0.025\\
D_2&=\{5,1,7,8\},& f&=5+2+21+32-30=\mathbf{30},& F&=1/30\approx 0.0333
\end{align*}
\]

**Pool \(\{Q_1,Q_2,D_1,D_2\}\)** → top 2:
\[
D_2=\{5,1,7,8\}\;(|f|=30),\qquad D_1=\{7,8,9,5\}\;(|f|=40)
\]

Best-so-far after 2 iterations: \(\{5,1,7,8\}\) with \(a+2b+3c+4d=5+2+21+32=60\), residual \(30\) (improving from initial best residual \(60\)). Not necessarily solved in 2 iterations — as the prompt allows.

---

## 12.4 Alternate track — sample’s mating \(P_1\times P_4\) but with CORRECT \(F\)

If the exam forces the sample’s parent choice \(P_1,P_4\) and the sample’s mutated children, **recompute with correct coefficients**:

After sample mutation:
\[
\begin{align*}
P_1&: f=171,\; F\approx 0.00585\\
P_4&: f=60,\; F\approx 0.01667\\
\{1,10,9,20\}&: f=1+20+27+80-30=\mathbf{98},\; F\approx 0.01020\\
\{7,19,20,30\}&: f=7+38+60+120-30=\mathbf{195},\; F\approx 0.00513
\end{align*}
\]
Top 2: \(Q_1=P_4=\{7,8,9,10\}\), \(Q_2=\{1,10,9,20\}\).  
Next pop: \(\{P_2,P_3,Q_1,Q_2\}\).

**Iter 2 parents** (correct \(F\)): \(Q_1,Q_2\) (\(p\approx 0.385,\;0.236\)).

Crossover → \(\{7,8,9,20\}\) (\(f=100\)), \(\{1,10,9,10\}\) (\(f=58\)).  
Example mutation \(20\to15\), \(10\to5\):
\[
\{7,8,9,15\}\;(f=80),\qquad \{1,5,9,10\}\;(f=48)
\]
Top of pool: \(\{1,5,9,10\}\) and \(Q_1=\{7,8,9,10\}\).

**For the exam:** prefer §12.3 (parents from **correct** probabilities). Know §12.1 sample text + §12.2 errata so you can spot the coefficient bug.

---

# 13. Exam angle — Lesson 8

1. Evolution as search / fitness landscape; GA \(\leftrightarrow\) genes, crossover, mutation.
2. Write the **GA loop** (init → eval → select → crossover → mutate → replace).
3. Define encoding for a stated problem (bits / ints / permutation).
4. Fitness for minimize-\(|f|\): \(F=1/(|f|+\varepsilon)\).
5. Roulette \(p_i=F_i/\sum F\); contrast rank, tournament, elitism.
6. Draw 1-point / 2-point / uniform crossover; state role of mutation.
7. Schema slogan: short, low-order, above-average schemas proliferate.
8. Pros/cons; when GA is a bad idea (primes, tiny pathfinding).
9. **Lab:** MP3/CD multi-knapsack design (encoding + multi-CD strategy).
10. **TSP:** permutation encoding + specialized crossover.
11. **Final practice:** full 2-iteration numeric GA with **correct** \(a+2b+3c+4d\).

---

# 14. Formula / memorize box (Lesson 8)

| Topic | Remember |
|-------|----------|
| Residual (Final) | \(f=a+2b+3c+4d-30\) |
| Fitness (min \(\lvert f\rvert\)) | \(F=1/(\lvert f\rvert+\varepsilon)\) or course \(1/f\) if \(f>0\) |
| Roulette | \(p_i=F_i/\sum_j F_j\) |
| GA loop | Init → Eval → [Select → Crossover → Mutate → Eval → Replace]\(^*\) → Best |
| 1-point XO | Cut at \(k\); swap tails |
| Mutation | Rare random gene change / bit flip |
| Schema slogan | Short, low-order, above-average schemas get more trials |
| Elitism | Copy best unchanged each generation |
| Knapsack bit | \(x_i\in\{0,1\}\); \(F=\mathrm{value}\) if weight \(\le W\) else penalty |
| TSP | Permutation chromosome; OX/PMX; \(F=1/\mathrm{length}\) |
| Correct \(P_2\) | \(5+2\cdot6+3\cdot7+4\cdot21-30=92\) (**not** \(805\)) |

---

# 15. Traps

1. **Final-practice coefficients:** always \(a+2b+3c+4d\), never \(a+10b+20c+30d\). Sample \(P_2=805\) is wrong; correct \(P_2\) residual is **\(92\)**.
2. **\(P_1\) sum:** \(1+20+60+120-30=171\), not \(71\).
3. Fitness for a **minimization** goal must **invert** cost (\(1/|f|\)); do not select on raw \(f\) if larger \(f\) is worse.
4. Use \(|f|\) or ensure \(f>0\); \(F=1/f\) blows up / ranks wrong if \(f\) can be \(0\) or negative.
5. Roulette on raw fitness can be dominated by one individual → consider rank/tournament + elitism.
6. TSP/permutation: **illegal** to use naive 1-point without repair.
7. Mutation rate too high ⇒ destroy building blocks; too low ⇒ stuck.
8. Encoding quality matters more than fancy operators — invalid or deceptive encodings kill GA.
9. GA \(\neq\) guaranteed optimum; report **best-so-far**.
10. Multi-CD: say explicitly how multiple bins are handled (sequential pack vs assignment chromosome).

---

# 16. One-page drill

**Correct init residuals:** \(P_1=171,\;P_2=92,\;P_3=181,\;P_4=60\).  
**Correct first parents:** \(P_4,P_2\).  
**Operators:** select → crossover → mutate → keep fittest.  
**Lab MP3:** knapsack-style bits or CD-assignment genes; fitness = fill / minimize #CDs; repeat or multi-bin chromosome.  
**Schema:** short low-order above-average templates spread.  
**Loop:** population evolves by fitness-biased mating + variation until stop.


---


<!-- ===== 11_Part9_Reinforcement_Learning.md ===== -->

# CS582 Machine Learning — Ultimate Study Guide
## Lesson 9: Reinforcement Learning (+ Final Practice RL)

> Everything you need for Lesson 9 / Final RL questions is in **this document**. Built from Lecture 9 slides, sheet equations, and Final Practice.

---

# 1. Where RL sits — vs supervised & unsupervised

| Paradigm | Feedback | What you get |
|----------|----------|--------------|
| **Supervised** | Full targets / labels | Told the **correct answer** (and thus how to fix the error) |
| **Unsupervised** | No targets | Exploit **structure / regularities** in data |
| **Reinforcement** | **Reward only** — right/wrong-ish signal | Told **how well** you did, **not how to improve** |
| Evolutionary (context) | Fitness | Search + exploitation + exploration |

- RL is the **middle ground** between supervised and unsupervised.
- Closely related to **biological learning** (Thorndike’s **Law of Effect**: if something is good, do it again; if not, don’t).
- The learner must **try strategies** and see which work → that “trying out” **is search**.
- Search is fundamental: search over state/action space to **maximize reward**.

**Task (one line):** learn how to behave successfully to achieve a goal **while interacting** with an external environment — learn via **experience**.

**Classic examples**
- Game playing: know win/lose, **not** which move at each step was “correct.”
- Control (traffic): measure delay, **not** told how to reduce it.

---

# 2. Agent–environment loop

### Roles
- **Agent** = the thing that is learning (senses + acts).
- **Environment** = where it learns / what it learns about; also provides the **reward**.
- Interaction is a loop over discrete time \(t\).

### Loop (memorize the cycle)

1. Agent figures out what **state** it is in.  
2. **Policy** decides what **action** to take.  
3. Action is executed in the environment.  
4. **Reward function** returns a numerical reward (from state + action).  
5. Agent arrives in a **new state**.  
6. Iterate.

### Notation at one step
\[
s_t \;\xrightarrow{a_t}\; (s_{t+1},\, r_{t+1})
\]
- Agent sends action \(a_t\) to environment.  
- Environment returns next state \(s_{t+1}\) and reward \(r_{t+1}\).  
- Percept often splits into **state** + **reward**.

### Core objects (exam definitions)

| Object | Meaning |
|--------|---------|
| **State** \(s\) | Agent’s perception / description of the environment (what the world is like now) |
| **Action** \(a\) | What the agent can do in the current state; changes the environment |
| **Reward** \(r\) | Numerical signal of how well the agent is doing — **no info on how to improve** |
| **Episode** | One “game” / trial from start to a **terminal** state (episodic tasks) |
| **Policy** \(\pi\) | Mapping from states to actions (what to do) |
| **Value** \(V\) / \(Q\) | How good a state (or state–action) is in terms of **expected future reward** |

**Goal of RL:** learn to map **states → actions** to maximize reward **now and in the future**.

### Experience-based decision rule (slides)
1. “When I’ve been in this state before, what reward did I get?”  
2. “How was this reward linked to the actions?”  
3. Select action that **maximizes expected reward**.  
4. Occasionally **try something different** (explore).

---

# 3. Agent types (context for RL)

### Reflex → model-based → goal → utility → learning
- **Reflex:** action from **current percept** only (ignore history).
- **Model-based reflex:** keep internal **state**; model “how world evolves” + “what my actions do”; then condition–action rules.
- **Goal-based:** choose actions that lead toward **goals** (binary happy / not happy).
- **Utility-based:** utility \(U(\text{state})\) → real number (“degree of happiness”); compare speed vs safety vs cost when goals conflict.  
  Right decision ≈ \(f(\text{percept}, \text{goal})\) + quicker + safer + reliable + less cost.
- **Learning agent** (RL focus): improves over time.

### Passive vs active learning
- **Passive:** watch the world; learn utilities of states (no acting to explore).  
- **Active:** also **acts** — **this is RL**.

### Four components of a learning agent (MEMORIZE names)
1. **Performance element** — selects external actions (the “agent” you’ve seen so far).  
2. **Learning element** — makes improvements to the performance element.  
3. **Critic** — feedback on success vs a fixed **performance standard**.  
4. **Problem generator** — suggests exploratory actions / experiments (find better long-run behavior; handle unseen cases).

Taxi example: critic sees rude honks after a bad lane change → learning element installs “that was bad” → problem generator may suggest brake tests on wet roads.

---

# 4. Policy \(\pi\) and Value \(V\) / \(Q\) — FINAL FAVORITES

> Final asks: **“What is Policy and Value in RL?”**

### Policy \(\pi\)
\[
\boxed{a_t = \pi(s_t)}
\]
- A **policy** is a mapping from **states to actions** (how the agent behaves).
- Can be deterministic \(\pi(s)=a\) or stochastic \(\pi(a\mid s)\).
- Soft-max and \(\varepsilon\)-greedy are **naïve** policies for action selection during learning.
- Better: learn a **state-specific** useful policy.
- Want the **optimal policy** \(\pi^*\) — the one that produces the **greatest rewards** (not necessarily unique).
- **Learning policies is the crux of RL.**
- Pure exploitation with \(\pi^*\) is fine **after** the system has learned; during learning you still need exploration.

### Value — two flavors

Want to maximize **expected future reward**.

| Function | Looks at | Meaning |
|----------|----------|---------|
| **State-value** \(V(s)\) | State only (avg over actions under \(\pi\)) | How good is it to **be in** \(s\)? |
| **Action-value** \(Q(s,a)\) | State **and** action | How good is it to take \(a\) **in** \(s\)? |

Formally (under policy \(\pi\)), with return \(G_t\) (next section):
\[
\boxed{V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]}
\]
\[
\boxed{Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t = s,\, a_t = a]}
\]

- Predict \(V\) or \(Q\) for each policy; then pick a policy whose values are **maximal** over states.
- Action selection often uses \(Q\): average past reward for taking \(a\) in \(s\) converges toward the true expected reward for that action:
  \[
  Q_{s,t}(a) = \text{running average reward for action } a \text{ in state } s \text{ after } t \text{ visits}
  \]

**One-sentence exam answer**
> **Policy** = what action to take in each state (\(\pi(s)\)). **Value** = expected future (discounted) reward from a state \(V(s)\) or from taking an action in a state \(Q(s,a)\).

---

# 5. Return, discounting, episodic vs continual

### Reward function
- Input: current state + chosen action → **numerical** reward (can be \(+\) or \(-\)).
- Choice of action based on **expected** reward.
- Must match the **goal**; a **well-designed reward is crucial** to good learning (slides: “super critical”).
- Similar spirit to a **fitness function** in GA (scalar goodness signal).

### Maze / robot example (reward design)
- Only \(+50\) at center → may wander forever before finding it.  
- \(-\,1\) per move **and** \(+50\) at center → incentivizes **efficiency** (shorter paths).

### Episodic vs continual
| | Episodic | Continual (continuing) |
|--|----------|-------------------------|
| Structure | Breaks into “games”; has **terminal** state | Runs indefinitely |
| Reward scheme example | \(+1\) each timestep you stay “alive” / succeed | \(-\,1\) on failure + **discounting** |
| Note | Both can be made **equivalent** in representation | Need discount so infinite sums don’t blow up |

**Continual learning:** reward can be given for every little step (dense) or sparsely.

### Discounted return (MEMORIZE)
Un-discounted infinite future sum may **diverge**. Introduce discount factor \(\gamma \in [0,1)\):

\[
\boxed{G_t \;\;(\text{or } R_t) = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k\, r_{t+k+1}}
\]

**Why discount?**
1. Keep the return **finite** / well-defined for continuing tasks.  
2. Prefer **sooner** rewards over distant uncertain ones (“discount belief in future”).  
3. Matches: maximize reward **now and in the future**, with future weighted by \(\gamma^k\).

| \(\gamma\) | Effect |
|------------|--------|
| \(\gamma \to 0\) | Myopic — care mostly about **immediate** reward |
| \(\gamma \to 1\) | Far-sighted — future rewards matter almost as much as now |

**Uncertainty:** future is unknown; value estimates are guesses that get refined by experience.

---

# 6. Exploration vs exploitation — FINAL FAVORITE

> Final: **“Why do we need both Exploitation and Exploration? Will just Exploration work?”**

### Definitions
- **Exploitation:** pick the currently best-known action (use what you know).  
- **Exploration:** try other actions to gather information (may discover better strategies).

### Three action-selection methods (slides)
1. **Greedy:** always pick \(\arg\max_a Q(s,a)\) — pure exploitation.  
2. **\(\varepsilon\)-greedy:** with prob \(1-\varepsilon\) be greedy; with prob \(\varepsilon\) pick a **random** other action. Finds **better solutions over time** than pure greedy.  
3. **Soft-max:** refine which alternatives get picked when exploring (prefer promising suboptimal actions over awful ones).

### Why both are required
| Only exploitation | Only exploration |
|-------------------|------------------|
| Stuck with early, possibly **suboptimal** habits | Never **commits** to using good knowledge |
| Miss better actions never tried | Reward is not systematically maximized |
| No learning of unknowns beyond first luck | Wandering forever — no policy improvement toward max reward |

**Will just Exploration work?** **No.**
- Exploration alone = endless random search; you collect data but do **not** reliably **use** the best actions to maximize cumulative reward.
- You need exploration to **discover**, and exploitation to **capitalize**.

**Exam one-liner:** Exploration finds better actions; exploitation uses them. Either alone fails — random forever, or permanently stuck.

---

# 7. Relate reward & actions in RL to BPN — FINAL FAVORITE

> Final: **“Can you relate reward and actions in an RL with BPN? If so how?”**

Yes — same “learn from a scalar signal + produce outputs” story, different packaging:

| RL | Backprop NN (BPN / MLP) |
|----|-------------------------|
| **Actions** | **Network outputs** (what the system produces) |
| **Policy** \(\pi(s)\) | **Network mapping** inputs → outputs (weights define the mapping) |
| **Reward** (and TD / return error) | **Target / error signal** that drives **weight updates** |
| Maximize expected reward | Minimize loss / error vs targets |
| Critic / reward from environment | \((y - t)\) or loss gradient from labeled targets |

**Unpack**
- In BPN, the **error** \((y-t)\) (or loss) tells each weight how to change via backprop.  
- In RL there is **no full target action sequence** — only a **reward**. That reward (or TD error built from it) plays the role of the **training signal** that adjusts the policy / value estimates (and, if the policy is a NN, its weights).  
- Choosing actions in RL ↔ producing outputs in BPN.  
- Learning a policy ↔ learning the network function.

**Exam answer skeleton**
> Reward in RL ≈ the **error / target signal** in BPN that drives learning; **actions** ≈ **outputs**; the **policy** ≈ the **network** mapping states (inputs) to actions (outputs).

---

# 8. Markov property & MDP (high level)

### Why Markov?
Full history probability
\[
P(r_t=r',\, s_{t+1}=s' \mid s_t,a_t,r_{t-1},s_{t-1},\ldots)
\]
is impossible (too much data; tiny probabilities). **Do we need it all?**

### Markov property
**Current state is enough** (given action):
\[
\boxed{P(r_t = r',\, s_{t+1} = s' \mid s_t,\, a_t)}
\]
- Future depends only on **present** state + action, not full past.  
- Example: chess board position.  
- Basis of RL and many algorithms.

### MDP model \(\langle S, A, T, R\rangle\)
| Symbol | Meaning |
|--------|---------|
| \(S\) | Set of states |
| \(A\) | Set of actions |
| \(T(s,a,s') = P(s'\mid s,a)\) | Transition probability |
| \(R(s,a)\) | Expected reward for taking \(a\) in \(s\) |

Loop: \(s_0 \xrightarrow{a_0} (r_0,s_1) \xrightarrow{a_1} (r_1,s_2)\cdots\)

### State / action spaces & curse of dimensionality
- Huge numbers of states and actions → **curse of dimensionality**.  
- Trade-off: throw away information (abstraction) vs problem becomes **not computable**.  
- Real states often **noisy**, **partially observable**, or **hidden** (POMDP / HMM territory — e.g. speech). Awareness only for this course.

### Toy MDP intuition (Bored / Scared / Tired)
States with transition probs; actions like “do nothing” (\(R=10\)), “think about exams” (\(R=-3\)), “play sport” (\(R=-5\)) — illustrates \(T\) and \(R\) attached to state–action choices.

---

# 9. Temporal difference (TD) learning

### Offline / Monte-Carlo-style (wait for episode end)
After episode return \(R_t = G_t\) is known:
\[
\boxed{V(s_t) \leftarrow V(s_t) + \alpha\bigl(R_t - V(s_t)\bigr)}
\]
Eventually converges to true values (for visited states). Needs episodic tasks / completed returns.

### Online TD (learn during the episode) — MEMORIZE
Use **bootstrapping**: guess of next state’s value as a stand-in for the rest of the return:
\[
\boxed{V(s_t) \leftarrow V(s_t) + \alpha\bigl(r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\bigr)}
\]

**TD error**
\[
\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
\]
= difference between **new backed-up estimate** and **old** estimate (“temporal difference”).

- Works for **non-episodic** / continuing tasks.  
- Learn **on-line** without waiting for game end.

### \(TD(\lambda)\) awareness
- Can use predictions further into the future.  
- **Eligibility traces:** track recently visited states; don’t fully trust updates to states you haven’t visited.  
- Trust fades with time (like discounting). Name to recognize: \(TD(\lambda)\).

---

# 10. Q-learning (off-policy) — MEMORIZE

Always chase the **optimal** action-values; search over actions via \(\max\):

\[
\boxed{Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\Bigl(r_{t+1} + \gamma \max_{a}\, Q(s_{t+1},a) - Q(s_t,a_t)\Bigr)}
\]

| Symbol | Role |
|--------|------|
| \(\alpha\) | Learning rate |
| \(r_{t+1}\) | Immediate reward after \(a_t\) |
| \(\gamma\) | Discount |
| \(\max_a Q(s_{t+1},a)\) | Best possible continuation from next state (**optimal** backup) |

**Off-policy:** the update uses \(\max_a Q(s',a)\) — the **greedy** continuation — **even if** the behavior policy that chose the actual next action was exploratory (\(\varepsilon\)-greedy, etc.). Learn about \(\pi^*\) while behaving with another policy.

### Sarsa (on-policy) — brief contrast
Final notes: **SARSA not covered** in depth; still useful contrast if asked “on- vs off-policy”:

\[
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\bigl(r_{t+1} + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)\bigr)
\]
- Uses the **actual next action** \(a_{t+1}\) chosen by the **same** behavior policy → **on-policy**.  
- Q-learning’s \(\max\) vs Sarsa’s \(Q(s',a')\) is the exam distinction.

### Learning policy from \(Q\)
After (or while) learning \(Q\): act greedily \(\pi(s)=\arg\max_a Q(s,a)\), with \(\varepsilon\)-greedy during training.

---

# 11. NN architecture search with RNN controller (Final awareness)

> Final: raise awareness — **NN architecture search using RNN + RL**.

**Idea**
1. An NN architecture (structure + connectivity) can be described by a **variable-length string**.  
2. A **controller** network (often an **RNN**) **generates** that string (token by token).  
3. The **child** network specified by the string is **built and trained** on real data.  
4. Validation **accuracy** becomes a **reward** signal.  
5. Use that reward to compute a **policy gradient** update on the controller.  
6. Architectures with **higher accuracy** get **higher probability** under the controller’s policy.

**Controller details (as in Final text)**
- Predicts filter height, filter width, stride height, etc.  
- Predictions via **softmax** classifiers; each prediction is fed into the **next time step** as input (RNN unrolling).  
- When generation finishes → train child net → reward controller.

**Why this is RL:** controller = agent/policy; generated architecture = action (sequence); validation accuracy = reward; goal = maximize expected reward over architectures.

---

# 12. Cautionary notes & applications

### Cautionary notes (slides)
1. RL **is search**.  
2. Can be **very slow**.  
3. **No guarantee** of convergence to a **global** optimum.  
4. Can get stuck in **flat regions**.  
5. **Reward function is super critical**.

### Also hard (Final commentary)
Too many spaces to search; determining proper actions; relating actions to rewards; discounting; policy determination; exploration; …

### Key applications
Robotics (clear a room, vacuum), **finance** / trading strategies, power-system control & protection, **self-driving cars**, etc.

### Example task: pole balancing
Apply forces to keep a pole upright; stay on the track — classic episodic control demo for RL.

---

# 13. Exam angle — Lesson 9 / Final RL

1. Place RL vs supervised / unsupervised (reward, not how to fix).  
2. Draw / narrate **agent–environment** loop: state, action, reward, next state; define **episode**.  
3. Define **Policy** \(\pi(s)\) and **Value** \(V(s)\), \(Q(s,a)\) in one clear paragraph.  
4. Write discounted return \(G_t=\sum_k\gamma^k r_{t+k+1}\); **why** \(\gamma\).  
5. **Exploration vs exploitation**; \(\varepsilon\)-greedy; why **both**; why exploration **alone fails**.  
6. Relate RL **reward / actions / policy** to BPN **error / outputs / network**.  
7. Write **TD** update for \(V\) and **Q-learning** update; say Q-learning is **off-policy** (\(\max\)).  
8. Markov property in one sentence + MDP tuple \(\langle S,A,T,R\rangle\).  
9. Awareness: **RNN controller** for NN architecture search (accuracy → reward → policy gradient).  
10. List cautionary notes (slow, local optima, reward design).

---

# 14. Formula box (Lesson 9)

| Topic | Formula / fact |
|-------|----------------|
| Policy | \(a_t=\pi(s_t)\) |
| Return | \(G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k+1}\) |
| State value | \(V^\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s]\) |
| Action value | \(Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a]\) |
| Markov | \(P(r',s'\mid s,a)\) — history not needed |
| MDP | \(\langle S,A,T,R\rangle\); \(T=P(s'\mid s,a)\); \(R(s,a)\) |
| MC-style \(V\) | \(V(s_t)\leftarrow V(s_t)+\alpha(R_t-V(s_t))\) |
| TD(\(0\)) \(V\) | \(V(s_t)\leftarrow V(s_t)+\alpha(r_{t+1}+\gamma V(s_{t+1})-V(s_t))\) |
| TD error | \(\delta_t=r_{t+1}+\gamma V(s_{t+1})-V(s_t)\) |
| Q-learning | \(Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\bigl(r_{t+1}+\gamma\max_a Q(s_{t+1},a)-Q(s_t,a_t)\bigr)\) |
| Sarsa (contrast) | same with \(\gamma Q(s_{t+1},a_{t+1})\) instead of \(\gamma\max_a Q\) |
| \(\varepsilon\)-greedy | \(1-\varepsilon\): \(\arg\max_a Q\); \(\varepsilon\): random explore |
| BPN ↔ RL | reward ≈ error/target signal; actions ≈ outputs; policy ≈ network |

---

# 15. Traps

1. Reward ≠ full supervision: you learn **that** something was good/bad, **not which correction** to make.  
2. **Policy** is behavior (\(\pi\)); **value** is expected return (\(V\)/\(Q\)) — don’t swap them on the Final.  
3. \(V(s)\) averages over actions under \(\pi\); \(Q(s,a)\) scores a **specific** action — greedy policy needs \(Q\) (or equivalent).  
4. Without \(\gamma\), continuing-task returns can be **infinite / divergent**.  
5. Pure **greedy** can lock onto a suboptimal action forever.  
6. Pure **exploration** never systematically maximizes reward — **both** required.  
7. Q-learning’s \(\max_a\) makes it **off-policy**; confusing it with Sarsa’s \(Q(s',a')\) loses the point.  
8. TD bootstraps with \(V(s')\) / \(Q(s',\cdot)\) — you do **not** always wait for the full episode return.  
9. Markov: **present state** ( + action) suffices — don’t claim you must store entire history for the basic MDP model.  
10. **Reward design** dominates success; wrong \(R\) → wrong behavior even if the algorithm is “correct.”  
11. RL is **search** and can be **slow** with **no global optimum guarantee** — don’t oversell convergence.  
12. BPN analogy: reward drives learning like **error**, but RL usually has **no labeled correct action** at each step.


---


<!-- ===== 12_Part10_DecisionTrees_Bayes.md ===== -->

# CS582 Machine Learning — Ultimate Study Guide
## Lesson 10–11: Decision Trees · Random Forests · Bayes / Naive Bayes

> Everything you need for Decision Trees, Random Forests, and Bayes classifiers is in **this document**. Entropy / IG numbers and the Bayes “Officer Drew” example were recomputed in code to match the course slides and Marsland Ch. 2 / 12 / 13.

---

# 1. Decision tree idea — recursive splitting

### What it is
- A **non-linear classifier** that is **easy to use** and **easy to interpret**.
- Split classification into a **series of choices about features**, laid out as a **tree**; walk from the **root** to a **leaf**.
- Each **internal node** = test on **one attribute**.
- Each **branch** = one possible attribute value.
- Each **leaf** = a **decision** (class / action).

### Anatomy (weather-style tree from slides)
Classic “play tennis” shape: root **Outlook** → Sunny / Overcast / Rain; then **Humidity** or **Windy** → Yes / No leaves.

### Evening-activity tree (course / Marsland example)
```
                    Party?
                   /      \
                 Yes       No
                  |         |
            Go to party   Deadline
                         /   |    \
                   Urgent  Near   None
                      |      |      |
                   Study   Lazy?   Go to pub
                          /    \
                        Yes     No
                         |      |
                   Watch TV   Study
```

### Rules from a tree
Any root→leaf path becomes an **if–then** rule, e.g.:
- if there is a party **then** go to it  
- else if deadline is urgent **then** study  
- …

### How trees are built (high level)
1. Choose **which feature** to test next.  
2. Choose the **order** of features (greedy, top-down).  
3. Recurse on each subset until leaves are pure / no features left / stopping rule.

**Search properties (MEMORIZE):**
- Search over possible trees is **greedy** — **no backtracking**.
- Susceptible to **local minima** (not globally optimal tree).
- Basic ID3 **uses features until done** unless you add **pruning**.
- Can deal with **noise**: leaf label = **most common** class in that leaf.

**Inductive bias:** prefer **shorter** trees (Occam’s Razor / KISS); put the **most useful** features near the root; minimise leftover impurity.

---

# 2. Entropy \(H(S)\)

Entropy measures **impurity / uncertainty** of a set of class labels. High entropy → mixed classes; entropy **0** → pure (one class only).

### Formula (MEMORIZE — slides note: **put the minus in front**)
\[
\boxed{H(S) = -\sum_i p_i\,\log_2 p_i}
\]
with convention \(0\log 0 = 0\). Use \(\log_2\) (bits). Calculator tip: \(\log_2 p = \ln p / \ln 2\).

### Binary intuition
For proportion \(p\) positive (rest \(1-p\)):
- \(p=0\) or \(p=1\) → \(H=0\) (pure).  
- \(p=0.5\) → \(H=1\) (maximum impurity for 2 classes).

Slides’ plot: entropy vs “proportion of positive examples” is the classic inverted-U peaking at 0.5.

### Multi-class
Same formula — one \(p_i\) per class. More mixed classes → higher \(H\).

---

# 3. Information Gain (ID3 split criterion)

**Idea:** pick the feature that **most reduces** entropy of the labels.

\[
\boxed{\mathrm{IG}(S,F)=H(S)-\sum_{v\in\mathrm{values}(F)}\frac{|S_v|}{|S|}\,H(S_v)}
\]

- \(S\) = examples at the current node (parent).  
- \(S_v\) = subset where feature \(F\) has value \(v\).  
- Weighted average of child entropies = **impurity after the split**.  
- **IG** = how much impurity you **removed**.

**ID3 rule:** at each node, choose \(F\) with **largest IG**. That is “all there is to ID3” for choosing splits.

### Stopping (leaf) rules
1. All examples same label → leaf with that label.  
2. No features left → leaf with **majority** label.  
3. (Optional) max depth / min samples / validation stop.

---

# 4. Gini impurity (CART / Random Forests)

CART often uses **Gini** instead of entropy. Goal is still **purity** at leaves.

If \(N(i)\) = fraction of examples in class \(i\) at a node:
\[
\boxed{G = 1 - \sum_i N(i)^2}
\]
- Pure node (\(N(k)=1\)): \(G=0\).  
- Two-class 50/50: \(G=1-(0.5^2+0.5^2)=0.5\).

**Gini gain** (same pattern as IG):
\[
\mathrm{GiniGain}(S,F)=G(S)-\sum_v\frac{|S_v|}{|S|}\,G(S_v)
\]
Pick feature with largest Gini gain. Course RF notes: splits in the forest use **Gini gain**.

Weighted / risk variant (book): multiply class-pair terms by misclassification costs \(\lambda_{ij}\) — know it exists; exam usually wants the simple \(1-\sum N(i)^2\).

---

# 5. Worked example — entropy + IG (choose best split)

**Training set (course slides / Marsland evening data), \(n=10\):**

| Deadline? | Party? | Lazy? | Activity |
|-----------|--------|-------|----------|
| Urgent | Yes | Yes | Party |
| Urgent | No | Yes | Study |
| Near | Yes | Yes | Party |
| None | Yes | No | Party |
| None | No | Yes | Pub |
| None | Yes | No | Party |
| Near | No | No | Study |
| Near | No | Yes | TV |
| Near | Yes | Yes | Party |
| Urgent | No | No | Study |

**Class counts:** Party 5, Study 3, Pub 1, TV 1.

### Step A — parent entropy
\[
\begin{align*}
H(S)
&= -\tfrac{5}{10}\log_2\tfrac{5}{10}
   -\tfrac{3}{10}\log_2\tfrac{3}{10}
   -\tfrac{1}{10}\log_2\tfrac{1}{10}
   -\tfrac{1}{10}\log_2\tfrac{1}{10}\\
&= 0.5 + 0.5211 + 0.3322 + 0.3322\\
&= \boxed{1.6855}
\end{align*}
\]

### Step B — IG for Deadline (Urgent / Near / None)
- Urgent (3): Party 1, Study 2 → \(H=\bigl(-\tfrac23\log_2\tfrac23-\tfrac13\log_2\tfrac13\bigr)\approx 0.918\)  
  weighted: \(\tfrac{3}{10}\times 0.918 \approx 0.2755\)
- Near (4): Party 2, Study 1, TV 1 → \(H=1.5\); weighted \(\tfrac{4}{10}\times 1.5 = 0.6\)
- None (3): Party 2, Pub 1 → \(H\approx 0.918\); weighted \(\approx 0.2755\)

\[
\mathrm{IG}(S,\mathrm{Deadline})=1.6855-0.2755-0.6-0.2755=\boxed{0.5345}
\]

### Step C — IG for Party (Yes / No)
- Party=Yes (5): all **Party** → \(H=0\)
- Party=No (5): Study 3, Pub 1, TV 1 → weighted entropy \(\approx 0.6855\)

\[
\mathrm{IG}(S,\mathrm{Party})=1.6855-0-0.6855=\boxed{1.0}
\]

### Step D — IG for Lazy (Yes / No)
- Lazy=Yes (6): Party 3, Study 1, Pub 1, TV 1 → weighted \(\approx 1.0755\)
- Lazy=No (4): Party 2, Study 2 → weighted \(0.4\)

\[
\mathrm{IG}(S,\mathrm{Lazy})=1.6855-1.0755-0.4=\boxed{0.21}
\]

### Step E — choose root
\[
\mathrm{IG}(\mathrm{Party})=1.0 > \mathrm{IG}(\mathrm{Deadline})=0.5345 > \mathrm{IG}(\mathrm{Lazy})=0.21
\]
**Root = Party?**  
- Yes → leaf **Go to party** (pure).  
- No → remaining 5 rows; recurse on Deadline / Lazy.

**Data left on Party=No branch:**

| Deadline? | Party? | Lazy? | Activity |
|-----------|--------|-------|----------|
| Urgent | No | Yes | Study |
| None | No | Yes | Pub |
| Near | No | No | Study |
| Near | No | Yes | TV |
| Urgent | No | No | Study |

Continue greedily (next best feature on this subset) until leaves fill in — same IG procedure.

### Tiny binary check (optional)
Pure set \(\{+,+,+\}\): \(H=0\).  
50/50 \(\{+,+\,,-\,,-\}\): \(H=1\).  
Split that isolates all \(+\) on one side → large IG.

---

# 6. Overfitting and pruning

### Why trees overfit
- ID3 can keep splitting until it **memorises** noise.  
- Deep trees → high **variance**; train accuracy ↑, test accuracy ↓.

### Defences
1. **Limit tree size** (max depth, min samples per leaf). Extreme: **stump** = **one node only** → **weak classifier**.  
2. **Early stopping** with a **validation** set (stop adding features when val error stops improving).  
3. **Post-pruning** (usually better than early stop alone — C4.5 idea):  
   - grow the **full** tree, then  
   - **chop** subtrees / replace with majority leaves if validation error does not get worse.  
4. C4.5 **rule post-pruning**: tree → if–then rules → drop preconditions when accuracy improves; sort rules by accuracy.

### Missing data (advantage vs NN)
If a feature is missing at test time: **skip that node**, follow **all** outgoing paths (or weight them), still get a classification. Neural nets struggle with missing inputs.

### C4.5 (one-liner)
Quinlan’s improved ID3: better continuous handling + **pruning** / validation to fight overfitting.

---

# 7. Random Forests — bagging + feature randomness

### Ensemble / committee idea
- “**Two heads are better than one.**”  
- Combine many **weak** classifiers by **averaging / voting** → **strong** classifier.  
- Helps especially with **limited / noisy** data.

### Bagging = Bootstrap AGGregatING
1. Draw **bootstrap** samples from the training set: sample **with replacement**, same size \(N\) as original (some points repeated, some left out). Take many such samples (\(B\) often 50–1000+).  
2. Train one model on each sample.  
3. **Aggregate:** classification → **majority vote**.

### Random Forest = bagging of trees + random features
Given \(N\) examples and \(M\) features:

1. Create many **bootstrap** samples.  
2. Grow a **decision tree** on each sample.  
3. At **each node**, when choosing the split feature, consider only a **random subset** of \(m < M\) features (not all \(M\)). Common heuristic: \(m \approx \sqrt{M}\).  
4. Typically use **Gini gain** for splits (course notes).  
5. Predict by **majority vote** of all trees.

**Why it works:** bootstrap diversity + feature randomness → less correlated trees → **lower variance**, little change in bias. Full trees usually **need not be pruned**. Robust to heterogeneous / noisy features.

**Stumping:** use only the root question as a classifier (weak); bagging/boosting many stumps can still do well.

---

# 8. Bayes theorem and Naive Bayes

### Bayes theorem (MEMORIZE)
\[
\boxed{P(C\mid X)=\frac{P(X\mid C)\,P(C)}{P(X)}}
\]

| Name | Symbol | Meaning |
|------|--------|---------|
| **Posterior** | \(P(C\mid X)\) | Prob. of class given evidence (what you want) |
| **Likelihood** | \(P(X\mid C)\) | Prob. of evidence given class |
| **Prior** | \(P(C)\) | Prob. of class before seeing \(X\) |
| **Evidence** | \(P(X)\) | Normaliser; \(P(X)=\sum_c P(X\mid c)P(c)\) |

**Classify:** choose class with largest posterior:
\[
\hat c = \arg\max_c P(c\mid X) = \arg\max_c P(X\mid c)\,P(c)
\]
(\(P(X)\) same for all \(c\) → can ignore for \(\arg\max\).)

### Naive Bayes independence assumption
For feature vector \(X=(x_1,\ldots,x_d)\), **assume features are conditionally independent given the class**:
\[
P(X\mid C)=\prod_{k=1}^{d} P(x_k\mid C)
\]
Then:
\[
\boxed{\hat c=\arg\max_c\; P(c)\prod_{k} P(x_k\mid c)}
\]
“Naive” because features often **are** dependent — but the classifier often still works well and fights the **curse of dimensionality** (no need to estimate a full joint histogram).

**Practical tip:** products of many small probs underflow → work in **log space**: \(\log P(c)+\sum_k\log P(x_k\mid c)\).

---

# 9. Worked Bayes / Naive Bayes examples

## 9A. Bayes theorem — “Officer Drew” (course Lecture 11)

**Data (\(n=8\)):**

| Name | Sex |
|------|-----|
| Drew | Male |
| Claudia | Female |
| Drew | Female |
| Drew | Female |
| Alberto | Male |
| Karin | Female |
| Nina | Female |
| Sergio | Male |

Priors: \(P(\mathrm{Male})=\tfrac{3}{8}\), \(P(\mathrm{Female})=\tfrac{5}{8}\).  
Among males, Drew appears \(1/3\); among females, \(2/5\).  
\(P(\mathrm{Drew})=\tfrac{3}{8}\).

\[
\begin{align*}
P(\mathrm{Male}\mid\mathrm{Drew})
&= \frac{(\tfrac13)(\tfrac38)}{\tfrac38}
= \frac{0.125}{3/8}
= \boxed{\tfrac13\approx 0.333}\\[6pt]
P(\mathrm{Female}\mid\mathrm{Drew})
&= \frac{(\tfrac25)(\tfrac58)}{\tfrac38}
= \frac{0.250}{3/8}
= \boxed{\tfrac23\approx 0.667}
\end{align*}
\]

**Conclusion:** Officer Drew is more likely **Female** (\(0.250>0.125\) in the numerators; same denominator).

---

## 9B. Naive Bayes — Play Tennis / weather (independence)

**Tiny training set (classic CS style):**

| Outlook | Temp | Humidity | Windy | Play |
|---------|------|----------|-------|------|
| Sunny | Hot | High | False | No |
| Sunny | Hot | High | True | No |
| Overcast | Hot | High | False | Yes |
| Rain | Mild | High | False | Yes |
| Rain | Cool | Normal | False | Yes |
| Rain | Cool | Normal | True | No |
| Overcast | Cool | Normal | True | Yes |
| Sunny | Mild | High | False | No |
| Sunny | Cool | Normal | False | Yes |
| Rain | Mild | Normal | False | Yes |
| Sunny | Mild | Normal | True | Yes |
| Overcast | Mild | High | True | Yes |
| Overcast | Hot | Normal | False | Yes |
| Rain | Mild | High | True | No |

Counts: Play=Yes \(9/14\), Play=No \(5/14\).

**New day:** Outlook=Sunny, Temp=Cool, Humidity=High, Windy=True.  
Classify Play = Yes or No?

**Priors:** \(P(\mathrm{Yes})=\tfrac{9}{14}\), \(P(\mathrm{No})=\tfrac{5}{14}\).

**Likelihoods (counts from Yes / No rows):**

| Feature value | \(P(\cdot\mid\mathrm{Yes})\) | \(P(\cdot\mid\mathrm{No})\) |
|---------------|------------------------------|-----------------------------|
| Outlook=Sunny | \(2/9\) | \(3/5\) |
| Temp=Cool | \(3/9\) | \(1/5\) |
| Humidity=High | \(3/9\) | \(4/5\) |
| Windy=True | \(3/9\) | \(3/5\) |

**Score (ignore \(P(X)\)):**
\[
\begin{align*}
S(\mathrm{Yes})
&= \tfrac{9}{14}\cdot\tfrac{2}{9}\cdot\tfrac{3}{9}\cdot\tfrac{3}{9}\cdot\tfrac{3}{9}
= \tfrac{9}{14}\cdot\tfrac{54}{6561}
\approx 0.0053\\[4pt]
S(\mathrm{No})
&= \tfrac{5}{14}\cdot\tfrac{3}{5}\cdot\tfrac{1}{5}\cdot\tfrac{4}{5}\cdot\tfrac{3}{5}
= \tfrac{5}{14}\cdot\tfrac{36}{625}
\approx 0.0206
\end{align*}
\]

Since \(S(\mathrm{No})>S(\mathrm{Yes})\) → predict **Play = No**.

(Same procedure for spam: classes {spam, ham}; features = word presence; multiply \(P(w_i\mid\mathrm{spam})\) under independence.)

---

## 9C. Naive Bayes on the evening data (zero-probability trap)

Query: Deadline=Near, Party=No, Lazy=Yes.

\[
\begin{align*}
P(\mathrm{Party})\prod\cdots &= \tfrac{5}{10}\cdot\tfrac{2}{5}\cdot\tfrac{0}{5}\cdot\tfrac{3}{5}=0\\
P(\mathrm{Study})\prod\cdots &= \tfrac{3}{10}\cdot\tfrac{1}{3}\cdot\tfrac{3}{3}\cdot\tfrac{1}{3}=\tfrac{1}{30}\\
P(\mathrm{Pub})\prod\cdots &= \tfrac{1}{10}\cdot\tfrac{0}{1}\cdots=0\\
P(\mathrm{TV})\prod\cdots &= \tfrac{1}{10}\cdot\tfrac{1}{1}\cdot\tfrac{1}{1}\cdot\tfrac{1}{1}=\tfrac{1}{10}
\end{align*}
\]
→ pick **TV**. Zeros appear when a feature never occurs with a class (**sparse counts**). Fix in practice: **Laplace smoothing** (add-one counts) — know the issue even if smoothing wasn’t stressed in lecture.

---

# 10. Exam checklist — memorize + traps

### Must write from memory
1. \(H(S)=-\sum p_i\log_2 p_i\) (**minus sign** — slides forget it in one figure).  
2. \(\mathrm{IG}(S,F)=H(S)-\sum_v\frac{|S_v|}{|S|}H(S_v)\).  
3. Gini \(G=1-\sum N(i)^2\).  
4. ID3: greedy max-IG split; leaf = pure or majority.  
5. Overfit → prune / limit depth / validation; stump = 1-node weak learner.  
6. RF = bootstrap trees + random \(m<M\) features per split + majority vote.  
7. \(P(C\mid X)=P(X\mid C)P(C)/P(X)\); NB: \(\arg\max_c P(c)\prod_k P(x_k\mid c)\).

### Traps
1. Entropy formula without the **minus** is wrong (impurity must be ≥ 0).  
2. IG weights children by \(|S_v|/|S|\) — **do not** average child entropies unweighted.  
3. Highest **entropy** feature ≠ split choice; choose highest **information gain**.  
4. Greedy ID3 ≠ globally optimal tree; no backtracking.  
5. Deeper tree ≠ better test accuracy (overfitting).  
6. Random Forest is **not** “just bagging” — also **feature subset** at each node.  
7. NB independence is an **assumption**, not a fact — still often works.  
8. For \(\arg\max\), you may drop \(P(X)\), but if you report actual posteriors you must normalise.  
9. Zero likelihood → whole product **0** (Officer Drew fine; multi-feature NB can die without smoothing).  
10. Leaf label under noise = **majority**, not “leave unlabeled.”  
11. Bagging sample size = \(N\) **with replacement**; left-out points ≈ “out-of-bag” for free error estimates (good to know).  
12. Binary entropy max is **1 bit** at 50/50; multi-class \(H\) can exceed 1.

---

# 11. Formula box (Decision Trees + Bayes)

| Topic | Formula |
|-------|---------|
| Entropy | \(H(S)=-\sum_i p_i\log_2 p_i\) |
| Information gain | \(\mathrm{IG}(S,F)=H(S)-\sum_v\frac{\|S_v\|}{\|S\|}H(S_v)\) |
| Gini | \(G=1-\sum_i N(i)^2\) |
| Gini gain | \(G(S)-\sum_v\frac{\|S_v\|}{\|S\|}G(S_v)\) |
| ID3 pick | \(F^*=\arg\max_F \mathrm{IG}(S,F)\) |
| Bayes | \(P(C\|X)=\dfrac{P(X\|C)P(C)}{P(X)}\) |
| Evidence | \(P(X)=\sum_c P(X\|c)P(c)\) |
| MAP | \(\hat c=\arg\max_c P(c\|X)\) |
| Naive Bayes | \(\hat c=\arg\max_c P(c)\prod_k P(x_k\|c)\) |
| RF vote | majority over trees from bootstrap + random feature subsets |
| Party example | \(H(S)=1.6855\); IG Party \(1.0\), Deadline \(0.5345\), Lazy \(0.21\) → root Party |
| Officer Drew | \(P(\mathrm{F}\|\mathrm{Drew})=\tfrac23 > P(\mathrm{M}\|\mathrm{Drew})=\tfrac13\) |

---

# 12. One-page “say it out loud” summary

A **decision tree** asks one feature at a time until it can label a leaf. **ID3** always asks the question with largest **information gain** (entropy drop). **Gini** is the CART twin of entropy. Trees **overfit**; **prune** or **limit depth**, or better: grow many trees on **bootstrap** samples and, in a **Random Forest**, only allow a **random subset of features** at each split, then **vote**.

**Bayes** flips likelihood×prior into a **posterior**. **Naive Bayes** pretends features are independent given the class so the likelihood **factors**, then picks \(\arg\max_c P(c)\prod P(x_k\mid c)\).

If you can recompute the **Party / Deadline / Lazy** IG table and the **Officer Drew** posteriors without notes, you are ready for this part of the exam.


---


<!-- ===== 13_Part12_Deep_Learning.md ===== -->

# CS582 Machine Learning — Ultimate Study Guide
## Lesson 12: Deep Learning & Convolutional Neural Networks

> Everything you need for Lesson 12 / Final Deep Learning questions is in **this document**. Built from Lecture 12 slides (incl. image diagrams), Marsland Ch. 4.4.5 + Ch. 17 (AE / RBM / DBN), Andrew Ng–style sparse AE notes referenced in the slides, and Final Practice DL prompts.

---

# 0. MEMORIZE BOX — Final Practice answers (one crisp paragraph each)

### What does Deep Learning really do?
Deep Learning **automates hierarchical feature extraction**: instead of hand-crafting edges / MFCCs and then classifying, a deep net learns **representations layer by layer** from raw (or lightly processed) input, then uses those features for the task. Lower layers find simple patterns (pixels → edges); higher layers compose them into parts and whole objects.

### How does it do it?
By stacking **representation learners** — historically **autoencoders** or **RBMs**, today often **CNN / MLP blocks** — so the hidden code of one stage becomes the input of the next. Compression / generative modeling forces useful features; a top supervised head (Perceptron / softmax / linear) does classification or regression. CNNs add **local convolution + pooling** so spatial features are learned efficiently.

### Function of the **sparsity term** — what it is & how it works
Even with a bottleneck (or especially with a **wide** hidden layer), unconstrained AEs can still use many hidden units for each input. A **sparsity penalty** pushes the **average activation** \(\hat\rho_j\) of each hidden unit toward a small target \(\rho\) (e.g. \(0.05\)). Typical form: add \(\beta\sum_j\mathrm{KL}(\rho\|\hat\rho_j)\) to the reconstruction loss. Result: most hidden units stay **near zero** most of the time; only a few fire for a given input → **selective, specialized features** (edge detectors, parts), better codes, less “blurry” representations.

### Purpose of the **Pooling** layer
Pooling is **non-linear downsampling** (usually **max** over non-overlapping windows). It **shrinks spatial size** (fewer activations / weights downstream), **reduces overfitting**, and gives mild **translation robustness** (exact location of an edge inside a \(2\times2\) window matters less).

### Why do we need an **FC** layer in a CNN?
After Conv/ReLU/Pool stacks, you have **3D feature volumes**. An **FC (fully connected)** layer **flattens** those maps into a vector and **combines** high-level features globally into **class scores** (or a regression output). Convolution finds *where* patterns appear; FC decides *what* the whole image is.

### Why **not sigmoid** in DL? Good alternative?
Sigmoid derivative \(y(1-y)\le 0.25\) and ≈0 when saturated. Backprop **multiplies** one such factor **per layer** → gradients **vanish exponentially** in deep nets → early layers barely learn. Also not zero-centred; \(\exp\) is costly. **Alternative: ReLU** \(f(x)=\max(0,x)\): derivative \(1\) for \(x>0\) (no vanishing on the active path), cheap, sparse activations. Variants: Leaky ReLU; tanh is zero-centred but still saturates.

### Can DL do **classification and regression**?
**Yes.** Feature learning is agnostic to the output task. Change the **output layer and loss**: softmax / sigmoid + cross-entropy (or 1-of-N) for classification; **linear** outputs + squared error for regression — same idea as an MLP, just deeper / with Conv stacks.

---

# 1. Setup — what changes vs classical ML / MLP

### Classical pipeline (slides)
1. Collect data  
2. **Hand-engineered features** (edges from pixels, MFCC from audio, SIFT, …)  
3. Train a classifier (SVM, shallow NN, …)

### Deep Learning idea
- Can a Neural Network also **extract key features automatically**?  
- **Answer: YES.** That *is* Deep Learning.
- Especially successful for **image / object recognition**; also speech, etc.
- Trend slide: NN does **both** feature extraction **and** classification (end-to-end), replacing pipelines like **SIFT → SVM → maxpool**.

### Comparison curve (Andrew Ng slide)
- On large amounts of **tagged numerical / sensory** data, **Deep Learning keeps improving** while older algorithms **plateau**.  
- Course caveat: for some **unstructured “semantics-only”** data that cannot be well represented numerically, DL is less magic — semantics ≠ free pixels.

---

# 2. What Deep Learning really does / how (hierarchical representation learning)

### Representation learning
Each layer maps its input to a new **representation** that is more useful for higher layers and for the final task.

### Feature hierarchy (MEMORIZE this chain)
\[
\textbf{pixels}\;\rightarrow\;\textbf{edges}\;\rightarrow\;\textbf{object parts}\;\rightarrow\;\textbf{object models}
\]
- Layer 1: edge / blob / color detectors (Gabor-like filters).  
- Layer 2: eyes, noses, mouth corners, oriented part combinations.  
- Layer 3+: whole faces / objects.  
(Slide visuals: stapler → edges; face photo → second-layer parts → third-layer faces.)

### How stacking works (Marsland Fig. 17.12–17.13)
1. Train one **autoencoder** (or RBM) on raw input → hidden code = compressed features.  
2. Use that **hidden activation** as input to the **next** autoencoder / RBM.  
3. Repeat → higher-order correlations / more abstract features.  
4. Optionally put a **supervised head** on top (Perceptron / softmax) for classification or regression.

**Core slogan for the exam:** Deep Learning = **hierarchical automated feature learning** (representation learning across layers), not “just a bigger MLP for fun.”

---

# 3. Why deep nets were historically hard; vanishing gradients; ReLU

### Theoretical vs practical
- A shallow MLP can approximate many functions **in theory**, but complex vision/speech tasks need **composition of many nonlinear transforms**.  
- Making one huge shallow net needs **enormous width** → huge #weights → data hunger + hard optimization.  
- **Deep** nets compose features naturally, but plain backprop through many sigmoid layers failed historically.

### Vanishing gradient (why sigmoid fails in DL) — MEMORIZE
Sigmoid:
\[
g(h)=\frac{1}{1+e^{-\beta h}},\qquad g'(h)=\beta\,g(h)\bigl(1-g(h)\bigr)
\]
- \(g'\le \beta/4\) (for \(\beta=1\): **max \(0.25\)**).  
- At saturation (\(g\approx 0\) or \(1\)), \(g'\approx 0\).  
- Hidden delta in backprop:
\[
\delta_h(j)=a_j(1-a_j)\sum_k w_{jk}\delta_o(k)
\]
Each layer multiplies by another \(a(1-a)\). Over \(L\) layers gradients to early weights scale like \(\sim (1/4)^L\) (order of magnitude) when saturated → **early layers stop learning** (**vanishing gradients**).  
Also: sigmoid outputs not zero-centred; \(\exp\) expensive.

### Why deep training was hard historically (course-level list)
1. **Vanishing / exploding gradients** with saturating activations.  
2. **Local minima / plateaus** in a huge nonconvex landscape.  
3. Need for **good initializations** and lots of data / compute.  
4. Pure stacked AEs: lower layers get **little supervised error** from the top task (book’s critique) → led to **greedy layerwise pretraining** (AE/RBM), then fine-tuning.  
Modern practice often trains deep nets end-to-end with **ReLU**, careful init, BatchNorm, etc. — but the Final still wants the **sigmoid vs ReLU** story.

### ReLU — the good alternative (MEMORIZE)
\[
\boxed{\mathrm{ReLU}(x)=\max(0,x)},\qquad
\mathrm{ReLU}'(x)=\begin{cases}1 & x>0\\ 0 & x<0\end{cases}
\]
- On the active path, derivative **1** → **no vanishing** from activation saturation.  
- Cheap; induces **sparse** activations (many zeros).  
- Used after every Conv in the course CNN diagram: **CONV → RELU → POOL**.  
- **Leaky ReLU:** small slope for \(x<0\) to avoid “dead” units.  
- **tanh** is zero-centred but still saturates → still can vanish.

---

# 4. Autoencoders (auto-associative networks)

### Definition
Train an MLP so **target = input** (reconstruction). Network learns to **reproduce** \(x\) through a hidden representation \(h\).

### Encoder / decoder / bottleneck
\[
\begin{aligned}
h &= f_{\mathrm{enc}}(x) &&\text{(encoder: input → hidden code)}\\
\hat x &= f_{\mathrm{dec}}(h) &&\text{(decoder: code → reconstruction)}
\end{aligned}
\]
- **Bottleneck:** \(\dim(h) < \dim(x)\) → forced **compression**.  
- Must keep **informative features**, throw away noise.  
- Reconstruction loss (typical):
\[
E=\frac12\|x-\hat x\|^2
\quad\text{(or cross-entropy for binary pixels)}
\]

### After training (slides)
- Can **remove** input-side weights for storage applications; keep **hidden codes + decoder weights**.  
- Feed compressed codes → rebuild **noise-free** input (**denoising / pattern completion**).  
- **Linear** hidden units → recovers **PCA**.  
- Nonlinear hidden units → nonlinear manifold features.

### Stacked autoencoders (Deep Learning core idea on slides)
- Hidden layer of AE\(_k\) becomes input of AE\(_{k+1}\).  
- Decoder halves shown in lighter grey / dotted weights in Fig. 17.13.  
- Top: add supervised layer for classification/regression.  
- **Limitation (book):** each AE only sees reconstruction error; **no direct label signal** to lower layers until you fine-tune or switch to generative stacks (RBM/DBN).

### Diagram to remember
Input (e.g. digit “6”) → wide input layer → **narrow bottleneck** → wide output reconstructing the “6”. Learned first-layer weights look like **oriented edges**.

---

# 5. Sparsity term (Final asks this!)

### Motivation
Bottleneck forces compression, but you may also want a **larger** hidden layer that still behaves like a small active set — **overcomplete sparse coding** (Andrew Ng sparse AE theme used in this course).

### What the sparsity term **is**
An **extra penalty** added to the reconstruction loss that encourages each hidden unit’s **mean activation over the training set** to stay near a small constant \(\rho\).

Let \(a_j^{(i)}\) be activation of hidden unit \(j\) on example \(i\). Average:
\[
\hat\rho_j=\frac1m\sum_{i=1}^m a_j^{(i)}
\]
Target sparsity \(\rho\) (e.g. \(0.05\)). Penalty (KL form — standard Ng notes):
\[
\sum_{j=1}^{n_h}\mathrm{KL}(\rho\|\hat\rho_j)
=\sum_{j=1}^{n_h}\Biggl[\rho\log\frac{\rho}{\hat\rho_j}+(1-\rho)\log\frac{1-\rho}{1-\hat\rho_j}\Biggr]
\]
Total objective:
\[
\boxed{J=E_{\mathrm{recon}}+\beta\sum_j\mathrm{KL}(\rho\|\hat\rho_j)}
\]
(\(\beta>0\) controls strength.)

### How it **works**
1. If \(\hat\rho_j>\rho\): unit fires too often → KL large → gradient **suppresses** that unit’s average firing.  
2. If \(\hat\rho_j<\rho\): unit almost never fires → penalty pushes it to fire **sometimes**.  
3. Equilibrium: each unit is a **specialist** — silent on most inputs, strongly active on a rare pattern (edge orientation, eye corner, …).  
4. Visual intuition: in a grid of hidden units, for one image only a **few** “light up” (sparse code).

### Why it helps Deep Learning
- More **interpretable / selective** features.  
- Better generalization than dense codes that smear information across all units.  
- Complements bottleneck compression; works even when hidden dim ≥ input dim.

---

# 6. RBM and DBN (course / Marsland Ch. 17 level)

### Restricted Boltzmann Machine (RBM)
- Two layers: **visible** \(v\) and **hidden** \(h\).  
- **No within-layer** connections (that is the “Restricted”).  
- **Symmetric** weights \(W\) between \(v\) and \(h\).  
- Stochastic binary units (sigmoid probabilities):
\[
\boxed{
p(h_j=1\mid v)=\sigma\!\Bigl(b_j+\sum_i v_i w_{ij}\Bigr),\quad
p(v_i=1\mid h)=\sigma\!\Bigl(a_i+\sum_j w_{ij}h_j\Bigr)
}
\]
- Train with **Contrastive Divergence (CD)** — approximate max-likelihood by a few steps of alternating Gibbs sampling (data “positive” phase vs reconstruction “negative” phase):
\[
\Delta w_{ij}\propto \langle v_i h_j\rangle_{\mathrm{data}}-\langle v_i h_j\rangle_{\mathrm{recon}}
\]
- Generative: sample \(h\) → reconstruct \(v\) (**pattern completion**, like Hopfield but bipartite).  
- Supervised variant: add **label** units on the visible side.

### Why RBM vs autoencoder for stacking
| | Autoencoder | RBM |
|--|-------------|-----|
| Weights | Directed encoder/decoder | **Symmetric** |
| Run backward? | Separate decoder | Same \(W\) generates \(v\) from \(h\) |
| Stacking issue | Hard to send supervised signal down | Can push info **down** the stack and refine lower models |

### Deep Belief Network (DBN)
- Stack of **unlabelled RBMs** + **labelled RBM on top**.  
- **Greedy layerwise pretraining:**  
  1. Train RBM\(_1\) on data.  
  2. Sample / take hidden activations → train RBM\(_2\).  
  3. Repeat; top RBM uses **labels**.  
- Then **wake–sleep / up–down**: untie **recognition** vs **generative** weights (except top); update generative going up, recognition going down.  
- Use: **recognize** (clamp visible, sample up) or **generate** (sample down).  

**Exam one-liner:** DBN = **stacked RBMs** trained **greedily**, then refined so the stack is both a good **feature hierarchy** and a **generative model**.

---

# 7. Convolutional Neural Networks (CNNs / ConvNets)

### Why not a plain fully connected NN on images?
1. **Too many connections / parameters** — does not scale (e.g. \(200\times200\times3\) flattened).  
2. Ignores **spatial structure** (except SOM-like ideas).  
3. CNN uses **local receptive fields** + **3D volumes** (width × height × depth), biologically motivated (vision).

### Convolution (math)
Continuous (slides):
\[
\boxed{(f*g)(t)=\int_{-\infty}^{\infty} f(\tau)\,g(t-\tau)\,d\tau}
\]
Discrete image form: slide a **filter / kernel** \(w\) over local patches; at each location output
\[
\boxed{z = w^T x_{\mathrm{patch}} + b}
\]
Example: \(5\times5\times3\) filter on RGB → **75**-dimensional dot product + bias → **1 number** in a feature map.

### Filters / kernels & feature maps
- Each filter detects one pattern (horizontal edge, color blob, …).  
- Sliding one filter over the image → one **feature map** (2D activation map).  
- Many filters → **depth** of the output volume.  
- **Parameter sharing:** same \(w\) at every location → translation structure; AlexNet-style first-layer filters look like oriented edges + color blobs.  
- Depth column: several neurons at one \((x,y)\) look at the **same** receptive field with different filters.

### CNN as 3D volumes
Every layer: **height × width × depth**.  
Input example: \(32\times32\times3\) (CIFAR-style RGB).  
Each layer transforms one volume into the next; final volume → class scores.

### Typical block (slides / activation demo)
\[
\textbf{CONV}\;\rightarrow\;\textbf{RELU}\;\rightarrow\;\textbf{CONV}\;\rightarrow\;\textbf{RELU}\;\rightarrow\;\textbf{POOL}
\]
(repeat) → **FC** → class scores (car / truck / airplane / …).

### Stride, padding, output size — MEMORIZE
\[
\boxed{n_{\mathrm{out}}=\frac{W-F+2P}{S}+1}\quad\text{(must be an integer)}
\]
| Symbol | Meaning |
|--------|---------|
| \(W\) | input spatial size |
| \(F\) | filter / receptive field size |
| \(P\) | zero-padding (zeros around border) |
| \(S\) | **stride** (step of the window) |

Examples (slides): \(W=5,F=3,P=1,S=1\) → \(5\); same with \(S=2\) → \(3\).  
Invalid: \(W=10,F=3,P=0,S=2\) → \(4.5\) — neurons don’t fit; pad/crop or change \(S\).

**Zero-padding:** preserve spatial size when desired (\(P\) chosen so \(n_{\mathrm{out}}=W\) for \(S=1\)).

### Pooling — purpose (Final asks!)
- Form of **non-linear downsampling**.  
- **Max pooling** most common: partition into non-overlapping rectangles; output **max** in each.  
- Slide example \(4\times4\rightarrow 2\times2\) with \(2\times2\) max, stride 2:
\[
\begin{bmatrix}1&0&2&3\\4&6&6&8\\3&1&1&0\\1&2&2&4\end{bmatrix}
\;\xrightarrow{\max}\;
\begin{bmatrix}6&8\\3&4\end{bmatrix}
\]
- **Purpose (write this):** reduce spatial resolution → **fewer parameters / weights**, less compute, **help prevent overfitting**, mild **invariance** to small shifts.

### Fully Connected (FC) layer — purpose (Final asks!)
- After last Pool/Conv, **flatten** \(H\times W\times D\) into a vector.  
- Dense layer(s) map that vector to **\(C\) class logits** (or continuous outputs).  
- **Why needed:** convolutional features are still **spatially organized**; classification needs a **global decision** that mixes all high-level detectors. FC (or global average pooling + linear, in modern nets) is that mixer. Course answer: **FC combines learned features into final class scores**.

### End-to-end story for images
Raw pixels → shared local filters (edges) → ReLU → pool → deeper filters (parts) → … → FC → softmax scores. **Automated spatial feature extraction + classification.**

---

# 8. Classification **and** regression with DL

Deep nets are **universal function approximators** with a learned feature front-end.

| Task | Output layer | Typical loss |
|------|--------------|--------------|
| Binary classification | 1 sigmoid | cross-entropy / logistic |
| Multi-class | softmax, 1-of-N | cross-entropy |
| Regression | **linear** outputs | \(\tfrac12\|y-t\|^2\) |

Same Conv/AE/MLP body; **only the head and loss change**. So: **DL can do both** — the “deep” part is representation learning; the top is ordinary supervised prediction.

---

# 9. Regularization in this lecture’s material

Slides explicitly: pooling **reduces weights/parameters and helps prevent overfitting**.

Related ideas you should connect (course-consistent, even if “dropout” is not a full Lecture 12 slide deck item):
- **Bottleneck + sparsity** → capacity control on codes.  
- **Weight decay** / small weights (as in MLP).  
- **Early stopping** on validation (MLP practice carries over).  
- **Dropout** (if asked generally): randomly zero activations during training so the net cannot co-adapt features → ensemble-like regularization; at test time scale weights. Not required to derive if not on slides — know pooling + sparsity as **this lecture’s** regularizers.

---

# 10. Exam angle — Lesson 12 / Final DL

Be ready to write, without notes:

1. **What DL does / how** — hierarchical representation learning; pixels→edges→parts→objects; stack AE/RBM/CNN.  
2. **Sparsity term** — KL (or similar) pushing mean hidden activation to small \(\rho\); selective features.  
3. **Pooling purpose** — downsample, fewer params, less overfit, shift robustness.  
4. **FC purpose** — map feature volumes to class scores / final decision.  
5. **Sigmoid vs ReLU** — vanishing gradients; ReLU derivative 1 when active.  
6. **Classification and regression** — yes; change output + loss.  
7. **AE:** target=input, bottleneck, encoder/decoder, stack.  
8. **RBM/DBN:** bipartite symmetric CD training; greedy stack + optional wake–sleep.  
9. **CNN size formula** \((W-F+2P)/S+1\); stride; padding; \(w^Tx+b\) local filter.  
10. Contrast **hand features + SVM** vs **end-to-end DL**.

---

# 11. Formula box (Lesson 12)

| Topic | Formula / fact |
|-------|----------------|
| AE target | \(\hat x \approx x\); bottleneck \(\dim(h)<\dim(x)\) |
| Recon loss | \(E=\tfrac12\|x-\hat x\|^2\) |
| Sparsity | \(J=E_{\mathrm{recon}}+\beta\sum_j\mathrm{KL}(\rho\|\hat\rho_j)\); \(\hat\rho_j=\tfrac1m\sum_i a_j^{(i)}\) |
| Sigmoid der | \(g'=g(1-g)\le 1/4\) → vanishing in depth |
| ReLU | \(\max(0,x)\); \(f'=1\) if \(x>0\) |
| RBM | \(p(h_j=1\mid v)=\sigma(b_j+\sum_i v_i w_{ij})\) (sym. \(W\)); CD train |
| DBN | stack RBMs greedily; top labelled; then wake–sleep |
| Conv continuous | \(\int f(\tau)g(t-\tau)\,d\tau\) |
| Conv discrete | \(z=w^T x_{\mathrm{patch}}+b\) (e.g. \(5{\times}5{\times}3\Rightarrow 75\)-D) |
| Spatial size | \((W-F+2P)/S+1\) integer |
| Max pool | max over \(F\times F\) window (e.g. \(2\times2\), stride 2) |
| Hierarchy | pixels → edges → parts → objects |
| Class / reg | softmax/sigmoid vs **linear** head |

---

# 12. Traps

1. DL is **not** “MLP with more layers” only — the exam wants **automated hierarchical features**.  
2. **Bottleneck ≠ sparsity.** Bottleneck = fewer units; sparsity = **penalty on average activation** (can use wide layers).  
3. Sparsity \(\rho\) is a **target mean firing rate**, not “set weights to zero.”  
4. Sigmoid is fine for **shallow** nets / output probabilities; it fails for **deep** hidden stacks because of **vanishing gradients**.  
5. ReLU derivative is **0** for \(x<0\) (dead ReLU risk) — still preferred over sigmoid in depth.  
6. Pooling **downsamples**; it does **not** learn a filter (no weights in standard max-pool).  
7. FC is **not** optional conceptually for “why CNN classifies” in this course’s wording — it turns maps into **decisions**.  
8. Convolution filter depth = **input depth** (RGB filter is \(F\times F\times 3\)).  
9. Output size formula must be **integer** — invalid stride/pad is a real design constraint.  
10. Stacked AE alone: lower layers may **not** get label gradients until fine-tuning; DBN/RBM story exists **because** of that.  
11. DL **can** regress — don’t say “DL = classification only.”  
12. Parameter sharing ≠ “one weight total for the whole net” — **one filter shared across locations**; many filters still.

---

# 13. Quick self-check (cover answers, speak out loud)

1. State pixels→…→objects.  
2. Draw AE with bottleneck; mark encoder/decoder.  
3. Write \(J\) with sparsity KL term; explain \(\rho\) vs \(\hat\rho_j\).  
4. Why sigmoid fails in depth; write ReLU.  
5. RBM: bipartite + CD one-liner; DBN stack one-liner.  
6. Compute \((W-F+2P)/S+1\) for \(W=5,F=3,P=1,S=1\) and \(S=2\).  
7. Max-pool the \(4\times4\) slide example mentally.  
8. One sentence each: pooling purpose; FC purpose.  
9. “Can DL do regression?” — answer in one breath.

If those nine are crisp, you are ready for the Deep Learning portion of the Final.


---


<!-- ===== 14_Final_QuestionBank_L6-L12.md ===== -->

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


---

