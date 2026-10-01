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
