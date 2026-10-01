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
