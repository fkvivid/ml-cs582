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
