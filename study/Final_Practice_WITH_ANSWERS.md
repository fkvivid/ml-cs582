# Sample Final Questions — with answers

> From `Final_Practice_up_DE.docx`. Full writeups also live in `14_Final_QuestionBank_L6-L12.md` and Parts 6–12.

---

## Short questions

### 1. Learning equation for unsupervised learning in an NN?
Winner-only competitive rule:
\[
\Delta w_{ij}=\eta(x_j-w_{ij})\quad\Leftrightarrow\quad\mathbf{w}\leftarrow\mathbf{w}+\eta(\mathbf{x}-\mathbf{w})
\]
Normalize weights (and inputs) so large \(\|\mathbf{w}\|\) does not fake a win.

### 2. Function of the Sparsity term in Deep Learning? How does it work?
Penalty so hidden units are **rarely active** (avg activation near small \(\rho\)). Often \(\beta\sum_j\mathrm{KL}(\rho\|\hat\rho_j)\) added to reconstruction loss → specialized features, less redundancy.

### 3. Relate reward and actions in RL with BPN?
| RL | BPN |
|---|---|
| Action | Network output |
| Policy | Input→output map (weights) |
| Reward / return | Like target / error signal |
| Improve policy | Like weight update |

### 4. Policy and Value in RL?
- **Policy** \(\pi(s)\) / \(\pi(a|s)\): which action in state \(s\).  
- **Value** \(V^\pi(s)\): expected discounted return from \(s\) under \(\pi\). \(Q(s,a)\) from taking \(a\) then \(\pi\).

### 5. Exploitation and Exploration? Exploration alone?
Need **both**: exploit best-known actions; explore to improve estimates. Exploration alone never locks onto good behavior → poor cumulative reward.

### 6. Purpose of Pooling in DL?
Downsample feature maps (max/avg): smaller size, mild translation tolerance, less compute; keep strong responses.

### 7. Why FC layer in CNN?
After conv/pool, flatten and **fully connect** to mix features into class scores / regression output.

### 8. Why not Sigmoid in DL? Alternative?
Saturates → **vanishing gradients** in deep stacks. **ReLU** \(\max(0,z)\).

### 9. Role of neighborhood function in SOM?
Which neurons **besides BMU** update toward the input. Large early → topology; shrink later → fine tune. Without it → plain competitive learning.

---

## Agents

### A. Definitions (handout)

- **Agent:** perceives via sensors, acts via actuators.  
- **Agent function:** \(f:\mathcal{P}^*\to\mathcal{A}\) (percept history → action).  
- **Agent program:** implements / approximates that function.  
- **Rationality:** best (expected) outcome.  
- **Autonomy:** handle unforeseen / partial or wrong prior knowledge.  
- **Reflex:** current percept only.  
- **Model-based:** uses world model.  
- **Goal-based:** achieve goals.  
- **Utility-based:** maximize utility.  
- **Learning:** improves over time.

### B. PEAS (handout)

| | Robot soccer | Internet book-shopping |
|---|---|---|
| **P** | Score, mistakes, follow rules, … | Min cost/time; interesting books |
| **E** | Ball, field, teammates, competitors | Internet, browsers, customer |
| **A** | Legs, arms, head, speakers | Speakers, display, place order |
| **S** | Camera, mic, touch, comm. sensors | Keyboard, mouse, mic, pages, clicks |

### C. Letter agent \(30\times30\) binary
- States: \(2^{900}\)  
- Functions to 26 letters: \(26^{2^{900}}\)  
- Practical features: strokes / holes / projections / learned CNN features — not raw lookup tables.

---

## Reinforcement Learning

- Search over state/action space to raise reward.  
- Maximize **discounted** return \(G_t=\sum_k\gamma^k R_{t+k+1}\).  
- Learning unit: update policy / value (e.g. TD, Q-learning) from experience.

**RNN architecture search (awareness):** controller RNN emits architecture string → train child → validation accuracy → policy gradient on controller.

---

## Unsupervised Learning

Work Lab 6; answers in `08_Part6_Unsupervised_Learning.md` (k-means final: \(\{1,2\}\) @ \((1.25,1.5)\), \(\{3..7\}\) @ \((3.9,5.1)\)).

---

## GA — sample problem & solution (handout + corrections)

Solve \(a+2b+3c+4d=30\); genes in \([0,30]\); pop 4; **2 iterations**.

Fitness: minimize \(r=a+2b+3c+4d-30\); use \(1/|r|\).

**Init:** P1\(\{1,10,20,30\}\), P2\(\{5,6,7,21\}\), P3\(\{10,11,21,29\}\), P4\(\{7,8,9,10\}\).

### Handout walkthrough (as printed — arithmetic uses wrong coeffs on P2–P4)

Handout treated terms like \(6\times10\) instead of \(2\times6\). Printed residuals ≈ 71, 805, 1380, 537 → parents **P1 & P4**; crossover at 3rd gene → \(\{1,10,9,10\}\), \(\{7,8,20,30\}\); mutate → \(\{1,10,9,20\}\), \(\{7,19,20,30\}\); then iteration 2 with P2, P3, Q1, Q2.

### Corrected residuals (prefer these)

| ID | \(r\) | \(1/|r|\) |
|---|---|---|
| P1 | **171** | 0.00585 |
| P2 | **92** | 0.01087 |
| P3 | **181** | 0.00552 |
| P4 | **60** | **0.01667** |

Top-2 parents: **P4, P2**. Continue crossover / mutation / 2nd iteration the same way. Full detail: `10_Part8_Genetic_Algorithms.md`.

---

## Deep Learning

1. **What it does:** hierarchical / automatic feature learning.  
2. **How:** stacked layers (AE/RBM/CNN/MLP) learn representations; top head does the task.  
3. **Class and regression?** **Yes** — change output layer + loss (softmax/CE vs linear/MSE).

Detail: `13_Part12_Deep_Learning.md`.

---

## Future Technologies

Not covered — skip unless asked.
