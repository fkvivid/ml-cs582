# CS582 Machine Learning — Study Guide
## Lectures 1–3: Introduction · Perceptron · Multi-Layer Perceptron (MLP) + Labs 2 & 3

> Split from the ultimate study guide. See also the other parts in this `study/` folder.
>
> **Sources:** course slides, labs (incl. Lab 2 solutions), Final Practice, Marsland textbook Ch. 2–4.
> Every worked-example number was re-computed with code.

## Companion reference (Parts 4–6)

Use after studying Parts 1–3. Contains the question bank, formula sheet, and common traps.

---

# PART 4 — QUESTION BANK WITH ANSWERS (test yourself)

**Lecture 1**

1. **What is ML? Contrast it with traditional programming.** — ML gives computers the ability to learn from data (programs itself). Traditional: data + program → output. ML: data + desired output → program/model.
2. **Name the 4 types of ML.** — Supervised (inputs + targets), Unsupervised (no targets, find similarities/groups), Reinforcement (told when wrong, not how to fix), Evolutionary (adaptation like biological evolution).
3. **List the steps of the ML process.** — Data collection & preparation, feature selection, algorithm choice, parameter & model selection, training, evaluation.
4. **Regression vs. classification?** — Regression predicts a continuous value; classification assigns one of N **discrete** classes.
5. **Why do we need ML rather than equations?** — No algorithms for many data-driven problems; closed-form solutions don't exist for complex, high-dimensional, non-linear problems; we may not even have equations (self-driving); existing algorithms may be too slow for big data.
6. **Why use squared error?** — So positive and negative errors don't cancel; smooth and differentiable.
7. **Derive the least-squares m and c.** — See §1.9 ($m = \frac{nS_{xy} - S_xS_y}{nS_{xx} - S_x^2}$, $c = \frac{S_y - mS_x}{n}$).
8. **What is generalization?** — Performing well on unseen data from the same distribution, i.e., learning the underlying function, not memorising.
9. **What % of time do data scientists spend on data prep?** — ~80%.
10. **Supervised/unsupervised × discrete/continuous table?** — Classification / Regression / Clustering / Dimensionality reduction.

**Lecture 2**

11. **Why use ANNs?** — Massive parallelism (efficiency), distributed representation (robustness, graceful degradation), intelligence emerging from simple units.
12. **Why must the brain use massive parallelism?** — Neurons switch in ms; tasks take ~0.1 s → only ~100 serial steps possible.
13. **State Hebb's rule.** — Neurons that fire together strengthen their synapse (optionally: weaken if they don't).
14. **Write the McCulloch–Pitts model; list its simplifications.** — $h = \sum w_ix_i$, fire if $h \ge \theta$. Simplifications: linear sum, no refractory period, single output instead of a spike train, runs on a clock.
15. **Where does learning happen in a neural network?** — In the weights (the synapses), between the neurons.
16. **Write the perceptron learning rule and explain each term.** — $w_{ij} \leftarrow w_{ij} - \eta(y_j - t_j)x_i$; η = learning rate; $(y - t)$ = error sign (too big/too small); $x_i$ = only change weights whose input contributed, and handle sign.
17. **Why do we need a bias input?** — So the threshold can be learned; without it, an all-zero input can't be classified either way. Fixed input −1 with a learnable weight = threshold.
18. **Effect of the learning rate?** — Large → unstable; small → slow but stable & noise-resistant; typical 0.1–0.4.
19. **What is an epoch?** — One pass through all the training data.
20. **Design perceptrons for AND, OR, NOT, NAND, NOR.** — AND (1,1,−1.5), OR (1,1,−0.5), NOT (−1, +0.5), NAND (−1,−1,+1.5), NOR (−1,−1,+0.5) [format: weights, bias; fire if sum > 0].
21. **What does a perceptron compute geometrically?** — A linear decision boundary (hyperplane) $w\cdot x = 0$; w is perpendicular to it.
22. **Why can't a perceptron learn XOR? What are the 2 fixes?** — XOR is not linearly separable. Fix 1: more complicated network (MLP). Fix 2: more complicated input (add a 3rd dimension, e.g. $x_1x_2$ or a feature that is 1 only at (0,0)).
23. **State the convergence and cycling theorems.** — Linearly separable → converges in finite steps (≤ $1/\gamma^2$, or $R^2/\gamma^2$). Not separable → weights eventually repeat → infinite loop.
24. **Prove convergence with ‖x‖ ≤ R.** — §2.19: $t\gamma \le \|w\| \le \sqrt{t}R$ ⇒ $t \le R^2/\gamma^2$.
25. **What was the impact of Minsky & Papert?** — Showed perceptron limits (XOR) → NN research stalled ~20 years, symbolic AI dominated until backprop (1986).
26. **Batch vs sequential?** — §2.15/§3.14.
27. **Perceptron performance?** — High bias (linear) but more expressive than pure conjunctions/disjunctions/M-of-N; converges quickly on separable data; partially-converged weights still usable.

**Lecture 3**

28. **Why do we need multi-layer networks?** — Most real problems are non-linear; MLPs can represent arbitrary functions (XOR etc.).
29. **Why is training an MLP harder than a perceptron?** — More weights, and we don't know which layer's weights are wrong; hidden nodes have no targets.
30. **Write the gradient descent update.** — $\Delta w = -\eta\,\partial E/\partial w$.
31. **Why can't we use the threshold function in an MLP? What properties do we want?** — Not differentiable. Want: differentiable, saturating at the ends, quick change between saturation values → sigmoid.
32. **Derive the sigmoid derivative.** — $y(1 - y)$ (§3.7).
33. **Write the backprop equations and explain each part.** — §3.8. Output delta = error × sigmoid derivative. Hidden delta = sigmoid derivative × (sum of output deltas weighted by the connecting weights). Update = −η × delta × incoming activation.
34. **What is the Delta Rule? The Generalized Delta Rule?** — Delta rule: gradient-descent weight update for single-layer (linear) units, $\Delta w = \eta(t - o)i$. Generalized: same form $\Delta w_{ji} = \eta\,\delta_{pj}\,o_{pi}$ with δ defined as $-\partial E/\partial net$, computed at the output with $f'$ and at hidden units by back-propagating $\sum_k \delta_k w_{kj}$ — for multi-layer nets with semi-linear (differentiable, non-decreasing) activations.
35. **How should weights be initialised and why?** — Small random ± values (≈ ±1/√n): too large → saturation, tiny gradients; too small → linear; identical → symmetry never breaks.
36. **What is a local minimum; how do we deal with it?** — A point lower than its neighbours but not the lowest; use multiple random restarts, momentum, sequential/stochastic updates.
37. **What is momentum and why use it?** — Adds α × the previous weight change (α ≈ 0.9): escapes small local minima, smoother/more stable, allows a smaller η, faster.
38. **How many hidden layers/nodes?** — No good answer: usually 1–2 hidden layers (at most 3 layers); try several sizes and pick by validation error; one hidden layer is theoretically sufficient (Universal Approximation).
39. **How much training data?** — ≥ 10× the number of weights $(L+1)M + (M+1)N$.
40. **Roles of the training, validation and test sets? Ratios?** — Train: fit weights. Validation: monitor learning / early stopping / model choice. Test: final unbiased check. 50:25:25 or 60:20:20; cross-validation if little data.
41. **What is early stopping?** — Stop training at the minimum of the validation error, where generalization is best.
42. **Define overfitting and undertraining.** — Overfitting: memorising training data including noise → poor on new data. Undertraining: hasn't learned the function → poor on new data too.
43. **Explain leave-some-out multi-fold cross-validation.** — Split into K folds; train different models each holding out different folds for validation/testing; choose the model with the lowest validation error.
44. **Define precision, recall, F1. When is accuracy misleading?** — §3.20; with highly unbalanced classes → use F1.
45. **What is an auto-associative network and what is it used for?** — Output = input through a smaller hidden (bottleneck) layer → compression, feature learning, denoising; linear version = PCA; later used in Deep Learning.
46. **Which output activation for regression vs classification?** — Regression/time series → linear; 2-class → sigmoid; multi-class → softmax with 1-of-N targets.
47. **Decision regions of 1, 2, 3-layer networks?** — Half-plane; convex regions; arbitrary regions.
48. **How does an MLP approximate any function?** — Sigmoids → ridges → bumps; the output layer sums bumps (local representation).
49. **Key differences between ANNs and the brain?** — Architecture, feedback paths, learning rules, bias/threshold function, semantics.
50. **Describe the general procedure for building a NN** — §3.26's 8 steps (specify problem → simplest network → find weights → test generalization → iterate back to 3, 2, 1).

**Bridge to the final (Deep Learning questions build on this material)**

51. **"Why can't we use the sigmoid in deep learning? What's a good alternative?"** — The sigmoid derivative $y(1-y)$ is at most **0.25** and ≈ 0 when saturated. Backprop multiplies one such factor **per layer** (see the hidden delta), so in deep nets the gradient shrinks exponentially toward the early layers (**vanishing gradient**) → early layers barely learn. Also not zero-centred, and exp is costly. **Alternative: ReLU** $f(x) = \max(0, x)$ (derivative 1 for x > 0 → no vanishing, cheap, sparse activations); variants: Leaky ReLU; tanh is zero-centred but still saturates.
52. **"Can Deep Learning do classification or regression?"** — Yes — like an MLP: sigmoid/softmax outputs + 1-of-N for classification, linear outputs for regression; DL is basically a deep MLP-style network that learns its own features layer by layer.
53. **"Can you relate reward and actions in RL with BPN?"** — In BPN, the error (target − output) drives weight changes through gradient descent; in RL there is no target, only a **reward** — a NN can be trained to predict values/choose actions and the reward (or the temporal-difference error) plays the role of the error signal that is back-propagated.

---

# PART 5 — ONE-PAGE FORMULA SHEET

| Topic | Formula |
|---|---|
| Linear regression | $m = \dfrac{nS_{xy} - S_xS_y}{nS_{xx} - S_x^2}$, $c = \dfrac{S_y - mS_x}{n}$; $m = r\,s_Y/s_X$, $c = M_Y - mM_X$ |
| Matrix least squares | $\beta = (X^TX)^{-1}X^Tt$ |
| McCulloch–Pitts | $h = \sum_i w_ix_i$; $o = 1$ if $h \ge \theta$ else 0 |
| Perceptron output | $y_j = 1$ if $\sum_{i=0}^m w_{ij}x_i > 0$ else 0, with $x_0 = -1$ (bias) |
| Perceptron rule | $w_{ij} \leftarrow w_{ij} - \eta(y_j - t_j)x_i$ ≡ $\Delta w_{ij} = \eta(t_j - y_j)x_i$ |
| Decision boundary | $w\cdot x = 0$ (w ⟂ boundary); distance $= |w^Tx' + b|/\|w\|$ |
| Convergence bound | $t \le 1/\gamma^2$ ($\|x\| \le 1$); $t \le R^2/\gamma^2$ ($\|x\| \le R$) |
| Sum-of-squares error | $E = \frac12\sum_k (t_k - y_k)^2$ |
| Gradient descent | $\Delta w = -\eta\,\partial E/\partial w$ |
| Delta rule (linear) | $\Delta w_{ik} = \eta(t_k - y_k)x_i$ |
| Sigmoid | $g(h) = 1/(1 + e^{-\beta h})$; $g' = \beta\,g(1 - g)$ |
| tanh | $\tanh(h)$, derivative $1 - \tanh^2 h$; $\tanh(h) = 2g(2h) - 1$ |
| Softmax | $y_k = e^{h_k}/\sum_j e^{h_j}$ |
| Output delta | $\delta_o(k) = (y_k - t_k)\,y_k(1 - y_k)$ [sigmoid]; $= (y_k - t_k)$ [linear or softmax + cross-entropy] |
| Hidden delta | $\delta_h(j) = a_j(1 - a_j)\sum_k w_{jk}\delta_o(k)$ |
| Updates | $w_{jk} \leftarrow w_{jk} - \eta\,\delta_o(k)\,a_j$; $\;v_{ij} \leftarrow v_{ij} - \eta\,\delta_h(j)\,x_i$ |
| Rumelhart form | $\Delta_p w_{ji} = \eta\,\delta_{pj}\,o_{pi}$; out: $\delta = (t - o)f'(net)$; hidden: $\delta = f'(net)\sum_k\delta_k w_{kj}$ |
| Momentum | $\Delta w^t = \eta\,\delta\,a + \alpha\,\Delta w^{t-1}$, α ≈ 0.9 |
| Weight init | $-1/\sqrt{n} < w < 1/\sqrt{n}$ |
| # weights | $(L+1)M + (M+1)N$; data ≥ 10 × #weights |
| Splits | 50:25:25 (plenty of data), 60:20:20 (less), else K-fold / leave-one-out |
| Time series | $x(t+\tau) = f(x(t), x(t-\tau), \dots, x(t-k\tau))$ |
| Metrics | Acc $= \frac{TP+TN}{\text{all}}$; Prec $= \frac{TP}{TP+FP}$; Rec $= \frac{TP}{TP+FN}$; $F_1 = \frac{2PR}{P+R}$ |
| Learning rate | typical 0.1 < η < 0.4 |

**Key numbers:** 10¹¹ neurons · 10¹⁴ synapses · ~10⁴ connections/neuron · 1.4 kg · ms vs ns · ~100 serial steps · 1943 M-P · 1958 perceptron · 1969 Minsky-Papert · 1986 backprop · 2 EB/day data · 80% data prep · resting −70 mV / threshold −55 mV / peak +40 mV.

---

# PART 6 — COMMON TRAPS (read before the exam)

1. **Sign conventions:** $\eta(t - y)x$ with "+" and $\eta(y - t)x$ with "−" are the **same** update. Pick one and be consistent. Same for backprop: book uses $(y - t)$ and subtracts; Rumelhart uses $(t - o)$ and adds.
2. **Bias input value:** the book/slides use **−1** (so the bias weight = threshold). Lab 2 Q1 and the Mazur example use an additive bias b (input +1). With input −1, a *positive* bias weight *raises* the threshold.
3. **"> 0" vs "≥ 0":** the book's code fires on **> 0**; the M-P slide says **≥ θ**. It matters for points exactly on the boundary (e.g., $D_{in} = 0$ in the XOR MLP does NOT fire). State your convention.
4. **In backprop, compute hidden deltas using the OLD output weights**, then update all weights.
5. **Don't forget the bias weight** when counting weights or when updating (its "input" is −1 or +1).
6. **Hidden deltas for bias nodes are dropped** (`deltah[:,:-1]`) — nothing feeds into a bias node.
7. **Regression → linear outputs**; using sigmoid outputs for values like 80–400 without scaling fails (sigmoid is limited to 0–1).
8. **Normalise before splitting** (same scaling for train and test); randomise the order before splitting.
9. **Never tune on the test set** — the validation set is for tuning/early stopping; the test set is used once.
10. **Accuracy on unbalanced data is misleading** → F1.
11. **Perceptron + non-separable data = cycling**, not slow convergence.
12. **Convergence bound with R** is $R^2/\gamma^2$, not $1/\gamma^2$.
13. **Early stopping point = minimum of the VALIDATION error**, not the training error.
14. **Sequential training: shuffle inputs each epoch** — except for recurrent/time-ordered training.

---

*Good luck — you've got this. If you can do the OR hand-trace (§2.14), the XOR proof (§2.17), the convergence proof (§2.19), the backprop numbers (§3.12), and explain every line of §3.8 and §3.18–3.20 without looking, you are ready for this part of the final.*
