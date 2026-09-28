# CS582 Machine Learning — Study Guide
## Lectures 1–3: Introduction · Perceptron · Multi-Layer Perceptron (MLP) + Labs 2 & 3

> Split from the ultimate study guide. See also the other parts in this `study/` folder.
>
> **Sources:** course slides, labs (incl. Lab 2 solutions), Final Practice, Marsland textbook Ch. 2–4.
> Every worked-example number was re-computed with code.

## How to use this part

1. Read all sections in order.
2. Each section ends with an **Exam angle** (in the original; check section ends).
3. After finishing Parts 1–3, use `04_QuestionBank_FormulaSheet_Traps.md` for practice + formula sheet + traps.

---

# PART 3 — LECTURE 3: MULTI-LAYER PERCEPTRON (MLP) & IMPORTANT NOTES

### 3.1 Why multi-layer networks?
- The perceptron can model **any linear problem** — but **most interesting practical problems are nonlinear** (e.g., XOR).
- **Multi-layer networks can represent arbitrary functions (linear or nonlinear).**
- But an effective learning algorithm was hard — took **30+ years** (backprop, 1986).
- Structure: **input layer, hidden layer(s), output layer**, each **fully connected** to the next, activation **feeding forward**.
- The **weights determine the function computed**. With enough hidden units, **any Boolean function can be computed with a single hidden layer.**
- Hidden layer name: we **can't see or correct** their values directly (no targets for them).
- **MLP is the most commonly used NN.**

### 3.2 MLP solving XOR (hand-verify — Book Problem 4.1)

Network: inputs A, B; hidden C, D; output E. Each neuron has a bias input −1 (weights shown). Fire if $h > 0$.
- **C:** weights A→1, B→1, bias weight 0.5 → $C_{in} = A + B - 0.5$ (acts like **OR**)
- **D:** weights A→1, B→1, bias weight 1 → $D_{in} = A + B - 1$ (acts like **AND**)
- **E:** weights C→1, D→−1, bias weight 0.5 → $E_{in} = C - D - 0.5$ (**"OR and not AND"**)

| A | B | $C_{in}$ | $C_{out}$ | $D_{in}$ | $D_{out}$ | $E_{in}$ | **E** |
|---|---|---|---|---|---|---|---|
| 0 | 0 | −0.5 | 0 | −1 | 0 | −0.5 | **0** |
| 0 | 1 | 0.5 | 1 | 0 | 0 | 0.5 | **1** |
| 1 | 0 | 0.5 | 1 | 0 | 0 | 0.5 | **1** |
| 1 | 1 | 1.5 | 1 | 1 | 1 | −0.5 | **0** |

→ E = XOR(A, B) ✓. (Note $D_{in} = 0$ does **not** fire because the rule is "> 0".)
**Intuition:** each hidden neuron draws **one line**; the output neuron combines them → the region **between two lines** = XOR.

### 3.3 Gradient descent
- MLP is harder than the perceptron: more weights; **which weights are wrong — input→hidden or hidden→output?**
- Use **gradient descent**: compute the gradient by **differentiation** and move downhill on the error surface:
$$\boxed{\Delta w_{ik} = -\eta\,\frac{\partial E}{\partial w_{ik}}}$$
- Picture: a ball rolling downhill on the error landscape until it settles in a **(local) minimum**.
- We differentiate w.r.t. the **weights** because inputs and the activation function are fixed during learning — only the weights can change.

### 3.4 Error function & the Delta Rule
- Perceptron used $(t - y)$ — but errors of opposite sign could cancel (sum = 0 even with errors).
- **Sum-of-squares error** (all positive; the ½ makes differentiation clean):
$$E(\vec{w}) = \frac{1}{2}\sum_k (t_k - y_k)^2 = \frac{1}{2}\sum_k\left(t_k - \sum_i w_{ik}x_i\right)^2$$
- Ignoring the threshold (linear units):
$$\frac{\partial E}{\partial w_{ik}} = \sum_k (t_k - y_k)(-x_i) \;\Rightarrow\; \Delta w_{ik} = \eta\,(t_k - y_k)\,x_i$$
- This is the **Delta Rule** (Widrow–Hoff / LMS): *a gradient-descent learning rule for updating the weights of the inputs to neurons in a **single-layer** network.* It looks just like the perceptron rule, but comes from gradient descent.

### 3.5 Training an MLP — Step 1: Forward pass
1. Put the input values in the input layer.
2. Calculate activations of the **hidden** nodes.
3. Calculate activations of the **output** nodes.
4. Calculate the errors using the targets.

### 3.6 The problem (why it's hard)
- For **output nodes** — we don't know the **inputs** (they are the hidden activations, which themselves depend on weights).
- For **hidden nodes** — we don't know the **targets**.
- For extra hidden layers — we know **neither**.
- ⇒ Hard to use gradient descent directly → **Backpropagation of errors**:
  - From output errors, update the **last layer** of weights.
  - From these errors, update the **next layer** back.
  - Work **backwards** through the network → the error is **back-propagated**.

### 3.7 Activation function — why we need the sigmoid
- In the analysis we ignored the activation function, but the **threshold (step) function is not differentiable** (discontinuous jump).
- **What we want in an activation function:**
  1. **Differentiable**
  2. **Should saturate** (become constant at the ends) — to act like fire / don't fire
  3. **Change between saturation values quickly**
- **Sigmoid (logistic):**
$$g(a) = \frac{1}{1 + e^{-\beta a}}$$
  S-shaped, ranges (0, 1), β > 0 controls steepness (large β → approaches step function; small, e.g. β ≤ 3, is more effective for learning).

**Derivative of the sigmoid (derive on exam!)** with β = 1, $y = \dfrac{1}{1+e^{-z}}$:
$$\frac{dy}{dz} = -(1+e^{-z})^{-2}\cdot(-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^2} = \frac{1}{1+e^{-z}}\cdot\frac{e^{-z}}{1+e^{-z}} = \frac{1}{1+e^{-z}}\left[1 - \frac{1}{1+e^{-z}}\right]$$
$$\boxed{\frac{dy}{dz} = y\,(1-y)}\qquad(\text{with }\beta:\; \beta\,y(1-y))$$
This is **why** the factors $y(1-y)$ and $a(1-a)$ appear in the backprop deltas ("due to sigmoid").

**Other activations (book §4.2.3):**
| Activation | Formula | Derivative / output delta | Use |
|---|---|---|---|
| Sigmoid | $1/(1+e^{-\beta h})$ | $\delta_o = (y-t)\,y(1-y)$ | hidden layers, classification outputs (0/1) |
| tanh | $\tanh(h)$, range (−1, 1) | $1 - \tanh^2(h)$ | hidden layers; $\tanh(h) = 2g(2h) - 1$ |
| **Linear** | $y = h$ | $\delta_o = (y - t)$ | **regression / time series** outputs |
| **Softmax** | $y_k = e^{h_k}/\sum_j e^{h_j}$ | with cross-entropy error: $\delta_o = (y - t)$ | multi-class, 1-of-N outputs (sum to 1) |

### 3.8 Error terms (deltas) and weight updates — THE CORE EQUATIONS

Notation (book): inputs $x_i$ ($x_0=-1$ bias), first-layer weights $v_{ij}$ (input→hidden), hidden activations $a_j$ ($a_0 = -1$ bias), second-layer weights $w_{jk}$ (hidden→output), outputs $y_k$, targets $t_k$.

**Output deltas** (eq 4.8):
$$\boxed{\delta_o(k) = (y_k - t_k)\,y_k\,(1 - y_k)}$$

**Hidden deltas** (eq 4.9):
$$\boxed{\delta_h(j) = a_j\,(1 - a_j)\sum_{k=1}^{N} w_{jk}\,\delta_o(k)}$$
- $a_j(1-a_j)$ = "**due to sigmoid**" (derivative of hidden activation)
- $\sum_k w_{jk}\delta_o(k)$ = "**total back-propagated error from the output layer**" — each output's error sent back, **scaled by the connecting weight**, and **summed**.

**Weight updates:**
$$\boxed{w_{jk} \leftarrow w_{jk} - \eta\,\delta_o(k)\,a_j^{\text{hidden}}}\quad(4.10)\qquad \boxed{v_{ij} \leftarrow v_{ij} - \eta\,\delta_h(j)\,x_i}\quad(4.11)$$

**Pattern to remember:** $\Delta(\text{weight}) = -\eta \times (\delta \text{ of the neuron the weight goes INTO}) \times (\text{activation coming OUT of the neuron the weight comes FROM})$.

### 3.9 The MLP Algorithm (write this from memory)

```
Initialisation:
    initialise all weights to small random values (positive and negative)
Training — repeat:
    for each input vector:
        FORWARD:
            h_j = Σ_i x_i v_ij ;   a_j = g(h_j) = 1/(1+exp(-β h_j))     (hidden)
            h_k = Σ_j a_j w_jk ;   y_k = g(h_k)                           (output)
        BACKWARD:
            δ_o(k) = (y_k − t_k) y_k (1 − y_k)                           (output error)
            δ_h(j) = a_j (1 − a_j) Σ_k w_jk δ_o(k)                        (hidden error)
            w_jk ← w_jk − η δ_o(k) a_j                                   (update output weights)
            v_ij ← v_ij − η δ_h(j) x_i                                   (update hidden weights)
    (if sequential) randomise the order of input vectors each epoch
until learning stops (early stopping)
Recall: use the forward phase only
```
Summary slides: *introduce inputs → feed forward → compute sum-of-squares error at outputs → compute output deltas by differentiation → update weights from outputs to last hidden layer → propagate errors back to hidden neurons → compute their deltas → update the next set of weights → repeat until reaching the inputs.*

### 3.10 Backpropagation — full derivation (Rumelhart, Hinton & Williams 1986)

Notation of the paper: p = input/output **pattern**, $t_{pj}$ = target, $o_{pj}$ = output of unit j, $i_{pi}$ = input i, $net_{pj}$ = weighted sum into unit j.

**Step 1 — prove the (standard) Delta Rule for linear units.**
Rule to prove: $\Delta_p w_{ji} = \eta\,(t_{pj} - o_{pj})\,i_{pi} = \eta\,\delta_{pj}\,i_{pi}$ (eq 1)

Error: $E_p = \frac{1}{2}\sum_j (t_{pj} - o_{pj})^2$ (eq 2). We want $-\dfrac{\partial E_p}{\partial w_{ji}} = \delta_{pj}\,i_{pi}$.

Chain rule: $\dfrac{\partial E_p}{\partial w_{ji}} = \dfrac{\partial E_p}{\partial o_{pj}}\cdot\dfrac{\partial o_{pj}}{\partial w_{ji}}$ (eq 3)
— 1st part: how the error changes with the unit's output; 2nd part: how changing $w_{ji}$ changes that output.

- From eq 2: $\dfrac{\partial E_p}{\partial o_{pj}} = -(t_{pj} - o_{pj}) = -\delta_{pj}$ (eq 4)
- Linear units: $o_{pj} = \sum_i w_{ji}\,i_{pi}$ (eq 5) ⇒ $\dfrac{\partial o_{pj}}{\partial w_{ji}} = i_{pi}$
- ⇒ $-\dfrac{\partial E_p}{\partial w_{ji}} = \delta_{pj}\,i_{pi}$ (eq 6) ✓ — gradient descent gives the delta rule.

**Step 2 — Generalized Delta Rule for semi-linear activation.**
$net_{pj} = \sum_i w_{ji}\,o_{pi}$ (eq 7) (with $o_i = i_i$ for input units), $o_{pj} = f_j(net_{pj})$ (eq 8), where **f is differentiable and non-decreasing** (= "semi-linear", e.g. sigmoid).

Chain rule: $\dfrac{\partial E_p}{\partial w_{ji}} = \dfrac{\partial E_p}{\partial net_{pj}}\cdot\dfrac{\partial net_{pj}}{\partial w_{ji}}$ (eq 9)
By eq 7: $\dfrac{\partial net_{pj}}{\partial w_{ji}} = \dfrac{\partial}{\partial w_{ji}}\sum_k w_{jk}o_{pk} = o_{pi}$ (eq 10)

**Define** $\delta_{pj} = -\dfrac{\partial E_p}{\partial net_{pj}}$ ⇒ $-\dfrac{\partial E_p}{\partial w_{ji}} = \delta_{pj}\,o_{pi}$ ⇒
$$\boxed{\Delta_p w_{ji} = \eta\,\delta_{pj}\,o_{pi}}\quad(11)$$
Same form as the standard delta rule — only δ is different.

**Step 3 — compute δ (chain rule again):**
$$\delta_{pj} = -\frac{\partial E_p}{\partial net_{pj}} = -\frac{\partial E_p}{\partial o_{pj}}\cdot\frac{\partial o_{pj}}{\partial net_{pj}},\qquad \frac{\partial o_{pj}}{\partial net_{pj}} = f'_j(net_{pj})\quad(12)$$
The first factor $\partial E_p/\partial o_{pj}$ is the one "whose style changes from layer to layer."

**Case 1 — output unit:** $\partial E_p/\partial o_{pj} = -(t_{pj} - o_{pj})$ ⇒
$$\boxed{\delta_{pj} = (t_{pj} - o_{pj})\,f'_j(net_{pj})}\quad(13)$$

**Case 2 — hidden unit** (no target — go *backwards* and add all contributions from the layer above):
$$\sum_k \frac{\partial E_p}{\partial net_{pk}}\frac{\partial net_{pk}}{\partial o_{pj}} = \sum_k\frac{\partial E_p}{\partial net_{pk}}\frac{\partial}{\partial o_{pj}}\sum_i w_{ki}o_{pi} = \sum_k\frac{\partial E_p}{\partial net_{pk}}w_{kj} = -\sum_k\delta_{pk}w_{kj}$$
$$\boxed{\delta_{pj} = f'_j(net_{pj})\sum_k \delta_{pk}\,w_{kj}}\quad(14)$$

**With the sigmoid** $f' = o(1-o)$: output $\delta = (t - o)\,o(1-o)$; hidden $\delta = o(1-o)\sum_k \delta_k w_{kj}$ — exactly eqs 4.8/4.9 (the book just uses $(y - t)$ and a **minus** in the update, the paper uses $(t - o)$ and a **plus**: same result).

### 3.11 Python implementation (book `mlp.py`, batch)

**Forward** ("not much different from the Perceptron, except do it twice"):
```python
inputs  = np.concatenate((inputs, -np.ones((self.ndata,1))), axis=1)
hidden  = np.dot(inputs, weights1)
hidden  = 1.0/(1.0 + np.exp(-beta*hidden))
hidden  = np.concatenate((hidden, -np.ones((self.ndata,1))), axis=1)
outputs = np.dot(hidden, weights2)
return 1.0/(1.0 + np.exp(-beta*outputs))
```
**Backward** (book's batch version, uses $t - y$ so it **adds**):
```python
deltao = (targets - self.outputs)*self.outputs*(1.0 - self.outputs)
deltah = self.hidden*(1.0 - self.hidden)*(np.dot(deltao, np.transpose(self.weights2)))
updatew1 = eta*(np.dot(np.transpose(inputs), deltah[:,:-1]))   # drop bias column of deltah
updatew2 = eta*(np.dot(np.transpose(self.hidden), deltao))
self.weights1 += updatew1
self.weights2 += updatew2
```
Loop form of the hidden delta (slide):
```python
deltah = np.zeros(nhidden+1)
for j in range(nhidden+1):
    sumk = np.sum(weights2[j,:]*deltao[d,:])
    deltah[j] = hidden[d,j]*(1.0 - hidden[d,j])*sumk
```
Book results: AND learned in ~1000 iterations; XOR needs ~5000 (sometimes more) → **MLP costs much more computation than the perceptron, even for linear problems.**

### 3.12 Worked backprop example (Matt Mazur — on the slides) — PRACTISE THIS

**Network:** 2 inputs, 2 hidden, 2 outputs, all sigmoid (β = 1), η = 0.5, biases are +1 inputs with weights b1, b2.
- Inputs: $i_1 = 0.05,\; i_2 = 0.10$; Targets: $t_1 = 0.01,\; t_2 = 0.99$
- $w_1 = .15$ (i1→h1), $w_2 = .20$ (i2→h1), $w_3 = .25$ (i1→h2), $w_4 = .30$ (i2→h2), $b_1 = .35$
- $w_5 = .40$ (h1→o1), $w_6 = .45$ (h2→o1), $w_7 = .50$ (h1→o2), $w_8 = .55$ (h2→o2), $b_2 = .60$
- Error: $E = \sum \frac12 (t - o)^2$

**Forward pass**
- $net_{h1} = 0.15(0.05) + 0.20(0.10) + 0.35 = 0.3775$ → $out_{h1} = \sigma(0.3775) = \mathbf{0.593270}$
- $net_{h2} = 0.25(0.05) + 0.30(0.10) + 0.35 = 0.3925$ → $out_{h2} = \mathbf{0.596884}$
- $net_{o1} = 0.40(0.593270) + 0.45(0.596884) + 0.60 = 1.105906$ → $out_{o1} = \mathbf{0.751365}$
- $net_{o2} = 0.50(0.593270) + 0.55(0.596884) + 0.60 = 1.224921$ → $out_{o2} = \mathbf{0.772928}$
- $E_{o1} = \frac12(0.01 - 0.751365)^2 = 0.274811$; $E_{o2} = \frac12(0.99 - 0.772928)^2 = 0.023560$; $E_{total} = \mathbf{0.298371}$

**Backward pass — output layer**
- $\delta_{o1} = (out_{o1} - t_1)\,out_{o1}(1 - out_{o1}) = (0.741365)(0.751365)(0.248635) = \mathbf{0.138499}$
- $\delta_{o2} = (0.772928 - 0.99)(0.772928)(0.227072) = \mathbf{-0.038098}$
- $\partial E/\partial w_5 = \delta_{o1}\cdot out_{h1} = 0.138499 \times 0.593270 = 0.082167$
- $w_5^+ = 0.40 - 0.5(0.082167) = \mathbf{0.358916}$
- $w_6^+ = 0.45 - 0.5(0.138499)(0.596884) = \mathbf{0.408666}$
- $w_7^+ = 0.50 - 0.5(-0.038098)(0.593270) = \mathbf{0.511301}$
- $w_8^+ = 0.55 - 0.5(-0.038098)(0.596884) = \mathbf{0.561370}$

**Backward pass — hidden layer** (use the **OLD** $w_5..w_8$!)
- Back-propagated error to h1: $w_5\delta_{o1} + w_7\delta_{o2} = 0.40(0.138499) + 0.50(-0.038098) = 0.036350$
- $\delta_{h1} = out_{h1}(1 - out_{h1}) \times 0.036350 = 0.593270 \times 0.406730 \times 0.036350 = \mathbf{0.008771}$
- $\delta_{h2} = out_{h2}(1 - out_{h2})(w_6\delta_{o1} + w_8\delta_{o2}) = \mathbf{0.009954}$
- $w_1^+ = 0.15 - 0.5(0.008771)(0.05) = \mathbf{0.149781}$
- $w_2^+ = 0.20 - 0.5(0.008771)(0.10) = \mathbf{0.199561}$
- $w_3^+ = 0.25 - 0.5(0.009954)(0.05) = \mathbf{0.249751}$
- $w_4^+ = 0.30 - 0.5(0.009954)(0.10) = \mathbf{0.299502}$

After this one update, the total error drops (≈0.2910); after 10,000 iterations it is ~0.0000351 (outputs ≈ 0.0159 and 0.9841).
**Trap:** compute ALL deltas using the **old** weights, then update.

### 3.13 Initialising the weights
- **Small random values, positive and negative.**
- **Too large** (close to ±1 and beyond): inputs to the sigmoid are large → neuron **saturates** at 0 or 1 → **gradients ≈ 0 → learning very slow**.
- **Too small** (≈ 0): neuron works in the sigmoid's **linear region** → network behaves like a **linear model**.
- **All zero / identical:** every hidden neuron gets the same updates → they stay identical (symmetry is never broken) — this is why we use **random** values.
- Common trick: $-\dfrac{1}{\sqrt{n}} < w < \dfrac{1}{\sqrt{n}}$, n = number of nodes feeding into that layer → total input to a neuron ≈ size 1.
- Keep all weights about the same size so they reach final values at about the same time → **uniform learning**.
- Random values → learning starts from different places each run → train several networks and pick the best.

### 3.14 Batch vs Sequential vs Minibatch vs SGD (important note on the slides)
**When should weights be updated?**
- **After all inputs are seen (Batch):** more accurate **estimate of the gradient**; **converges to a local minimum faster**.
- **After each input (Sequential/online):** sometimes **simpler to program**; **may escape local minima** (change the order of presentation — shuffle each epoch).
- **Both ways need many epochs** (passes through the whole dataset).
- **Minibatch:** split the training set into random batches, update after each batch → middle ground.
- **Stochastic Gradient Descent (SGD):** one randomly chosen input per update — good for huge datasets.

### 3.15 Local minima, momentum, weight decay
- Gradient descent only guarantees a **local minimum**. Which one you end up in depends on the **starting point**.
- Fixes: **train several networks from different random starting points**; **momentum**.
- **Momentum** — add part of the previous weight change (ball with mass keeps rolling over small bumps):
$$w^{t} \leftarrow w^{t-1} + \eta\,\delta\,a + \alpha\,\Delta w^{t-1},\qquad 0 < \alpha < 1,\; \text{typically } \alpha = 0.9$$
  Benefits: helps **escape local minima**, **more stable** dynamics, **allows a smaller learning rate**, faster learning.
- **Weight decay:** after each epoch multiply every weight by $0 < \epsilon < 1$ → smaller weights → network closer to linear; only essential weights stay large. Can sometimes make things worse — tune experimentally.
- Other: **reduce the learning rate as training progresses**; use **second-derivative** information.

### 3.16 Network topology & learning capacity
- **How many layers? How many neurons per layer?** → **"No good answers."** Rules of thumb: **at most 3 layers, usually 2** (i.e., 1–2 hidden layers); guess sizes (layers usually **get smaller** toward the output); **test several different networks**.
- **Universal Approximation Theorem:** **one hidden layer** with enough hidden nodes can approximate any continuous function. Two hidden layers are *sufficient* (not necessary) for any decision boundary.
- **How the MLP learns (Fig 4.9/4.10):** (a) one sigmoid neuron = a **step/ridge** → (b) add a reversed sigmoid → a **hill/ridge** → (c) add another hill at 90° → a **bump** → (d) sharpen the bump; the output layer **adds bumps together** → the MLP learns a **local representation** and can approximate anything.

**Decision regions by structure (classic table on the slides):**
| Structure | Decision regions | XOR? |
|---|---|---|
| **Single layer** (perceptron) | **Half-plane** bounded by a hyperplane | ✗ |
| **Two layer** (1 hidden) | **Convex** open or closed regions | ✓ |
| **Three layer** (2 hidden) | **Arbitrary** (complexity limited by number of nodes) | ✓ |

### 3.17 Amount of training — how much data, how many epochs?
- Number of weights for one hidden layer: $\boxed{(L+1)\times M + (M+1)\times N}$ (L inputs, M hidden, N outputs; +1 = bias).
  Example: 8 inputs, 5 hidden, 1 output → $9\times5 + 6\times1 = 51$ weights → need ≈ **510** training examples.
- **Rule of thumb: use 10 times more data than the number of weights.**
- Epochs: no fixed number → use a **validation set + early stopping**.

### 3.18 Training, Validation, Testing
- **Training set:** used to train; its error is minimised during training.
- **Validation set:** tracks **how well the NN is doing while it learns** (checks whether learning works, detects overfitting, used for early stopping and choosing the architecture).
- **Test set:** used **once, at the end**, to check the **overall performance** on unseen data.
- Split: **50:25:25** if plenty of data; otherwise **60:20:20** (Fig 2.6). The exact ratio is up to you and the application.
- **Randomise** before splitting (e.g., the Iris file is sorted by class) — otherwise one set may contain mostly one class.
- **Limited data → leave-some-out, multi-fold cross-validation** (Fig 2.7): split the data into K subsets; train a model on most sets, hold one out for validation (and another for testing); train **different models with different sets held out**; pick the model with the **lowest validation error**. Extreme case: **leave-one-out**.

### 3.19 Generalization, Early Stopping, Overfitting, Undertraining
- **Generalization** = a major advantage of NNs: a trained net can correctly classify data **it has never seen** from the same classes.
- Typical curves: **training error keeps decreasing**; **validation error decreases, then starts increasing**.
- **Early stopping:** stop at the **minimum of the validation error** — there the net generalizes best ("Time to stop training").
- **Overfitting (over-training):** the network models the training data too closely, **including the noise** → it has **memorised** the training examples → **higher error on new data** → poor generalization (right figure: wiggly curve through every point). Left figure: smooth curve following the overall trend = good generalization.
- **Undertraining:** no overfitting, but the model hasn't learned the underlying function → also **more errors on unseen data**.
- ⇒ **Neither over- nor under-training is good.**
- Too many hidden nodes also overfit (book table: validation error lowest for 2–10 hidden nodes, rising for 25–50).
- Book early-stopping code idea: train ~100 iterations, compute validation error, continue while it is still decreasing (tracks the last two changes to avoid stopping on small fluctuations).

### 3.20 Evaluating results — Confusion Matrix, Accuracy, Precision, Recall, F1

**Confusion matrix:** all classes on both axes; one axis = **targets (true class)**, the other = **predicted outputs** (the slide puts predicted outputs on top). **Diagonal = correct.**

Slide example (rows = true class, columns = outputs):
| | $C_1$ | $C_2$ | $C_3$ |
|---|---|---|---|
| $C_1$ | **5** | 1 | 0 |
| $C_2$ | 1 | **4** | 1 |
| $C_3$ | 2 | 0 | **4** |

Accuracy = diagonal / total = $(5+4+4)/18 = 13/18 \approx$ **72.2%**.

**Accuracy matrix (binary):**
| | Predicted positive | Predicted negative |
|---|---|---|
| **Actually positive** | True Positive (TP) | False Negative (FN) |
| **Actually negative** | False Positive (FP) | True Negative (TN) |

$$\text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN}\qquad \text{Precision} = \frac{TP}{TP + FP}\qquad \text{Recall (Sensitivity)} = \frac{TP}{TP + FN}$$
$$F_1 = 2\,\frac{\text{precision}\times\text{recall}}{\text{precision} + \text{recall}}\qquad (\text{Specificity} = \frac{TN}{TN+FP})$$

- **Precision:** of everything predicted positive, how many really are? **Recall:** of all real positives, how many did we find?
- **F1 is key for highly unbalanced datasets** — accuracy is misleading there. (Book: if 90% of data is class 1, a classifier that always says "class 1" gets 90% accuracy but is useless → balance classes in training or use **novelty detection**.)

**Worked example:** TP = 40, FP = 10, FN = 20, TN = 30 →
Accuracy = 70/100 = **0.70**; Precision = 40/50 = **0.80**; Recall = 40/60 = **0.667**; F1 = 2(0.8)(0.667)/(1.467) = **0.727**; Specificity = 30/40 = 0.75.

In code (`confmat`): 1 output → threshold at 0.5 (or 0); several outputs (1-of-N) → **argmax**; accuracy = trace(cm)/sum(cm).

### 3.21 Four kinds of problems an MLP solves (book §4.4)
1. **Regression** — continuous output → **linear output nodes**, sum-of-squares error. (Book: noisy sine data; normalise; split 50:25:25; 3 hidden nodes, η = 0.25; early stopping.)
2. **Classification** — **1-of-N encoding**: one output node per class, target e.g. (0,0,0,1,0,0) = class 4 of 6; pick the **largest output (hard-max)** or use **soft-max**. (Alternative of thresholding one linear output into ranges is impractical.) Book Iris example: 5 hidden, softmax, early stopping → ~92% on test.
3. **Time-series prediction** — see Lab 3 Q1.
4. **Data compression / denoising** — **auto-associative network (autoencoder).**

### 3.22 The Auto-Associative Network (Autoencoder)
- **Output = Input** (train the network to reproduce its input).
- The hidden layer has **fewer neurons** than the input ("**bottleneck**") → it is **forced to compress** the data and learn the important **features** of the input (ignoring noise).
- **Compression:** image → cut into strips → 1-D input vector → train → store only the **hidden activations** (compressed code) + **second-layer weights**; feed hidden values forward to rebuild the image.
- **Denoising:** feed a noisy image → the network outputs the closest clean image it learned.
- With **linear** hidden nodes it learns the **Principal Components (PCA)**.
- **Will be used later in Deep Learning** (stacked autoencoders / pre-training — relevant for the final's DL questions).

### 3.23 Other successful MLP applications
Text-to-speech (**NetTalk**), **fraud detection**, financial applications (**HNC**, bought by Fair Isaac), chemical plant control (**Pavilion Technologies**), automated vehicles, game playing (**Neurogammon**), **handwriting recognition**; in general classification, prediction, logic, fraud detection…

### 3.24 The MLP recipe (book §4.5 — great for "design" questions)
1. **Select inputs and outputs** (features, output encoding: sigmoid vs. linear vs. softmax).
2. **Normalise inputs** (subtract mean, divide by variance — or by max/−min).
3. **Split data** into training / validation / test (≈50:25:25; cross-validation if little data).
4. **Select a network architecture** (inputs & outputs are fixed by the data; choose the number of hidden layers/nodes — try several).
5. **Train** with backprop + **early stopping** on the validation set.
6. **Test** once on the test set.

### 3.25 Large gap between ANN and biological learning
Key differences: **network architecture**, **feedback paths**, **learning rules**, **bias & threshold function**, **semantics**, and more. Yet the NN is still a **great learning system** — the most used in ML (especially Deep Learning), and improvements will keep making it better.

---

### 3.26 Lecture 3 Extension — Design Example: classifying aeroplanes

**Data** (mass, speed → class):
| Mass | Speed | Class |
|---|---|---|
| 1.0 | 0.1 | Bomber |
| 2.0 | 0.2 | Bomber |
| 0.1 | 0.3 | Fighter |
| 2.0 | 0.3 | Bomber |
| 0.2 | 0.4 | Fighter |
| 3.0 | 0.4 | Bomber |
| 0.1 | 0.5 | Fighter |
| 1.5 | 0.5 | Bomber |
| 0.5 | 0.6 | Fighter |
| 1.6 | 0.7 | Fighter |

**General procedure for building neural networks (8 steps — memorise):**
1. **Understand and specify the problem** in terms of inputs and required outputs (for classification, outputs = classes, usually binary vectors).
2. Take the **simplest form of network** that might solve it (e.g., a simple Perceptron).
3. Find appropriate **connection weights** (including thresholds) so the network gives the right outputs on the training data.
4. Make sure it works on the **training data**, and **test generalization** on new test data.
5. If not good enough → go back to **stage 3** and try harder.
6. Still not good enough → back to **stage 2** (a more complex network).
7. Still not good enough → back to **stage 1** (re-specify the problem/inputs).
8. **Problem solved** → move on.

**Design:** inputs = direct encodings of mass and speed; with 2 classes use **one output unit** (1 = fighter, 0 = bomber). Simplest network: a **Perceptron**:
$$\text{class} = \text{sgn}(w_0 + w_1\cdot\text{Mass} + w_2\cdot\text{Speed})$$

**Is it linearly separable? Yes.** One separating line: **Fighter if $\text{Speed} - 0.2\,\text{Mass} - 0.24 > 0$** (all 10 points classified correctly — hardest points: (1.5, 0.5) bomber gives −0.04, (0.1, 0.3) fighter gives +0.04).
Training a perceptron (sequential, η = 0.1, bias input −1, starting from 0) converges in 6 epochs to $w_{bias} = -0.1,\; w_{mass} = -0.18,\; w_{speed} = 0.33$, i.e. fighter iff $0.1 - 0.18\,M + 0.33\,S > 0$.
Intuition: **fighters are light and fast; bombers are heavy and slow.**

**Training / validation:** determine training, validation and test sets; train with the perceptron code; check validation (**early stopping**); evaluate with the **confusion matrix** and **accuracy matrix**; repeat steps as needed.

---

### 3.27 ✅ LAB 3 SOLUTIONS

#### Lab 3 Q1 — Study §4.4.4, run the MLP on `PNOz.dat` (Palmerston North ozone), reproduce Fig 4.16

**Data:** daily ozone-layer thickness above Palmerston North, NZ, 1996–2004; **2855 readings**; 4 columns: year, day of year, **ozone level (col 2)**, sulphur dioxide level. Ozone varies **seasonally** over the year.

**Time-series formulation:**
$$y = x(t + \tau) = f\big(x(t),\, x(t-\tau),\, \dots,\, x(t - k\tau)\big)$$
- **k** = how many past points are used as inputs; **τ** = the spacing between them.
- Example τ = 2, k = 3: inputs = elements 1, 3, 5 → target = element 7; next: 2, 4, 6 → 8; then 3, 5, 7 → 9.

```python
import numpy as np, pylab as pl, mlp
PNoz = np.loadtxt('PNOz.dat')
pl.plot(np.arange(np.shape(PNoz)[0]), PNoz[:,2], '.')           # look at the data

# normalise the ozone column
PNoz[:,2] = PNoz[:,2] - PNoz[:,2].mean()
PNoz[:,2] = PNoz[:,2] / PNoz[:,2].max()

# build input vectors (k values, spacing t) and targets
t, k = 2, 3
lastPoint = np.shape(PNoz)[0] - t*k
inputs  = np.zeros((lastPoint, k))
targets = np.zeros((lastPoint, 1))
for i in range(lastPoint):
    inputs[i,:] = PNoz[i:i+t*k:t, 2]
    targets[i]  = PNoz[i+t*k, 2]

# last 400 points = test; the rest alternate train / validation
test  = inputs[-400:,:];      testtargets  = targets[-400:]
train = inputs[:-400:2,:];    traintargets = targets[:-400:2]
valid = inputs[1:-400:2,:];   validtargets = targets[1:-400:2]

net = mlp.mlp(train, traintargets, 3, outtype='linear')   # REGRESSION → linear outputs
net.earlystopping(train, traintargets, valid, validtargets, 0.25)

testin = np.concatenate((test, -np.ones((np.shape(test)[0],1))), axis=1)
out = net.mlpfwd(testin)
print('Test error:', 0.5*np.sum((targets[-400:] - out)**2))
pl.figure(); pl.plot(np.arange(400), out, '.'); pl.plot(np.arange(400), testtargets, 'x')
pl.legend(('Predictions','Targets')); pl.show()
```
**Expected result / what to write:** the predictions track the seasonal ozone curve closely (like Fig 4.16: 400 predicted vs actual values, k = 3, τ = 2). It's a **regression** problem → **linear output nodes**, **sum-of-squares error**, no confusion matrix. Beyond the number of hidden nodes you must also **experiment with τ and k**. Be careful not to select train/val/test **systematically** (e.g., a pattern only on odd days would be missed); randomise, or use the **end of the series** as the test set (which also mimics predicting the future).

#### Lab 3 Q2 — Predict electricity demand for the next 5 days (5 years of daily data, demand 80–400)

**(a) How to use an MLP, parameters, sensible values**
- Treat it as **time-series regression** exactly like the ozone problem.
- **Preprocess:** scale demand (80–400) to roughly [0, 1] or [−1, 1] (e.g., $(d - 80)/320$ or subtract mean/divide by max); undo the scaling on the outputs.
- **Inputs:** the last **k** days of demand with spacing **τ**. Sensible: **τ = 1** (daily) and **k = 7** (one full week, to capture the weekly cycle), or k = 14. Optionally add **lag-365** (same day last year) for the seasonal effect, and **day-of-week** (7 inputs, 1-of-N) and **month/season** inputs.
- **Outputs:** either **5 linear output nodes** (days t+1 … t+5 directly) or **1 output** predicting the next day, fed back in recursively to get 5 days (errors accumulate).
- **Output activation: linear** (regression), error = sum of squares.
- **Hidden layer:** 1 hidden layer; try e.g. **5–20** hidden nodes and pick the best on the validation set (train each size several times because of random initialisation).
- **Data:** 5 years ≈ **1825 days** → ≈1818 input/target pairs. Check the **10× weights rule**: with 7 inputs, 10 hidden, 5 outputs → $(7+1)\cdot10 + (10+1)\cdot5 = 135$ weights → ~1350 examples — OK.
- **Split:** e.g. first ~3.5 years train, next ~0.75 validation, last ~0.75 test (or 50:25:25 randomised); **early stopping** on the validation error.
- **Learning:** η ≈ 0.1–0.25, **momentum α ≈ 0.9**, small random initial weights (±1/√n).

**(b) Adding the weather forecast (day and night temperatures)**
- Add **2 more input nodes** (forecast day temperature and night temperature for the day being predicted), **normalised** like the other inputs. For training use the **actual recorded temperatures** of those days. If forecasts exist for each of the 5 days, add 2 inputs per day (10 inputs). Possibly add derived inputs like "heating/cooling degrees" (distance from a comfortable temperature), because demand rises both when very hot (AC) and very cold (heating) — the MLP's hidden layer can learn this non-linearity.

**(c) Will it work well? What can't it predict?**
- It should work **reasonably well for regular patterns**: weekly cycles, seasons, temperature effects (demand is strongly patterned).
- It **cannot predict** things **not represented in the inputs or the training data**:
  - **Holidays and special events** (Christmas, big sports events) unless a holiday input is added;
  - **Extreme / unusual weather** outside the range seen in 5 years (the network **extrapolates poorly**);
  - **Sudden changes**: power outages, a new factory opening/closing, economic changes, new technologies (e.g., electric cars), price changes;
  - **Long-term trends** (population growth) if they go beyond the training range.
- Prediction accuracy also **decreases for days further ahead** (day 5 is harder than day 1), especially with recursive prediction.

#### Lab 3 Q3 — Add another hidden layer (derive the gradients) and test on Pima

**Network:** inputs $x_i$ → weights $u_{ij}$ → hidden layer 1 ($a^{(1)}_j$) → weights $v_{jl}$ → hidden layer 2 ($a^{(2)}_l$) → weights $w_{lk}$ → outputs $y_k$. All hidden sigmoid.

**Derivation (same chain-rule idea, one extra step backwards):**
- Output: $\delta_o(k) = (y_k - t_k)\,y_k(1 - y_k)$ (sigmoid output) or $(y_k - t_k)$ (linear output)
- Hidden layer 2: $\delta_{h2}(l) = a^{(2)}_l(1 - a^{(2)}_l)\displaystyle\sum_k w_{lk}\,\delta_o(k)$
- Hidden layer 1 (**new**): $\delta_{h1}(j) = a^{(1)}_j(1 - a^{(1)}_j)\displaystyle\sum_l v_{jl}\,\delta_{h2}(l)$
- Updates: $w_{lk} \leftarrow w_{lk} - \eta\,\delta_o(k)\,a^{(2)}_l$; $\;v_{jl} \leftarrow v_{jl} - \eta\,\delta_{h2}(l)\,a^{(1)}_j$; $\;u_{ij} \leftarrow u_{ij} - \eta\,\delta_{h1}(j)\,x_i$

**Why:** $\dfrac{\partial E}{\partial u_{ij}} = \dfrac{\partial E}{\partial h^{(1)}_j}\,x_i$ and $\dfrac{\partial E}{\partial h^{(1)}_j} = g'(h^{(1)}_j)\sum_l \dfrac{\partial E}{\partial h^{(2)}_l}\,v_{jl}$ — each layer's delta = (its sigmoid derivative) × (weighted sum of the deltas of the layer above). **Generalises to any number of layers.**

**Code (batch, with momentum — tested: learns XOR, outputs ≈ 0.007, 0.994, 0.994, 0.006):**
```python
import numpy as np

class mlp2:
    def __init__(self, inputs, targets, nh1, nh2, beta=1.0, momentum=0.9, outtype='logistic'):
        self.nin, self.nout, self.ndata = inputs.shape[1], targets.shape[1], inputs.shape[0]
        self.beta, self.momentum, self.outtype = beta, momentum, outtype
        self.w1 = (np.random.rand(self.nin+1, nh1) - 0.5) * 2/np.sqrt(self.nin)
        self.w2 = (np.random.rand(nh1+1, nh2)      - 0.5) * 2/np.sqrt(nh1)
        self.w3 = (np.random.rand(nh2+1, self.nout) - 0.5) * 2/np.sqrt(nh2)

    def _addbias(self, a):
        return np.concatenate((a, -np.ones((a.shape[0], 1))), axis=1)

    def fwd(self, inputs):                       # inputs already include the bias column
        self.h1 = self._addbias(1.0/(1.0 + np.exp(-self.beta*np.dot(inputs,  self.w1))))
        self.h2 = self._addbias(1.0/(1.0 + np.exp(-self.beta*np.dot(self.h1, self.w2))))
        out = np.dot(self.h2, self.w3)
        return out if self.outtype == 'linear' else 1.0/(1.0 + np.exp(-self.beta*out))

    def train(self, inputs, targets, eta, niterations):
        inputs = self._addbias(inputs)
        u1, u2, u3 = np.zeros_like(self.w1), np.zeros_like(self.w2), np.zeros_like(self.w3)
        for n in range(niterations):
            y = self.fwd(inputs)
            if self.outtype == 'linear':
                d3 = (y - targets)/self.ndata
            else:
                d3 = self.beta*(y - targets)*y*(1.0 - y)
            d2 = self.beta*self.h2*(1.0 - self.h2)*np.dot(d3, self.w3.T)
            d1 = self.beta*self.h1*(1.0 - self.h1)*np.dot(d2[:, :-1], self.w2.T)   # drop bias delta
            u3 = eta*np.dot(self.h2.T, d3)          + self.momentum*u3
            u2 = eta*np.dot(self.h1.T, d2[:, :-1])  + self.momentum*u2
            u1 = eta*np.dot(inputs.T,  d1[:, :-1])  + self.momentum*u1
            self.w3 -= u3; self.w2 -= u2; self.w1 -= u1
        return 0.5*np.sum((y - targets)**2)
```
**Testing on Pima:** normalise the 8 inputs, split train/valid/test, e.g. `net = mlp2(train, traint, 8, 4)`, train with early stopping, evaluate with a confusion matrix. **Expected observation:** accuracy is usually better and more stable than the perceptron's 50–70% (once inputs are normalised), but the **second hidden layer gives little or no gain over one hidden layer** (Universal Approximation Theorem: one hidden layer is enough), while it adds weights (needs more data, trains slower, may overfit).

#### Lab 3 Q4 — Recurrent network (outputs at time t fed back as inputs at t+1), test on ozone data

**Idea:** add the previous output $y(t-1)$ as an **extra input** (a **Jordan network**; feeding back the hidden layer instead = **Elman network**). The network then has memory of its own past predictions → another way to handle time series (fewer explicit lagged inputs needed). Must be trained **sequentially in time order** (no shuffling), because each step needs the previous output.

```python
class rnn_mlp:
    """One hidden layer; the previous output y(t-1) is fed back as an extra input."""
    def __init__(self, nin, nhidden, beta=1.0):
        self.nin, self.nh, self.beta = nin + 1, nhidden, beta          # +1 = feedback input
        self.v = (np.random.rand(self.nin + 1, nhidden) - 0.5) * 2/np.sqrt(self.nin)
        self.w = (np.random.rand(nhidden + 1, 1)        - 0.5) * 2/np.sqrt(nhidden)

    def step(self, x, yprev):
        xin = np.concatenate((x, [yprev, -1.0]))                       # inputs + feedback + bias
        a = 1.0/(1.0 + np.exp(-self.beta*xin @ self.v))
        ab = np.concatenate((a, [-1.0]))
        return xin, ab, float((ab @ self.w)[0])                        # linear output

    def train(self, inputs, targets, eta, epochs):
        for e in range(epochs):
            yprev = 0.0
            for x, t in zip(inputs, targets):                          # in TIME ORDER
                xin, ab, y = self.step(x, yprev)
                do = y - t                                             # linear output delta
                dh = self.beta*ab[:-1]*(1 - ab[:-1])*(self.w[:-1, 0]*do)
                self.w -= eta*np.outer(ab, do)
                self.v -= eta*np.outer(xin, dh)
                yprev = y                                              # feed output back

    def predict(self, inputs):
        yprev, out = 0.0, []
        for x in inputs:
            _, _, yprev = self.step(x, yprev)
            out.append(yprev)
        return np.array(out)

# usage with the ozone inputs/targets built in Q1 (k = 3, tau = 2):
# r = rnn_mlp(3, 5); r.train(train_in, train_t, 0.05, 30); pred = r.predict(test_in)
```
(This simple version treats the fed-back value as a constant input when computing gradients — "truncated" backprop; full training would use **backpropagation through time**.) **Observation to write:** it follows the seasonal pattern similarly to the plain MLP; feedback lets the network use its own recent prediction (smoother output), but training is sequential, slower, and can be less stable (errors can feed back and accumulate).

> **Exam angle (Lecture 3):** why multilayer; XOR MLP table; gradient descent equation; why sigmoid (3 properties) + derivative; the 4 backprop equations and what each term means; derivation outline (chain rule, δ definition, output vs hidden case); weight init; batch vs sequential; local minima & momentum; how many layers/nodes; 10× rule and weight count; train/val/test and cross-validation; early stopping / overfitting / undertraining; confusion matrix, precision/recall/F1; autoencoder; the design procedure; MLP design for a time series.

---
