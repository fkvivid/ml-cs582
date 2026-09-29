# CS582 — Lessons 4 & 5 Companion
## Question Bank · Combined Formula Sheet · Traps

> Self-contained drill sheet for Lessons 4 and 5. Answer out loud with notes closed.

---

# PART A — Question bank with short answers

## Lesson 4 — Dimensionality reduction & VC

1. **Why reduce dimensions?**  
   Fewer weights/neurons, less data needed, less compute, better visualization, noise reduction, easier interpretation.

2. **What is the curse of dimensionality?**  
   As dimension grows, unit hypersphere volume eventually shrinks → space is sparse → need far more samples to generalize.

3. **Why does hypersphere volume fall after dim ≈ 5?**  
   Volume concentrates in hypercube corners outside the ball; \(v_n=(2\pi/n)v_{n-2}\) shrinks for \(n>2\pi\). Peaks at n=5 (~5.26), ~0 by n≈20.

4. **Three ways to do DR?**  
   Feature selection; feature derivation/extraction (transforms); clustering.

5. **LDA: supervised or unsupervised? Goal?**  
   Supervised. Maximize \(S_B/S_W\) (between / within class scatter) via projection \(z=w^Tx\).

6. **Write \(S_W\) and \(S_B\).**  
   \(S_W=\sum_c\sum_{j\in c}p_c(x_j-\mu_c)(x_j-\mu_c)^T\);  
   \(S_B=\sum_c(\mu_c-\mu)(\mu_c-\mu)^T\).

7. **PCA: supervised or unsupervised? Goal?**  
   Unsupervised. Orthogonal axes of max variance; drop low-variance axes; cov becomes diagonal.

8. **PCA algorithm steps?**  
   Center → covariance → eigenvectors/eigenvalues → sort → keep top k → project \(Y=P^TX\).

9. **LDA vs PCA in one line each.**  
   LDA uses labels to separate classes; PCA ignores labels and keeps variance.

10. **Link PCA ↔ MLP?**  
    Linear autoencoder bottleneck ≈ PCA compression.

11. **What does VC stand for? Define shatter and VC dimension.**  
    **VC = Vapnik–Chervonenkis.** Shatter = all \(2^{|S|}\) labelings realizable with zero error. VC dimension = size of the largest shatterable set (capacity of the model class).

12. **VC of a line in 2D?**  
    3 (can shatter 3 points; cannot shatter 4 in the XOR/diagonal pattern).

13. **Perceptron VCd?**  
    \(m\) = input dimension; order \(O(W)\).

14. **Why does SVM care about VC?**  
    Max margin controls effective capacity → better generalization than “any” separating line.

15. **Feature selection complexity?**  
    \(2^d-1\) subsets → usually greedy search.

---

## Lesson 5 — SVM

16. **Why prefer the middle separating line?**  
    Largest margin → less sensitive to new points near the boundary.

17. **Define margin and support vector.**  
    Margin = max empty strip half-width around the boundary. SVs = points on the margin edges that define the boundary.

18. **Primal SVM problem?**  
    \(\min\tfrac12 w^Tw\) s.t. \(t_i(w^Tx_i+b)\ge 1\).

19. **Relation margin ↔ \(\|w\|\)?**  
    \(M=1/\|w\|\). Max margin ⇔ min \(\|w\|\).

20. **Why QP / convex?**  
    Quadratic convex objective + linear constraints → unique global minimum.

21. **Dual expression for \(w^*\)?**  
    \(w^*=\sum\lambda_i t_i x_i\); \(\sum\lambda_i t_i=0\); \(\lambda_i>0\) only on SVs.

22. **How classify a new \(z\)?**  
    \(\mathrm{sign}(\sum\lambda_i t_i K(x_i,z)+b^*)\).

23. **What is the kernel trick?**  
    Compute \(\phi(x)^T\phi(y)\) via \(K(x,y)\) without building \(\phi\); enables nonlinear SVM in original space cost.

24. **Name three kernels.**  
    Poly \((1+x^Ty)^s\); sigmoid \(\tanh(\kappa x^Ty-\delta)\); RBF \(\exp(-\|x-y\|^2/(2\sigma^2))\).

25. **Soft margin: role of \(C\)?**  
    Small \(C\): larger margin, more slack OK. Large \(C\): fewer errors, smaller margin.

26. **Multi-class how?**  
    One-vs-rest: \(N\) binary SVMs; pick strongest score.

27. **Three advantages of SVM.**  
    Global min; max margin; sparse (SVs only).

28. **Three disadvantages.**  
    Binary by nature; kernel choice hard; slow/large-scale limits.

29. **Perceptron vs SVM on same separable data?**  
    Both separate; SVM unique max-margin; Perceptron any separator / smaller margin / depends on training path.

30. **Circles → 3D with what map?**  
    \(\phi=(x_1,x_2,x_1^2+x_2^2)\); separate with a plane on the third coord (radius²).

31. **Lab Q1: line, SVs, \(M\)?**  
    \(x_1+x_2=1.5\); SVs \((1,1),(1,0),(0,1)\); \(M=1/(2\sqrt{2})\).

32. **Can SVM and Perceptron give different lines when both “correct”?**  
    Yes — that’s the whole point of max margin.

---

# PART B — Combined formula sheet (L4 + L5)

| Topic | Formula / fact |
|-------|----------------|
| Hypersphere | \(v_n=(2\pi/n)v_{n-2}\); peak \(n=5\); ~0 for \(n\gtrsim20\) |
| \(S_W\) | \(\sum_c\sum_{j\in c}p_c(x_j-\mu_c)(x_j-\mu_c)^T\) |
| \(S_B\) | \(\sum_c(\mu_c-\mu)(\mu_c-\mu)^T\) |
| LDA goal | max \(w^TS_Bw / w^TS_Ww\); \(z=w^Tx\) |
| PCA | center; eig(\(\mathrm{cov}\)); \(Y=P^TX\); keep top \(\lambda\) |
| VC (2D line) | 3 |
| Perceptron VCd | \(m\) / \(O(W)\) |
| SVM primal | \(\min\tfrac12 w^Tw\) s.t. \(t_i(w^Tx_i+b)\ge1\) |
| Margin | \(M=1/\|w\|\) |
| Dual \(w\) | \(w^*=\sum\lambda_i t_i x_i\) |
| Predict | \(\sum\lambda_i t_i K(x_i,z)+b^*\) |
| Soft | \(0\le\lambda_i\le C\) |
| Poly / RBF / tanh | \((1+x^Ty)^s\) / \(\exp(-\|x-y\|^2/2\sigma^2)\) / \(\tanh(\kappa x^Ty-\delta)\) |
| Circle lift | \((x_1,x_2,x_1^2+x_2^2)\) |
| Lab5 Q1 | line \(x+y=1.5\); SVs (1,1)(1,0)(0,1); \(M=1/(2\sqrt2)\) |

**Key numbers:** volume peak dim **5**; VC line **3**; Vapnik SVM **1992**; hypersphere ~dead by dim **20**.

---

# PART C — Traps (L4 + L5)

1. Curse ≠ “volume always decreases” — it **increases then decreases** (peak at 5).
2. LDA needs **labels**; PCA does **not**.
3. PCA does **not** optimize class separation (may mix classes).
4. VC: to claim VC ≥ D find **one** shatterable set of size D; to claim VC = D show **no** set of size D+1 shatters.
5. SVM targets **±1**.
6. Margin half vs full width — state convention.
7. Kernel trick ≠ “always go to infinite dimensions blindly” — still validate kernel/params.
8. Soft-margin \(C\) direction: small C → soft/large margin.
9. After SVM training, non-SV training points are **redundant**.
10. “SVM can’t do nonlinear” is **false** — kernels make nonlinear decision boundaries in input space.

---

# PART D — 60-second oral drills

**Drill 1:** Explain curse → why DR → LDA vs PCA in under 60 seconds.  
**Drill 2:** Draw margin, mark SVs, write primal, say \(M=1/\|w\|\).  
**Drill 3:** Kernel trick in 3 sentences + list 3 kernels.  
**Drill 4:** Lab Q1 numbers from memory.  
**Drill 5:** VC shatter definition + why 2D line has VC = 3.

If you can do all five without notes, Lessons 4–5 are exam-ready.

**Oral answer for “What is VC?”:**  
Vapnik–Chervonenkis dimension — a number measuring model capacity: the largest number of points the model can shatter (classify correctly under every possible labeling). For straight lines in 2D it is 3.
