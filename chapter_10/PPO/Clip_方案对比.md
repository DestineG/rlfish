---

---

## PPO Clip 方案对比

设 $Clip(r) = \max(1-\epsilon, \min(1+\epsilon, r)), \epsilon = 0.2$

### 方案 A：$A \cdot \text{clip}(r)$

$$
A_{\pi}(s, a) \min \big(\frac{\tilde{\pi}(a|s)}{\pi(a|s)}, \quad Clip(\frac{\tilde{\pi}(a|s)}{\pi(a|s)})\big)
$$

| 条件 | $r \le 0.8$ | $0.8 < r \le 1.2$ | $1.2 < r$ |
| :--- | :---: | :---: | :---: |
| **$A > 0$** (积极优势) | $r$ | $r$ | $1.2$ |
| **$A < 0$** (消极优势) | $r$ | $r$ | $1.2$ |

---

### 方案 B：$\text{clip}(A \cdot r)$

$$
\min \big(\frac{\tilde{\pi}(a|s)}{\pi(a|s)} A_{\pi}(s, a), \quad Clip(\frac{\tilde{\pi}(a|s)}{\pi(a|s)} A_{\pi}(s, a))\big)
$$

| 条件 | $r \le 0.8$ | $0.8 < r \le 1.2$ | $1.2 < r$ |
| :--- | :---: | :---: | :---: |
| **$A > 0$** (积极优势) | $r$ | $r$ | $1.2$ |
| **$A < 0$** (消极优势) | $0.8$ | $r$ | $r$ |

---

## 差异分析

* **方案 A** 对 $r$ 的裁剪是无偏的，不管 $A$ 是正还是负，都会对 $r$ 进行同样的裁剪
* **方案 B** 在动作变好时($A > 0$)抑制更新步长，在动作变差时($A < 0$)加重更新步长