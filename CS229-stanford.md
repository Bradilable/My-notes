# Part 1 Supervised Learning
## Chapter 0 Basic concepts
$x^{(i)}$ is called an input **feature** and $y^{(i)}$ is called a **target**  
$(x^{(i)},y^{(i)})$ is called a **training example**, a list of training examples is called a **training set**  
$\mathcal{X}$ denotes the space of input values, $\mathcal{Y}$ denotes the space of output values  
Our goal is to learn a function $h : \mathcal{X} \mapsto \mathcal{Y}$ to predict the value of $y$  
$h$ is called z **hypothesis**  
If our target is continuous, we call the problem a **regression**; if the target only takes on discrete values, we call the problem a **classification**  
<img width="157" height="149" alt="image" src="https://github.com/user-attachments/assets/cc002d22-6e5f-45dc-bbbb-10ed58778dcc" />

## Chapter 1 Linear regression
$$
h(x) = \sum_{i=0}^{d} \theta_i x_i = \theta^T x
$$  

here $\theta_i$'s are parameters(weights), $x_0 = 1$(this is the interpret term)  
The cost function will be:  

$$
J(\theta) = \frac{1}{2} \sum_{i=1}^{n} \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2
$$  

This is the **odinary least squares** regression model  
### 1.1 LMS algorithm
We use the **gradient descent** to update $\theta$  

$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta).
$$  

Here $\alpha$ is called the **learning rate**, repeat the following step until convergence

$$
\theta_j := \theta_j + \alpha \sum_{i=1}^{n} \left(y^{(i)} - h_\theta(x^{(i)})\right)x_j^{(i)}, \quad (\text{for every } j)
$$  

This rule is called the **LMS**(stands for least mean squares) or the **Widrow-Hoff** learning rule  
We rewrite in the form of vector  

$$
\theta := \theta + \alpha \sum_{i=1}^{n} \left(y^{(i)} - h_\theta(x^{(i)})\right)x^{(i)}
$$  

This method looks at every example in the entire training set on every step, and is called **batch gradient descent**  
Since $J$ is a convex quadratic function, so we can always converge to a local optima(with appropriate $\alpha$)  
<img width="229" height="180" alt="image" src="https://github.com/user-attachments/assets/11bbea6d-2156-4628-9dc1-a2502eb43557" />  

$$
\theta := \theta + \alpha \left(y^{(i)} - h_\theta(x^{(i)})\right)x^{(i)}
$$  

In this algorithm, we repeatedly run through the training set, and each time we encounter a training example, we update the parameters according to the gradient of the error with respect to that single training example only. This algorithm is called **stochastic gradient descent**(also **incremental gradient descent**)  
Stachastic gradient descent "converge" faster in most cases, but it may never truly converge, when the training set is big, stochastic gradient descent is often preferred  
### 1.2 The normal equations
We define the derivative of $f$ with respect to $A$ to be:  

$$
\nabla_A f(A) =
\begin{bmatrix}
\frac{\partial f}{\partial A_{11}} & \cdots & \frac{\partial f}{\partial A_{1d}} \\
\vdots & \ddots & \vdots \\
\frac{\partial f}{\partial A_{n1}} & \cdots & \frac{\partial f}{\partial A_{nd}}
\end{bmatrix}
$$  

```math
\begin{aligned}
X\theta - \vec{y}
&=
\begin{bmatrix}
(x^{(1)})^T\theta \\
\vdots \\
(x^{(n)})^T\theta
\end{bmatrix}
-
\begin{bmatrix}
y^{(1)} \\
\vdots \\
y^{(n)}
\end{bmatrix} \\
&=
\begin{bmatrix}
h_\theta(x^{(1)}) - y^{(1)} \\
\vdots \\
h_\theta(x^{(n)}) - y^{(n)}
\end{bmatrix}.
\end{aligned}
```

$$
\begin{aligned}
\frac{1}{2}(X\theta - \vec{y})^T(X\theta - \vec{y})
&= \frac{1}{2}\sum_{i=1}^{n}
\left(h_\theta(x^{(i)}) - y^{(i)}\right)^2 \\
&= J(\theta).
\end{aligned}
$$

Calculate the derivatives and set them to zero, we obtain the **normal equations**

$$
X^T X\theta = X^T \vec{y}
$$  

### 1.3 Probabilistic interpretation
