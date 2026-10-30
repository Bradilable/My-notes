# Part 1 Supervised Learning
## Chapter 0 Basic concepts
$x^{(i)}$ is called an input **feature** and $y^{(i)}$ is called a **target**  
$(x^{(i)},y^{(i)})$ is called a **training example**, a list of training examples is called a **training set**  
$\mathcal{X}$ denotes the space of input values, $\mathcal{Y}$ denotes the space of output values  
Our goal is to learn a function $h : \mathcal{X} \mapsto \mathcal{Y}$ to predict the value of $y$  
$h$ is called a **hypothesis**  
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
Let us assume that the target variables and the inputs are related via the equation  

$$
y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}
$$  

where $\epsilon^{(i)}$ is an error term, and we assume $\epsilon^{(i)}$ are distributed IID according to Gaussian distribution  

$\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$  

$$
p(\epsilon^{(i)}) = \frac{1}{\sqrt{2\pi}\sigma}
\exp\left(-\frac{(\epsilon^{(i)})^2}{2\sigma^2}\right).
$$  

$$
p(y^{(i)} \mid x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi}\sigma}
\exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right).
$$

We have the **likelihood function**  

$$
L(\theta) = L(\theta; X, \vec{y}) = p(\vec{y} \mid X; \theta).
$$  

By the independence assumption, this can also be written  

$$
\begin{aligned}
L(\theta)
&= \prod_{i=1}^{n} p(y^{(i)} \mid x^{(i)}; \theta) \\
&= \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi}\sigma}
\exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right).
\end{aligned}
$$  

According to the **maximum likelihood** principal and **log likelihood** $\mathcal{l}(\theta)$  

$$
\begin{aligned}
\ell(\theta)
&= \log L(\theta) \\
&= \log \prod_{i=1}^{n}
\frac{1}{\sqrt{2\pi}\sigma}
\exp\left(-\frac{(y^{(i)}-\theta^T x^{(i)})^2}{2\sigma^2}\right) \\
&= \sum_{i=1}^{n} \log\left[
\frac{1}{\sqrt{2\pi}\sigma}
\exp\left(-\frac{(y^{(i)}-\theta^T x^{(i)})^2}{2\sigma^2}\right)
\right] \\
&= n\log\frac{1}{\sqrt{2\pi}\sigma}
-\frac{1}{\sigma^2}\cdot\frac{1}{2}
\sum_{i=1}^{n}(y^{(i)}-\theta^T x^{(i)})^2.
\end{aligned}
$$  

So maximizing $\mathcal{l}(\theta)$ gives the same answer as maximizing $\frac{1}{2}\sum_{i=1}^{n}\left(y^{(i)}-\theta^T x^{(i)}\right)^2$, which is same as LMS  
### 1.4  Locally weighted linear regression
The figure on the left: underfitting, the figure on the right: overfitting
<img width="470" height="131" alt="image" src="https://github.com/user-attachments/assets/43f40398-1b0a-4fb1-b513-a77dfd789ad8" />  
One solution: using **locally weighted linear regression**(LWR) which works as follows:  
1. Fit $\theta$ to minimize $\omega^{(i)}\sum_{i=1}^{n}\left(y^{(i)}-\theta^T x^{(i)}\right)^2$
2. Output $\theta^T x^{(i)}$
Here $\omega^{(i)}$ is the weight
A fairly stabdard choice for the weights is

$$
w^{(i)} = \exp\left(-\frac{(x^{(i)} - x)^2}{2\tau^2}\right)
$$  

$\tau$ is called the **bandwidth** parameter  
LMR is a sort of **non-parametric** algorithm, in contrast to (unweighted) linear regression, which is known as **parametric** algorithm  
## Chapter 2 Classification and logistic regression  
In this chapter we'll focus on the **binary classification**  
0 is called the negative class, 1 is called the positive class, $y$ given $x$ is also called the label
### 2.1 Logistic regression
First of all we need to change our hypothesis to fit our question  

$$
h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
$$

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

where $g(z)$ is called the **logistic function** or the **sigmoid function**
<img width="286" height="221" alt="image" src="https://github.com/user-attachments/assets/e7e87ea5-2e45-453b-bc01-1669f54d8efc" />  
The derivative of $g(z)$ is $g(z)(1 - g(z))$  
We assume that

$$
\begin{aligned}
P(y = 1 \mid x; \theta) &= h_\theta(x) \\
P(y = 0 \mid x; \theta) &= 1 - h_\theta(x)
\end{aligned}
$$

This can be written more compactly as  

$$
p(y \mid x; \theta) = (h_\theta(x))^y (1 - h_\theta(x))^{1-y}
$$

Assuming the n training examples were generated independently

$$
\begin{aligned}
L(\theta)
&= p(\vec{y} \mid X; \theta) \\
&= \prod_{i=1}^{n} p(y^{(i)} \mid x^{(i)}; \theta) \\
&= \prod_{i=1}^{n}
\left(h_\theta(x^{(i)})\right)^{y^{(i)}}
\left(1 - h_\theta(x^{(i)})\right)^{1-y^{(i)}}
\end{aligned}
$$

Again we'll take it's log likelihood

$$
\ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} y^{(i)} \log h(x^{(i)}) + (1-y^{(i)}) \log(1-h(x^{(i)}))
$$

Take it's derivative we have the stochastic gradient ascent rule

$$
\theta_j := \theta_j + \alpha \left(y^{(i)} - h_\theta(x^{(i)})\right)x_j^{(i)}
$$

### 2.2 Digression: the perceptron learning algorithm

$$
g(z) =
\begin{cases}
1 & \text{if } z \ge 0 \\
0 & \text{if } z < 0
\end{cases}
$$

If we use this threshold function to replace sigmoid function, we get the **perceptron learning algorithm**
### 2.3 Multi-class classification
We use the **softmax function** to handle the multi-class classification

```math
\begin{bmatrix}
P(y=1 \mid x;\theta) \\
\vdots \\
P(y=k \mid x;\theta)
\end{bmatrix}
=
\mathrm{softmax}(t_1,\ldots,t_k)
=
\begin{bmatrix}
\dfrac{\exp(\theta_1^T x)}{\sum_{j=1}^{k}\exp(\theta_j^T x)} \\
\vdots \\
\dfrac{\exp(\theta_k^T x)}{\sum_{j=1}^{k}\exp(\theta_j^T x)}
\end{bmatrix}.
```

So the negative log-likelihood will be  

$$
\ell(\theta) = \sum_{i=1}^{n} -\log\left(\frac{\exp(\theta_{y^{(i)}}^T x^{(i)})}{\sum_{j=1}^{k}\exp(\theta_j^T x^{(i)})}\right)
$$

It’s convenient to define the **cross-entropy** loss $\ell_{\mathrm{ce}} : \mathbb{R}^k \times \{1,\ldots,k\} \to \mathbb{R}_{\ge 0}$

$$
\ell_{\mathrm{ce}}((t_1,\ldots,t_k),y) = -\log\left(\frac{\exp(t_y)}{\sum_{j=1}^{k}\exp(t_j)}\right)
$$

With this notation we can simply rewrite our negative log-likelihood as

$$
\ell(\theta) = \sum_{i=1}^{n} \ell_{\mathrm{ce}}\left((\theta_1^T x^{(i)},\ldots,\theta_k^T x^{(i)}),y^{(i)}\right).
$$

The cross-entropy loss also has a simple gradient

$$
\frac{\partial \ell_{\text{ce}}(t, y)}{\partial t_i} = \phi_i - \mathbf{1}\{y = i\}
$$

In vectorized notations it has the following form

$$
\frac{\partial \ell_{\text{ce}}(t, y)}{\partial t} = \phi - e_y
$$

$$
\frac{\partial \ell(\theta)}{\partial \theta_i} = \sum_{j=1}^n (\phi_i^{(j)} - \mathbf{1}\{y^{(j)} = i\}) \cdot x^{(j)}
$$

### 2.4 Another algorithm for maximizing $\mathcal{l}(\theta)$
We can use Newton's method to find the zero of a function, similarly, we can use it to maximizing $\mathcal{l}(\theta)$  

$$
\theta := \theta - \frac{\ell'(\theta)}{\ell''(\theta)}
$$

generalize this to logistic regression, we get Newton-Paphson method

$$
\theta := \theta - H^{-1} \nabla_\theta \ell(\theta)
$$

Where $H$ is the **Hessian**  
The resulting method using Newton's method is also called **Fisher scoring**
## Chapter 3 Generalized linear models
### The exponential family
We say that a class of distributions is in the exponential family if it can be written in the form  

$$
p(y; \eta) = b(y) \exp(\eta^T T(y) - a(\eta))
$$

Here $\eta$ is called the **natural parameter**(also called the canonical parameter); $T(y)$ is the **sufficient statistic**(in most circumstances $T(y) = y$); and $a(\eta)$ is the **log partition function**  
It's easy to verify that Bernoulli and Gaussian distributions are examples of exponential family distributions  
There’re many other distributions that are members of the exponential family: The multinomial, the Poisson, the gamma and the exponential, the beta and the Dirichlet...  
### 3.2 Constructing GLMs
To derive a GLM model, we will make the following three assumptions:  
1. $y|x;\theta \sim$ ExponentialFamily($\eta$)
2. $h(x) = \mathrm{E}[y|x]$
3. $\eta$ and $x$ are related linearly
The target variable $y$ is also called **reponse variable** in GLM terminology
## Chapter 4 Generative learning algorithms
Algorithms that try to learn $p(y|x)$ directly are called **discriminative** learning algorithms, algorithms that instead try to model $p(x|y)$ are called **generative** learning algorithms  
After modeling $p(y)$ and $p(x|y)$, we can use Bayes rule to derive the posterior distribution 

$$
p(y|x) = \frac{p(x|y)p(y)}{p(x)}
$$

### 4.1 Gaussian discriminant analysis

$$
\begin{aligned}
p(y) &= \phi^y(1 - \phi)^{1-y} \\
p(x|y=0) &= \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp \left( -\frac{1}{2}(x - \mu_0)^T \Sigma^{-1} (x - \mu_0) \right) \\
p(x|y=1) &= \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp \left( -\frac{1}{2}(x - \mu_1)^T \Sigma^{-1} (x - \mu_1) \right)
\end{aligned}
$$

