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

Note that this model is usually applied using only one covariance matrix  
The log likelihood of the data is give by  

$$
\begin{aligned}
\ell(\phi, \mu_0, \mu_1, \Sigma) &= \log \prod_{i=1}^n p(x^{(i)}, y^{(i)}; \phi, \mu_0, \mu_1, \Sigma) \\
&= \log \prod_{i=1}^n p(x^{(i)}|y^{(i)}; \mu_0, \mu_1, \Sigma)p(y^{(i)}; \phi).
\end{aligned}
$$  

By maximizing with respect to the parameters, we find the maximum likelihood estimate of the parameters to be  

$$
\begin{aligned}
\phi &= \frac{1}{n} \sum_{i=1}^n 1\{y^{(i)} = 1\} \\
\mu_0 &= \frac{\sum_{i=1}^n 1\{y^{(i)} = 0\}x^{(i)}}{\sum_{i=1}^n 1\{y^{(i)} = 0\}} \\
\mu_1 &= \frac{\sum_{i=1}^n 1\{y^{(i)} = 1\}x^{(i)}}{\sum_{i=1}^n 1\{y^{(i)} = 1\}} \\
\Sigma &= \frac{1}{n} \sum_{i=1}^n (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T
\end{aligned}
$$  

<img width="275" height="222" alt="image" src="https://github.com/user-attachments/assets/15b7476a-39c9-4ce6-8fb2-1b50a566f42a" />

The GDA model has an interesting relationship to logistic regression. GDA makes stronger modeling assumptions about the data than does logistic regression  
Specifically, when $p(x|y)$ is indeed Gaussian (with shared $\Sigma$), then GDA is **asymptotically efficient**  
### 4.2 Naive bayes

## Chapter 5 Kernel methods
### 5.1 Feature maps
Instaed of using a linear function, consider fitting cubic function $y = \theta_3 x^3 + \theta_2 x^2 + \theta_1 x + \theta_0$  

$$
\phi(x) = \begin{bmatrix} 1 \\ x \\ x^2 \\ x^3 \end{bmatrix} \in \mathbb{R}^4.
$$

We call the map $\phi$ as **feature map**, the original input $x$ are called the **attributes**, the new quantities are called **features**
### 5.2 LMS with features  
Replace all the occurrences of $x^{(i)}$ in the algorithm by $\phi(x^{(i)})$ give us the new update  

$$
\theta := \theta + \alpha \sum_{i=1}^n (y^{(i)} - \theta^T \phi(x^{(i)})) \phi(x^{(i)})
$$

$$
\theta := \theta + \alpha \left( y^{(i)} - \theta^T \phi(x^{(i)}) \right) \phi(x^{(i)})
$$

### 5.3 LMS with the kernel trick

$$
\phi(x) = \begin{bmatrix} 1 & x_1 & x_2 & \cdots & x_1^2 & x_1x_2 & x_1x_3 & \cdots & x_2x_1 & \cdots & x_1^3 & x_1^2x_2 & \cdots \end{bmatrix}^T.
$$

The dimension of our features seems too big which makes the algorithm computationally expensive  
Assume at some point, $\theta$ can be represented as  

$$
\theta = \sum_{i=1}^n \beta_i \phi(x^{(i)})
$$

Then we claim that in the next round, $\theta$ is still a linear combination of $\phi(x^{(1)})$,..., $\phi(x^{(n)})$ because

$$
\begin{aligned}
\theta &:= \theta + \alpha \sum_{i=1}^n (y^{(i)} - \theta^T \phi(x^{(i)})) \phi(x^{(i)}) \\
&= \sum_{i=1}^n \beta_i \phi(x^{(i)}) + \alpha \sum_{i=1}^n (y^{(i)} - \theta^T \phi(x^{(i)})) \phi(x^{(i)}) \\
&= \sum_{i=1}^n \underbrace{(\beta_i + \alpha (y^{(i)} - \theta^T \phi(x^{(i)})))}_{\text{new } \beta_i} \phi(x^{(i)})
\end{aligned}
$$

$$
\beta_i := \beta_i + \alpha \left( y^{(i)} - \theta^T \phi(x^{(i)}) \right)
$$

$$
\forall i \in \{1, \dots, n\}, \beta_i := \beta_i + \alpha \left( y^{(i)} - \sum_{j=1}^n \beta_j \phi(x^{(j)})^T \phi(x^{(i)}) \right)
$$

1)We can pre-compute the pairwise inner products < $\phi(x^{(i)})$, $\phi(x^{(j)})$ > for all pairs of $i$, $j$ before the loop starts  
2)Instaed of computing < $\phi(x)$, $\phi(z)$ >, we can use < $x$, $z$ > to calculate the inner products  
We define the **Kernel** as a function  

$$
K(x, z) \triangleq \langle \phi(x), \phi(z) \rangle
$$

$K$ is also used to present the kernel matrix, which is a $n \times n$ matrix with $K_{ij} = K(x^{(i)},x^{(j)})$, we have

$$
\beta := \beta + \alpha(\vec{y} - K\beta)
$$

$$
\theta^T \phi(x) = \sum_{i=1}^n \beta_i \phi(x^{(i)})^T \phi(x) = \sum_{i=1}^n \beta_i K(x^{(i)}, x)
$$

### 5.4 Properties of kernels
The kernel $K(x, z) = (x^T z + c)^k$ correspondsto a feature mapping to an $\binom{d+k}{k}$ feature space, corresponding of all monomials of the form $x_{i_1}$ $x_{i_2}$ ... $x_{i_k}$ that are up to order $k$  
**Kernels as similarity metrics**: since Kernel contains inner product, it could be seen as a metric of similarity  
An example is **Gaussian kernel**

$$
K(x, z) = \exp \left( -\frac{\|x - z\|^2}{2\sigma^2} \right)
$$

The necessary and sufficient condition for $K$ to be a valid kernel(also called a **Mercer kernel**)  

<img width="504" height="78" alt="image" src="https://github.com/user-attachments/assets/45457809-3718-4c36-bef7-d0f867eba735" />

## Chapter 6 Support vector machines  
### 6.1 Margins: intuition
The decision boundary given by the equation $\theta^{T}x = 0$ is called the **separating hyperplane** 

<img width="295" height="269" alt="image" src="https://github.com/user-attachments/assets/6a7e2def-f39c-4ec0-92f0-00fb607a5e9e" />

We see that if a point is far from the separating hyperplane, then we may be significantly more confident in our predictions  
Informally we think it would be nice if, given a training set, we manage to find a decision boundary that
allows us to make all correct and confident predictions on the training examples  
### 6.2 Notation  
We now use $y \in \\{-1, 1\\}$ (instead of $\\{0, 1\\}$)  
Rather thanparameterizing our linear classifier with the vector $\theta$, we write our classifier as

$$
h_{w,b}(x) = g(\omega^T x + b)
$$

### 6.3 Functional and geometric margins
Given a training example($x^{(i)}$, $y^{(i)}$), we define the **functional margin** of ($\omega$, $b$)  
We see that by scaling $\omega$ and $b$ we can make the functional margin arbitrarily large without really changing anything meaningful  
We do the normalization by replacing ($\omega$, $b$) with $(w / \|w\|_2, b / \|w\|_2)$  
For a training set $S = \\{(x^{(i)},y^{(i)}); i = 1,...,n \\}$, we also define the functional margin of ($\omega$, $b$)with respect to $S$ as the smallest of the functional margins of the individual training examples as follows

$$
\hat{\gamma} = \min_{i=1,\dots,n} \hat{\gamma}^{(i)}
$$

<img width="298" height="257" alt="image" src="https://github.com/user-attachments/assets/a0c19f58-e1e6-4caf-b983-03073e9f2d3d" />

$$
w^T \left( x^{(i)} - \gamma^{(i)} \frac{w}{\|w\|} \right) + b = 0.
$$

$$
\gamma^{(i)} = \frac{w^T x^{(i)} + b}{\|w\|} = \left( \frac{w}{\|w\|} \right)^T x^{(i)} + \frac{b}{\|w\|}.
$$

Gernerally, we define the **geometric margin**

$$
\gamma^{(i)} = y^{(i)} \left( \left( \frac{w}{\|w\|} \right)^T x^{(i)} + \frac{b}{\|w\|} \right)
$$

For a training set $S$, we also define the geometric margin as

$$
{\gamma} = \min_{i=1,\dots,n} {\gamma}^{(i)}
$$

### 6.4 The optimal margin classifier
Assume that we are given a training set that is linearly separable
The optimization problem turns out to be  

$$
\begin{aligned}
&\max_{\gamma, w, b} \quad \gamma \\
&\text{s.t.} \quad y^{(i)}(w^T x^{(i)} + b) \ge \gamma, \quad i = 1, \dots, n \\
&\quad \quad \ \, \|w\| = 1.
\end{aligned}
$$

The $\|\omega\| = 1$ constraint is a non-comvex one, so instead of optimizing $\gamma$ we choose $\hat{\gamma} / \|{\omega}\|$ as our goal

$$
\begin{aligned}
&\max_{\hat{\gamma}, w, b} \quad \frac{\hat{\gamma}}{\|w\|} \\
&\text{s.t.} \quad y^{(i)}(w^T x^{(i)} + b) \ge \hat{\gamma}, \quad i = 1, \dots, n
\end{aligned}
$$

We will introduce the scaling constraint that the functional margin of $\omega$, $b$ with respect to the training set must be 1， and we have the following optimization problem

$$
\begin{aligned}
\min_{w,b} \quad & \frac{1}{2} \|w\|^2 \\
\text{s.t.} \quad & y^{(i)}(w^T x^{(i)} + b) \ge 1, \quad i = 1, \dots, n
\end{aligned}
$$

The solution gives us the **optimal margin classifier**, this optimization problem can be solved using commercial quadratic programming(QP) code
### 6.5 Lagrange duality
Consider a problem of the following form:

$$
\begin{aligned}
\min_w \quad & f(w) \\
\text{s.t.} \quad & h_i(w) = 0, \quad i = 1, \dots, l.
\end{aligned}
$$

We define the **Lagrangian** to be

$$
\mathcal{L}(w, \beta) = f(w) + \sum_{i=1}^l \beta_i h_i(w)
$$

Here the $\bata_i$'s are called the **Lagrange multipliers**  
Set $\mathcal{L}$'s partial derivatives to zero

$$
\frac{\partial \mathcal{L}}{\partial w_i} = 0; \quad \frac{\partial \mathcal{L}}{\partial \beta_i} = 0
$$

Consider the **primal** optimization problem

$$
\begin{aligned}
\min_w \quad & f(w) \\
\text{s.t.} \quad & g_i(w) \le 0, \quad i = 1, \dots, k \\
& h_i(w) = 0, \quad i = 1, \dots, l.
\end{aligned}
$$

To solve it, we start by defining the **generalized Lagrangian**

$$
\mathcal{L}(w, \alpha, \beta) = f(w) + \sum_{i=1}^k \alpha_i g_i(w) + \sum_{i=1}^l \beta_i h_i(w)
$$

$$
\theta_{\mathcal{P}}(w) = \max_{\alpha, \beta : \alpha_i \ge 0} \mathcal{L}(w, \alpha, \beta)
$$

We have $\theta$ as a criterion

$$
\theta_{\mathcal{P}}(w) = \begin{cases} 
f(w) & \text{if } w \text{ satisfies primal constraints} \\
\infty & \text{otherwise.}
\end{cases}
$$

# Part 2 Deep learning
