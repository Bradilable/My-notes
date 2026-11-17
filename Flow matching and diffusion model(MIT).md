# 1 Introduction
We identify the objects being generated as vectors $z \in \mathcal{R}^d$  
We can view the diversity of possible images as a **probability distribution** over the **space of images**  
This distribution is called the **data distribution**, which is denoted by $p_{data}$  
$p_{data}$ could be thought of as a probability density, which gives higher value on true images rather than noise  
Generating an object $z$ is modeled as sampling from the data distribution $z \sim p_{data}$  
A dataset consists of a finite number of iid samples $z_1,...,z_N \sim p_{data}$  
Guided generation involves sampling from $z \sim p_{data}(·|y)$, where $y$ is a conditioning variable  
We call the $p_{data}(·|y)$ the **guided data distribution**  
# 2 Flow and Diffusion Models
## 2.1 Flow Models
A solution to an ODE is defined by a **trajectory**

$$
X : [0, 1] \to \mathbb{R}^d, \quad t \mapsto X_t
$$

Every ODE is defined by a **vector field** $u$, i.e. a function of the form  

$$
u : \mathbb{R}^d \times [0, 1] \rightarrow \mathbb{R}^d, \quad (x, t) \mapsto u_t(x)
$$

We want a trajectory $X$ that “follows along the lines” of the vector field $u_t$, starting at the point $x_0$  
We may now ask: if we start at $X_0=x_0$ at $t=0$, where are we at time $t$ (what is $X_t$)? This question is answered by a function called the **flow**, which is a solution to the ODE  

$$
\begin{gathered}
\psi : \mathbb{R}^d \times [0, 1] \rightarrow \mathbb{R}^d, \quad (x_0, t) \mapsto \psi_t(x_0) \\
\frac{\mathrm{d}}{\mathrm{d}t} \psi_t(x_0) = u_t(\psi_t(x_0)) \\
\psi_0(x_0) = x_0
\end{gathered}
$$

<img width="495" height="114" alt="image" src="https://github.com/user-attachments/assets/96e6fe6c-2c86-4cc5-a2a1-eaee8a0d8407" />

**Flow existence and uniqueness**
If $u : \mathbb{R}^d \times [0, 1] \rightarrow \mathbb{R}^d$ is continuously differentiable with a bounded derivative, then the ODE has a unique solution given by a flow $\psi_t$. In this case, $\psi_t$ is a **diffeomorphism** for all t, i.e. $\psi_t$ is continuously differentiable with a continuously differentiable inverse $\psi_t^{-1}$  
In general, it's not possible to compute the flow $\psi_t$ explicitly if $u_t$ is not that simple. In these cases, one uses **numerical methods** to simulate ODEs  
One of the simplest and most intuitive methods is the **Euler method** in which we initialize $X_0=x_0$ and update via  

$$
X_{t+h} = X_t + h u_t(X_t) \quad (t = 0, h, 2h, 3h, \dots, 1 - h)
$$

where $h=n^{-1}$ is the stepsize  
We also have **Huen's method**  

$$
\begin{aligned}
X'_{t+h} &= X_t + h u_t(X_t) \\
X_{t+h} &= X_t + \frac{h}{2} \left( u_t(X_t) + u_{t+h}(X'_{t+h}) \right)
\end{aligned}
$$

We can now construct a generative model via an ODE by making the vector field a neural network vector field $u^{\theta}_t$  

To inject some randomness, we simple make the initial condition $X_0$ random. Specifically, we choose an initial distribution$p_{inti}$. In most cases, we set $p_{inti} \sim \mathcal{N}(0,I_d)$ to be a simple
standard Gaussian.
So a flow model will be  

$$
\begin{gathered}
X_0 \sim p_{\text{init}} \\
\frac{\mathrm{d}}{\mathrm{d}t} X_t = u_t^\theta(X_t)
\end{gathered}
$$

Our goal is to make the endpoint $X_1$ of the trajectory have distribution $p_{data}$, i.e.  

$$
X_1 \sim p_{\text{data}} \quad \Leftrightarrow \quad \psi_1^\theta(X_0) \sim p_{\text{data}}
$$

Note however: although it is called flow model, the neural network parameterizes the vector field, not the flow  

<img width="517" height="236" alt="image" src="https://github.com/user-attachments/assets/e3341530-0df8-4fde-815c-af7a3f891a3e" />

## 2.2 Diffusion Models
A stochastic trajectory is commonly called a stochastic process $(X_t)_{0 \le t \le 1}$ and is given by

$$
\begin{aligned}
& \qquad \qquad \qquad X_t \text{ is a random variable for every } 0 \le t \le 1 \\
& X : [0, 1] \to \mathbb{R}^d, \quad t \mapsto X_t \text{ is a random trajectory for every draw of } X
\end{aligned}
$$

SDEs are constructed via a **Brownian motion**. A Brownian motion $W=(W_t)_{0 \le t \le 1}$ is a stochastic
process such that $W_0=0$, the trajectories $t \to W_t$ are continuous, and the following two conditions hold:  

$$
\begin{aligned}
& \textbf{1. Normal increments:} \quad W_t - W_s \sim \mathcal{N}(0, (t-s)I_d) \text{ for all } 0 \le s < t, \text{ i.e. increments have a Gaussian distribution} \\
& \text{with variance increasing linearly in time } (I_d \text{ is the identity matrix}). \\
& \textbf{2. Independent increments:} \quad \text{For any } 0 \le t_0 < t_1 < \dots < t_n = 1, \text{ the increments } W_{t_1} - W_{t_0}, \dots, W_{t_n} - W_{t_{n-1}} \\
& \text{are independent random variables.}
\end{aligned}
$$

Brownian motion is also called a **Wiener process**  
We can easily simulate a Brownian motion approximately  

$$
W_{t+h} = W_t + \sqrt{h}\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I_d) \quad (t = 0, h, 2h, \dots, 1 - h)
$$

We need an equivalent formulation of ODEs that does not use derivatives so that we can use it on SDEs  
Since we already know  

$$
\begin{aligned}
& \hspace{1.2cm} \frac{\mathrm{d}}{\mathrm{d}t} X_t = u_t(X_t) \\
\{\Leftrightarrow} \quad & \frac{1}{h} (X_{t+h} - X_t) = u_t(X_t) + R_t(h) \\
\Leftrightarrow \quad & X_{t+h} = X_t + h u_t(X_t) + h R_t(h)
\end{aligned}
$$

```math
X_{t+h} = X_t
+ \underbrace{h u_t(X_t)}_{\text{deterministic}}
+ \sigma_t \underbrace{(W_{t+h} - W_t)}_{\text{stochastic}}
+ \underbrace{h R_t(h)}_{\text{error term}}
```

where $\sigma_t ≥ 0$ describes the **diffusion coefficient**  
The above describes a stochastic differential equation(SDE). It is common to denote it in the following symbolic notation:  

$$
\begin{gathered}
\mathrm{d}X_t = u_t(X_t)\mathrm{d}t + \sigma_t \mathrm{d}W_t \\
X_0 = x_0
\end{gathered}
$$

We use **Euler-Maruyama method** to simulate the SDE process  

$$
X_{t+h} = X_t + h u_t(X_t) + \sqrt{h}\sigma_t \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I_d)
$$

Thus, the diffusion model is given by  

$$
\begin{gathered}
X_0 \sim p_{\text{init}} \\
\mathrm{d}X_t = u_t^\theta(X_t)\mathrm{d}t + \sigma_t \mathrm{d}W_t
\end{gathered}
$$

<img width="509" height="227" alt="image" src="https://github.com/user-attachments/assets/85d6dc79-80fd-49e3-87fc-1b6e74532215" />

# 3 Flow Matching
We first consider the ODE process, we have our flow model as  

$$
X_0 \sim p_{\text{init}}, \quad \mathrm{d}X_t = u_t^\theta(X_t)\mathrm{d}t
$$

We use the end point $X_1$ as samples, our goal is to let the distribution of $X_1$ approximate $p_{data}$  
## 3.1 Conditional and Marginal Probability Path
The first step of flow matching is to specify a probability path, which specifies a gradual interpolation between noise $p_{inti}$ and data $p_{data}$  
For a data point $z \in \mathcal{R}^d$, we denote with $\delta z$ the **Dirac delta** “distribution”, sampling from $\delta z$ always returns z.  A **conditional (interpolating) probability path** is a set of distribution $p_t(x|z)$ over $\mathcal{R}^d$ such that:  

$$
p_0(\cdot | z) = p_{\text{init}}, \quad p_1(\cdot | z) = \delta_z \quad \text{for all } z \in \mathbb{R}^d
$$


