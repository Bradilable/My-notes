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
