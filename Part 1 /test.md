Suppose we want to solve a machine learning problem. Our design is a deep neural network. Then our goal is to find the best suitable parameters of the neural network that solves the problem.

Let's write the parameters as $\theta$. We will introduce two different approaches of solving the parameters such that the neural network does what we want.

**APPROACH 1: Non-Probabilistic Learning**

Our first approach is non-probabilistic learning. We define our neural network as:

$f_{\theta} : \mathcal{X} \Rightarrow \mathcal{Y}$. Our function $f_{\theta}$ is a function of the family $\mathcal{F} = ( f_{\theta}(x) = y \vert x \in \mathcal{X}, y \in \mathcal{Y}, \theta \in \Theta )$.

What this means is that the function depends on the variable $\theta$. And depending on how we choose $\theta$, the function will return different values. So in a way, our function can behave differently depending on $\theta$.

Obviously, it is not the easiest task to find a suitible function $f_{\theta}$ for a given task. If it was, then Deep Learning Theory would be a solved problem. Hence it is up to the engineer to find a good architecture for $f_{\theta}$.

Assuming that we've chosen a neural network, we need to define a loss function. This is essentially something that we'd like to minimize. In a machine learning problem, we often have data $\mathcal{D} = \{ (x_i, y_i) \}_{i=1}^n$. In supervised tasks, we want to find a function $f_{\theta}(x)$ such that $f_{\theta}(x) \approx y$. Then it is obvious that $f_{\theta}(x) - y \approx 0$. If we define the loss function as $\mathcal{l}(y, f_{\theta}(x)) = (y-f_{\theta}(x))^2$, then obviously it would be good if for all points $x \in \mathcal{X}$ we have that $f_{\theta}(x) = y$. Then $f_{\theta}(x)$ would exactly be the function that gives the value $y$. This is obviously something that we want. But there's more to this depending on the problem. For example, the loss function that we just described (mean square error) is good for regression problems, while cross-entropy might be good for classification problems, and IoU for semantic segmation. The purpose of a loss function is that it should tell us how "well" the DNN is performing.

Obviously data can be corrupted with noise, and this in term may cause the neural network to adapt to the noise - which is something that we do not want. Because noise is error. Therefore we can use something called "regularization", which essentially helps with the corrupted data. If the neural network is trained, and the weights $\theta$ are changed in a way such that they behave as the corrupted data - then there is some error in the weights. We can adjust it by adding the regularization term into our training-part (which will be discussed soon). Let's just call a function $\Omega(\theta)$ as a function that does something with the weights.

When finding the appropriate weights, we'd like to minimize the loss function, but at the same time, if the loss function is too small, then it implies that the neural network has adapted towards the error. Therefore we can use this "weight-term", $\Omega(\theta)$. Now we'd like to solve this minimization problem:

$$\min_{\theta} \mathcal{l}(y, f_{\theta}(x)) + \Omega( \theta )$$

Obviously we're given datapoints, hence we'd like to solve this problem with $(x,y)$ being datapoints. Note that the loss function can be a very complicated one. For example, in SciML, one may use rules of differential equation such that the loss function contains this information.

**APPROACH 2: Probabilistic Learning**
