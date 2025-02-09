# Projects

## Project list (spring 2025)
### Gradient-based hyperparameter optimization
**Motivation**:  Hyperparameter optimization is a problem of finding suitable hyperparameters given a validation (or sometimes test) dataset.
In contrast to classical hyperparameter optimization methods, gradient-based methods allow the researchers to perform hyperparameter optimization over a billion-dimmension search space. 
 
**Algorithms to implement (from simplets to hardest):**
* [T1-T2](http://proceedings.mlr.press/v48/luketina16.pdf) with [DARTS numerical approximation (section 2.3)](https://arxiv.org/pdf/1806.09055) 
* [Optimizing billion hyperparameters](http://proceedings.mlr.press/v108/lorraine20a/lorraine20a.pdf)
* [HOAG](http://proceedings.mlr.press/v48/pedregosa16.pdf)
* [DrMAD](https://arxiv.org/pdf/1601.00917) (attention: the algorithm is described poorly, need to proofread before implementation)

**Recommended stack**:
JAX or pytorch. **As with Evidence-based operators, the main problem here is to make this usable for a broad-class of tasks (provide good API).**

**Project details:** the code of T1-T2 and "Optimizing billion hyperparameters..." is already implemented on JAX by Konstantin Yakovlev.
T1-T2 [can be also found here](https://nni.readthedocs.io/en/v2.1/NAS/DARTS.html)

### Data-free distillation
**Motivation**:  Knowledge distillation is an approach to train/improve target model performace (student) using information from already trained large model (teacher). Classical approaches to this problem perofrm distillation using logits, responses or hidden state from teacher obtained from data. However in some cases we cannot use the original data, and thus this method become unapplicable. The idea of data-free distillation is to obtain some elements of data from the teacher and then perform distillation.

**Algorithms to implement (from simplets to hardest):**
* [Original paper, statistical features](https://arxiv.org/pdf/1710.07535)
* [Original paper, spectral features](https://arxiv.org/pdf/1710.07535)
* [Data-free adversarial distillation](https://arxiv.org/pdf/1912.11006)
* [Deep inversion](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yin_Dreaming_to_Distill_Data-Free_Knowledge_Transfer_via_DeepInversion_CVPR_2020_paper.pdf)

**Project details:** some mechanism of analyzing hidden states must be implemented (in pytorch this can be implemented using hooks or via computational graph inspection, for example).

## Project activities
Each team must assign roles for all teammates. Each activity is evaluated independently, thus number of activites per each teammate must be ~equal.

* Project planning: the student must describe (or gather with the team) a document with the following information:
	* Project name
	* Project scope (what algorithms will be implemented)
	* Project stack (what libraries will be used,  how the project will be integrated with other libraries)
	* A scheme of the project (no certain requirements for the format, but you can use IDEF or UML for better clarity if you want)
	* Present this plan
	* **Evaluation criterion:** presentation at Tech. meeting 1
  
* Proof of concept:this code must either:
	* Implement experiment with similar problem statement but with simplier tools
 		*Example: if you are working on feature selection, find a suitable model, dataset and perform L2-regularization
	* The implemented code must be used as a baseline for demo
 	* The implemented plots and performance crtieria must be used as a baseline for demo 
	* **Evaluation criterion:** presentation  at Tech. meeting 2


* Algorithm implementation (1 activity per teammate)
	*  **Evaluation criterion:** presentation at Tech. meeting 1, cross-review, presentation, tests correctly run the algorithm,  presentation at Tech. meeting 3
* Project wrapping:
	* Making the code uniform acros all the teammates
	* Controling the code quality/consistency
	* Wrapping the code into library, repository creation and support
	* **Evaluation criterion:** cross-review,  presentation, the links are correct, the code can be installed without problems, all the workflows are setup,  presentation at Tech. meeting 3
* Tests writing
	* **Evaluation criterion:** test coverage > 90%,  presentation at Tech. meeting 3
* Documentation writing
	* **Evaluation criterion:** cross-review,   presentation at Tech. meeting 2 (intermediate version), presentation at Tech. meeting 3
* Cross-review (1 activity per teammate)
* Making a final demo based on the basic code
	* **Evaluation criterion:** cross-review,  presentation\
* Blog post
	* **Evaluation criterion:** ,  presentation at Tech. meeting 2 (intermediate version), ,presentation at Tech. meeting 3, possibely external review

For the team of 2 people it means that each teammate must have 5 acitvities
For the team of 4 people each teammate must have 3 or 4 activities



## Projects

### Implitic reparametrization trick
**Number of people in team:** 2-4

**Motivation**: The vanilla ELBO estimation in generative models like VAE uses reparametrization trick: a method of sampling random variables with low variance of the gradient of parameter distribution. The problem with this method is that it's available only for the limited number of distributions. [There is a paper](https://arxiv.org/abs/1805.08498)  which proposes reformulation of sampling allowing to sample variables from any continuos distribution. The goal of this project is to wrap partial cases described in this paper and (in the perfect case) the most general method for arbitary distribution.
 
**Algorithms to implement (from simplets to hardest):**
* Sample from Gaussian distribution (for comparison with reparametrization trick)
* Sample from Dirichlet distribution
* Sample from the mixture of distribution of the same family
* Sample from any arbitary factorized distribution

  
**Recommended stack**: one can use pytorch/pyro (both have interfaces for distributions) or distrax for JAX. Other stack can also be used.

**Other links:**
* [Discussion of IRP in the TF probability thread](https://github.com/tensorflow/probability/issues/51?ref=https://githubhelp.com)
* [Some more discussions in the pytorch thread](https://discuss.pytorch.org/t/pytorch-reparametrization-method-for-gamma-dirichlet-von-mises-distribution/109472)
* [Mixutre distribution implementation](https://github.com/vsimkus/torch-reparametrised-mixture-distribution) 
* [Lab on the implementation of IRT](https://github.com/intsystems/BMM/blob/main-22/lab2/BarabanshchikovaTask2.ipynb)

  
**Problem details:** the team must understand what methods are must be implemented, how this must be integrated with large libraries
### Stochastic gating
**Number of people in team:** 2-4

 **Motivation**: One of the methods of proper feature/variable/parameter selection is called [Stochastic gating](http://proceedings.mlr.press/v119/yamada20a/yamada20a.pdf).The principle is the following: multiply each parameter of the model (or the feature if we are working with linear models) with some discrete variable and optimize both the parameter and the distribution of the discrete variable. Due to the continuity of the optimization, the discrete variable must be replaced with some kind of relaxation. 
 
**Algorithms to implement (from simplets to hardest):**
* Feature selection with L2 regularization and straight-through estimation [see here for example](https://arxiv.org/pdf/2006.06880) , see also [basic paper for ST-estimator](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=62c76ca0b2790c34e85ba1cce09d47be317c7235) 
* Gumbel-softmax for gating [see here](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720239.pdf) 
* [Original stochastic gating](https://proceedings.mlr.press/v119/yamada20a/yamada20a.pdf) 
* [Correlated features in stochastic gating](https://openreview.net/pdf?id=oDFvtxzPOx)
  
**Other links:**
* [Stochasting gating library](https://runopti.github.io/stg/)

**Problem details:** the team must decide the structure of the project: are the modules they propose are just layer classes (like dropout layers) or they propose some "solvers"? What's more preferable for reproducibility and furhter usage? 


### Discrete variables relaxation
**Number of people in team:** 2-4

 **Motivation**: For lots of mathematical problems we need an ability to sample discrete random variables. The problem is that due to continuos nature of deep learning optimization, the usage of truely discrete random variables is infeasible. Thus we use different relaxation method. One of them, [Concrete distribution](https://arxiv.org/abs/1611.01144) or [Gumbel-softmax](https://arxiv.org/abs/1611.00712)  (this is one distribution proposed in parallel by two research groups) is implemented in different DL packages. In this project the authors need to implement different alternatives to it.  Since the implementation is quite simple, each teammate must implement 2 algorithms, not 1.

**Algorithms to implement (from simplets to hardest):**
* [Relaxed Bernoulli](http://proceedings.mlr.press/v119/yamada20a/yamada20a.pdf) 
* [Correlated relaxed Bernoulli](https://openreview.net/pdf?id=oDFvtxzPOx)
*  [Gumbel-softmax TOP-K](https://arxiv.org/pdf/1903.06059) 
* [Straight-Trhrough Bernoulli, distrubtion (don't mix with Relaxed distribution from pyro)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=62c76ca0b2790c34e85ba1cce09d47be317c7235) 
* [Invertible Gaussian reparametrization](https://arxiv.org/abs/1912.09588) with KL implemented
* [Hard concrete](https://arxiv.org/pdf/1712.01312) 
* [REINFORCE](http://www.cs.toronto.edu/~tingwuwang/REINFORCE.pdf)  (not a distribution actually, think how to integrate it with other distributions)
*  [Logit-normal distribution](https://en.wikipedia.org/wiki/Logit-normal_distribution) with KL implemented and [Laplace-form approximation of Dirichlet](https://stats.stackexchange.com/questions/535560/approximating-the-logit-normal-by-dirichlet) 


**Recommended stack**:
* Some of the alternatives for GS were implemented in [pyro](https://docs.pyro.ai/en/dev/distributions.html), so it might be useful to play with them also.
  
**Problem details:** To make to library constistent, I propose to integrate imports of distriubtions from pyro and pytorch into the library, so that all the categorial distributions can be imported from one entrypoint. If you use another stack (say, JAX), please implement all the distributions from pyro/torch.

**Other links:**
* [About top-k GS](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/subsets.html) 


### Optimization operator as evidence estimators
**Number of people in team:**  2-4

**Motivation**:  "Classical" evidence lower bound approaches allows researcher to perform a simplified Bayesian inference over quite complex models, like deep learning models. This approach involves MC-like sampling at each optimization iteration. Alternative approach is to consider parameters W as a sample from unknown distribution that changes under action of optimization operator (like SGD) at each optimization step. From the researcher perspective, this approach is useufl because doesn't need to change the optimization at all.
 
**Algorithms to implement (from simplets to hardest):**
* Current standard for these algorithms: [SWA-G](https://github.com/wjmaddox/swa_gaussian) and [Stein variational gradient](https://docs.pyro.ai/en/dev/_modules/pyro/infer/svgd.html) (you can just call if from the library)
* [Evidence lower bound using SGD](https://arxiv.org/abs/1504.01344) ([There is also a Russian paper describing this approach](https://www.mathnet.ru/links/72b1758955d42cb2ad370be46365035a/at14742.pdf)
* ELBO, but with [preconditioned-SGLD](https://icml.cc/2011/papers/398_icmlpaper.pdf) instead of SGD.  ([There is also a Russian paper describing this approach](https://www.mathnet.ru/links/72b1758955d42cb2ad370be46365035a/at14742.pdf)  How to calculate entropy for this case: [Link](https://approximateinference.org/2015/accepted/AltieriDuvenaud2015.pdf) 
* [Competiting approach](https://www.jmlr.org/papers/volume18/17-214/17-214.pdf) 


### Bayesian deep compression
**Number of people in team:**  2-4

**Motivation**:  the project proposes to develop the library that unifies different approaches for model selection and parameter pruning. The focus here more on the model pruning: each algorithm proposes different strategies to prune/remove uninformative parameters. As a bonus for the 4th member of the team we propose to adapt Renyi divergence optimization for the model selection/pruning.

**Algorithms to implement (from simplets to hardest):**
* [A baseline ELBO: proposed by Graves in 2011](https://papers.nips.cc/paper/4329-practical-variational-inference-for-neural-networks). Must be implemented with [local reparameterization trick](https://arxiv.org/abs/1506.02557) , hyperparameter optimization and pruning.
* [Bayesian deep comrpession method: uses similar framework, but allows to effectively prune parameters due to the more sophisticated prior.](https://proceedings.neurips.cc/paper_files/paper/2017/file/69d1fc78dbda242c43ad6590368912d4-Paper.pdf) 
* [Alternative method: scalable Laplace approximation](https://discovery.ucl.ac.uk/id/eprint/10080902/1/kflaplace.pdf) 
* [Renyi divergence: a generalization of the ELBO which can potentially plugged into the algorithms 1 and 2 and make models more extedable](https://arxiv.org/abs/1602.02311) 
**Links**:
* [Most parts of the algorithms are implemented here](https://github.com/JavierAntoran/Bayesian-Neural-Networks) 
* [Renyi](https://docs.pyro.ai/en/1.9.1/_modules/pyro/infer/renyi_elbo.html) 

**Problem details:** one of the algorithms that must be implemented as a baseline (maybe just using pyro) is HMC. This is a very long running algorithm for posterio sampling/model selection, but with limits to short chains it can evaluate our Evidence and can be considered as a ground truth. See [some papers that evaluate model Evidence with HMC](https://arxiv.org/pdf/2002.02405).   and [this one, but it uses a very heavy HMC, we don't need it](http://proceedings.mlr.press/v139/izmailov21a/izmailov21a.pdf) .


### MDL for language model evaluation
**Number of people in team:** 2-3

 **Motivation**: One of the method of language model analysis (and usage) is probing: we train a classifier for some specific layer of language model to extract some information about the analyzed words. For example, for PoS or syntactic properties of the words. The question we want to analyze is that which layer should we use, how should we choose this layer, and, generally, how much information about the downstream task can we capture from this layer/model. For this task multiple researchers proposed different theoretical frameworks [1](https://arxiv.org/pdf/2109.03853) , [2](https://arxiv.org/pdf/2004.03061), [3](https://arxiv.org/pdf/2003.12298) . We propose to implement and compare different approaches used for this task.
 
**Algorithms to implement (from simplets to hardest):**
* [MDL approach, online coding](https://arxiv.org/pdf/2003.12298) 
* [MDL approach, variational coding](https://arxiv.org/pdf/2003.12298)
* [Bayesian approach](https://arxiv.org/pdf/2109.03853)

**Recommended stack**:
The implemented methods must be compatible with Hugging Face.

### Graphical model generation
**Number of people in team:**  2 

**Motivation**: Graphical models([wiki](https://en.wikipedia.org/wiki/Graphical_model) , see also [Bishop](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) is a wide class of probabilistic models covering both very simple models (like linear regression) and complex ones (for example, VAE or LDA models in machine learning). The question of automatic model construction is still open, since it involves different model design decision and interpretation from user. The project proposes to use LLM to generate such models by the description in natural langauage.
 
**Algorithms to implement:**
* Code generation including the properties of the model, input distriubtions, dependency of variables, etc (or graph generation that will be converted into code)
* Visualization of results
  
**Problem details:** among all the projects this looks most simple, but in reality this project is more a research project than a project on implementation of existing methods. There is a belief it will work (since it's very similar to code generation task), but there is no proof. So, please think twice before involving in this project.
   
**Stack:**
* The project must be integrated with hugging face. It's recommended to focus only on one certian LLM  family on your choice (LLama? Gemma? Mixtral?) to make prompts well tested. 
* [The standard python library for graphical models](https://github.com/pgmpy/pgmpy) 
* [Visualization of graphical models](http://theoryandpractice.org/stats-ds-book/pgm/daft.html) 
* Depending on the format of the resulting library, it can be deployed with a very simple web interface, see [Gradio, it's well integrated with HF](https://www.gradio.app/) .

