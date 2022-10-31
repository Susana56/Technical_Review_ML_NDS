# Technical_Review_ML_NDS
Technical Review of Scientific Paper. 

For my Math Modeling course, we wrote a Technical Review and presented our results to our classmates. The paper we focused on was Machine Learning in Nonlinear Dynamical Systems
[https://arxiv.org/abs/2008.13496]. 

Here is our Technical Review Paper, it was a collaboration with fellow peers Joseph Diaz and Stephan Cline: https://drive.google.com/file/d/1tTSi9qxnEwvc0Pn6riOngtgRdmg4_utN/view?usp=sharing

The code above is my attempt at recreating their results on page 5. It was an exploration to using Tensorflow to code a NN
with their defined requirements. Our team was curious of potential extensions to these results, ie time-series data from a physcial source that is not uniformly spaced with noise etc. 

Coding a specificed problem allows you to realize misconceptions held and understand what is being done. The model works in predicting one time step
in the future, NOT a number of time steps in the future. This isn't that practical in real applications as you would need a large amount of data to correctly predict in the future for a long period of time. 

I had assumed it takes the predicted 1 step data as input to predict the following time step. This clearly leads to an accumulation of error as you continue. 

<img width="1124" alt="Lorenz_system" src="https://user-images.githubusercontent.com/38049811/199127406-23478aba-25c0-4700-9610-f25864213dda.png">
