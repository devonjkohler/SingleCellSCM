# SingleCellSCM

A repository for fitting a structured causal model (SCM) to single cell experimental data. Can build a graphical models 
of the corresponding proteomic network using Indra.


## Development needs

1. Upstream data processing
   1. Conversion into required format 
   2. Missing value imputation
   3. Feature summarization
   4. Batch correction/normalization
2. Graph creation
   1. Retrieve biological relationships**
      1. Indra?
   2. Reduce to DAG
      1. Remove "bad" nodes (feature reduction)
   3. Build causal graph with latent edges*
3. Apply causal inference
   1. Find identifiable queries
      1. y0
   2. Fit model
      1. SCM
         1. What functional form
         2. doWhy(?) - https://py-why.github.io/dowhy/main/user_guide/gcm_based_inference
      2. non-parametric - Ananke(?)
      3. More advanced methods for dealing with cycles (relates to graph creation)
         1. Latent Variable Models - Sara's code
   3. Intervention Analysis
      1. ACE
      2. Confidence Intervals
      3. Distributional changes(?)
   4. Counterfactual Analysis
      1. Cluster cells and predict output?
         1. Look at impact of counterfactuals in different clusters and derive some meaning
      2. Simple application with doWhy is easy.. the question is why to do this?

* Available in package
** Partially available