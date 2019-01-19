## Experimental correlation analysis of bicluster coherence measures and gene ontology information

Victor A. Padilha and André C. P. L. F. de Carvalho<br/>
Institute of Mathematics and Computer Sciences, University of São Paulo, São Carlos, SP, Brazil<br/>
Email: victorpadilha@usp.br, andre@icmc.usp.br<br/>

*Abstract* — Biclustering algorithms have become popular tools for the analysis of gene expression data. They allow the identification of local patterns defined by subsets of genes and subsets of samples, which cannot be detected by traditional clustering algorithms. In spite of being useful, biclustering is a NP-hard problem. Therefore, the majority of biclustering algorithms look for biclusters optimizing a pre-established coherence measure. Many heuristics and validation measures have been proposed for biclustering in the last 20 years. However, there is a lack of an extensive comparison of bicluster coherence measures on practical scenarios. To deal with this lack, this paper experimentally analyzes $17$ bicluster publicly available coherence measures and external measures calculated from information obtained in the gene ontologies. In this analysis, results were produced by $10$ algorithms from the literature on $19$ gene expression datasets. According to the experimental results, a few pairs of strongly correlated coherence measures could be identified, which suggests redundancy. Moreover, the pairs of strongly correlated measures might change when dealing with normalized or non normalized data and biclusters enriched by different ontologies. Finally, there was no clear relation between coherence measures and assessment using information from gene ontology.

[Additional figures.](heatmaps.zip)<br/>
[Measures implementations.](bicmeasures.py)
