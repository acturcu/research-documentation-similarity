# Finding similar repositories using documentation

This project is part of the [2024 Research Project](https://github.com/TU-Delft-CSE/Research-Project) at [TU Delft](https://github.com/TU-Delft-CSE).

## Abstract
This repository is the code support for a paper that aims to study the importance of considering the documentation side of GitHub repositories when assessing the similarity between two or more applications. Readme and Wiki files, along with Comments from the source files are the set of dimensions proposed to be analyzed through our methodology and experiments. We propose a pipeline that first extracts text fragments from these dimensions and then applies Natural Language Processing techniques to further prepare our data for evaluation. To gather a similarity score we first vectorize our processed data with TF-IDF and then use cosine distance to obtain the score. Combinations of the three dimensions, ranging from using only one dimension to using all of them, are considered throughout our study. Moreover, additional information has been extracted from the plain text, such as URLs referred to and Licenses usage, the similarity of which was calculated using Jaccard distance.     

Two experiments were performed. The first one aims to observe the behavioral tendencies of our methodology applied to a small dataset, while the second one aims to validate our results. By evaluating them, we found sufficient data that supported our presented conclusion: documentation represents a valuable asset in gathering a pool of similar applications.        

## Experimental setup
To reproduce our experiments a couple of scripts must be run, in both Java (14) and Python (3.10). The required packages for the Python scrips are availble in 
requirements.txt. To install them, you can use `pip install -r requirements.txt`

Following, firstly you have to run `GetDocumentation.java` (src/main/java) to extract the documentation locally. 
Next, the Python scripts in folder scripts/ must be run in the following order: `text_processing.py`, `similarity_calculation.py`,
`crosssim_evaluation.py` and lastly `docs_distribution.py`.

Three scenarios for handling missing documentation have been analyzed. To gather an accuracy plot with all the scenarios,
you have to uncomment lines 150-204, 234-272 and 71 in `scripts/similarity_calculation.py`.

## Acknowledgment
I would like to acknowledge the developers of [CrossSim](https://github.com/crossminer/CrossSim) for using their evaluated repositories dataset during our validation experiment.