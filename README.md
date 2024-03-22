# Dataset Discovery and Exploration: State-of-the-art, Challenges, and Opportunities



This repository contains the Python Notebooks
which illustrate the techniques and strategies discussed in the tutorial named
"Dataset Discovery and Exploration: State-of-the-art, Challenges, and Opportunities," 
set to be presented at EDBT 2024. The tutorial examines four key functionalities of dataset discovery and exploration:

This repository hosts the Python Notebooks that bring to life the methodologies and 
approaches covered in the tutorial titled 
"Dataset Discovery and Exploration: State-of-the-art, Challenges, and Opportunities,"
scheduled for presentation at EDBT 2024.
The 4 dataset discovery and exploration functionalities reviewed in the tutorial are:

- **Dataset Discovery.**
- **Dataset Navigation.**
- **Dataset Annotation.**
- **Schema Inference.**

As an example of how the four functionalities of data discovery and exploration work in practice,
this repository includes the following frameworks, or parts of them:
- **D3L [1] - Dataset Discovery.**
- **Aurum [2] - Dataset Navigation.**
- **TableMiner+ [3] - Dataset Annotation.**
- **Starmie [4] - Schema Inference.**



**NOTE:Semprop[5] is the currently implemented core function of Aurum. Our version is the simplified version
based on the description in Aurum [2].**

[1] A. Bogatu, A. A. A. Fernandes, N. W. Paton and N. Konstantinou, "Dataset Discovery in Data Lakes," 
2020 IEEE 36th International Conference on Data Engineering (ICDE), Dallas, TX, USA, 2020, pp. 709-720, doi: 10.1109/ICDE48307.2020.00067.

[2] R. Castro Fernandez, Z. Abedjan, F. Koko, G. Yuan, S. Madden and M. Stonebraker, "Aurum: A Data Discovery System," 
2018 IEEE 34th International Conference on Data Engineering (ICDE), Paris, France, 2018, pp. 1001-1012, doi: 10.1109/ICDE.2018.00094. 

[3] Zhang, Ziqi. “Effective and efficient Semantic Table Interpretation using TableMiner+.” Semantic Web 8 (2017): 921-957.

[4] Grace Fan, Jin Wang, Yuliang Li, Dan Zhang, and Renée J. Miller. 2023. Semantics-Aware Dataset Discovery from Data Lakes with Contextualized Column-Based Representation Learning. 
Proc. VLDB Endow. 16, 7 (March 2023), 1726–1739.

[5] R. Castro Fernandez et al., "Seeping Semantics: Linking Datasets Using Word Embeddings for Data Discovery," 2018 IEEE 34th International Conference on Data Engineering (ICDE), 
Paris, France, 2018, pp. 989-1000, doi: 10.1109/ICDE.2018.00093.


## Dataset Overview
This project utilizes structured data derived from the Web Data Commons project, focusing on:

- **T2Dv2 Gold Standard for Matching Web Tables to DBpedia**: 108 tables from 9 entity classes. [Link](https://webdatacommons.org/webtables/goldstandardV2.html).
- **Schema.org Table Corpus 2023**: 92 tables from 8 entity classes. [Link](https://webdatacommons.org/structureddata/schemaorgtables/2023/index.html#toc3).

Modifications are applied to the Schema.org Table Corpus data, which include:
- Conversion from JSON to CSV format for easier processing and analysis.
- Flattening of nested JSON structures (e.g., address objects) to create clear, tabular data with columns.

The datasets have been uploaded to this project as `Datasets.zip`. Simply extract this file in the current folder to access the data.



## Prerequisites
### Requirements (follows the requirement of all included framework)
- **Python 3.7.10 or higher.**
All needed packages are specified in requirement.txt.

For d3l and the Starmie framework, 
please directly download the GitHub repositories pointed to by the submodules. 
Follow the installation instructions provided in those repositories 
to ensure proper setup. Once installed, they can be used as intended.

**Subject Column Detection in TableMiner+**: This section needs to use the Google api for Google Custom Search
Engine, you can build your own engine via [Here](https://programmablesearchengine.google.com/intl/en_uk/about/). After setting the engine, in webSearchAPI.py, **cse_id** and **my_api_key** are needed to be changed.  
Currently, we use the parameter **SearchingWeb** 
TableColumnAnnotation class in TableAnnotation.py to disable the web search function in the Notebook.

Ensure you have the following installed:
- Python 3.7+ (Better to use Python 3.11)
- Jupyter Notebook
### Installation

Clone the repository to your local machine:
```bash
git clone https://github.com/PierreWoL/EDBTDemo
```
Install required packages.
```bash
pip install -r requirements.txt
```
Unzip the datasets.
```bash
unzip Datasets.zip
```
## Copyright and License Notice
This project incorporates data used under the **Apache License 2.0**. We respect all original data copyrights and license requirements and share our work on this basis.

## Dataset Overview
This project utilizes structured data derived from the Web Data Commons project, focusing on:

- **T2Dv2 Gold Standard for Matching Web Tables to DBpedia**: 108 tables from 9 entity classes. [Access here](https://webdatacommons.org/webtables/goldstandardV2.html).
- **Schema.org Table Corpus 2023**: 92 tables from 8 entity classes. [Access here](https://webdatacommons.org/structureddata/schemaorgtables/2023/index.html#toc3).

### Usage Guidelines

Datasets are not hosted directly due to licensing. Users are encouraged to download the data via the provided links and follow the project's scripts for processing.


## Citation
If you are using the code in this repo, please cite the following in your work:
```bibtex
Will provide reference later.
```