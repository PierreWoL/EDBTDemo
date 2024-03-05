# Dataset Discovery and Exploration: State-of-the-art, Challenges, and Opportunities

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
- **D3L - Dataset Discovery.**
- **Aurum - Dataset Navigation.**
- **TableMiner+ - Dataset Annotation.**
- **Starmie - Schema Inference.**


___NOTE: This repo is currently in progress. 
We will upload all frameworks along with their usage instructions 
to this repository prior to the commencement of the EDBT conference.___

## Prerequisites
### Requirements (follows the requirement of all included framework)
- **Python 3.7.10 or higher.**

- **PyTorch 1.9.0+cu111**

- **Transformers 4.9.2**

- **NVIDIA Apex (fp16 training)**
- **TBC**


For d3l and the Starmie framework, 
please directly download the GitHub repositories pointed to by the submodules. 
Follow the installation instructions provided in those repositories 
to ensure proper setup. Once installed, they can be used as intended.

Install requirements(will provide the repuirements later):

```bash
pip install -r requirements
```

Ensure you have the following installed:
- Python 3.x
- Jupyter Notebook
### Installation

Clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/yourrepositoryname.git
```

## Dataset Overview
This project utilizes structured data derived from the Web Data Commons project, focusing on:

- **2Dv2 Gold Standard for Matching Web Tables to DBpedia**: 108 tables from 9 entity classes. [Access here](https://webdatacommons.org/webtables/goldstandardV2.html).
- **Schema.org Table Corpus 2023**: 92 tables from 8 entity classes. [Access here](https://webdatacommons.org/structureddata/schemaorgtables/2023/index.html#toc3).

### Usage Guidelines

Datasets are not hosted directly due to licensing. Users are encouraged to download the data via the provided links and follow the project's scripts for processing.
For more information about data citations, please view the CITATION.cff file.




## Citation
If you are using the code in this repo, please cite the following in your work:
```bibtex
Will provide reference later.
```