# MIMIC-III Clinical NLP: ICU Entity Extraction & Medical Embeddings

## Overview
An advanced NLP pipeline leveraging spaCy, SciSpacy, MedSpacy, and ClinicalBERT to extract clinically relevant entities from MIMIC-III ICU notes, with a focus on stroke (ICD-9: 430, 431, 434.x) and other high-impact conditions. It applies medical-specific text normalization, syntactic dependency parsing, and custom word embeddings (Word2Vec, BioWordVec, ClinicalBERT) to enhance entity recognition and semantic analysis. The project integrates t-SNE clustering, frequency distributions, and dependency visualization to uncover patterns in disease progression, treatment pathways, and patient stratification within critical care settings.

## Features
- **Efficient Data Loading**: Uses DuckDB queries for optimized retrieval of clinical notes based on ICD-9 codes and note categories.
- **Multi-Model Entity Extraction**: Compares SpaCy’s general-purpose model with SciSpacy’s biomedical model and MedSpacy for clinical context handling.
- **Word Embeddings & Semantic Analysis**: Trains custom Word2Vec embeddings on extracted entities or loads pre-trained embeddings from Gensim.
- **Relationship Extraction & Knowledge Graphs**: Identifies subject-verb-object relationships and constructs knowledge graphs using NetworkX.
- **Advanced Visualization**: Provides t-SNE embeddings, entity overlap plots, contextual attribute displays, and dependency parsing visualizations.
- **GPU Acceleration**: Leverages CUDA-enabled hardware for efficient deep learning-based text processing.

## Installation
### Prerequisites
Ensure Python 3.10+ is installed, along with the necessary dependencies. It is recommended to use a Conda environment for dependency management.

```bash
# 1. Create and activate the conda environment
conda create -n mimic_spacy python=3.10 -y
conda activate mimic_spacy

# 2. Install essential libraries via conda-forge
conda install -c conda-forge duckdb gensim scikit-learn matplotlib -y
conda install -c conda-forge pandas numpy tqdm networkx seaborn umap-learn matplotlib-venn -y

# 3. Install CUDA support for GPU acceleration
conda install -c conda-forge cudatoolkit=11.8 -y
conda install -c conda-forge cudnn -y
conda install -c conda-forge pytorch torchvision torchaudio pytorch-cuda=11.8 -y

# 4. Install specific spaCy version (NOT via conda to ensure version control)
pip install spacy==3.4.4

# 5. Install base spaCy model
python -m spacy download en_core_web_sm

# 6. Install specific scispacy version and models
pip install scispacy==0.5.1
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz

# 7. Install transformers for ClinicalBERT
conda install -c conda-forge transformers -y

# 8. Verify installation
python -c "import spacy; import scispacy; import torch; print(f'SpaCy version: {spacy.__version__}'); print(f'GPU Available: {torch.cuda.is_available()}')"
```

## Usage
Run the script from a terminal:

```bash
python mimic_nlp.py \
  --db_path "path/to/mimic.duckdb" \
  --icd9_codes "430,431,434.x" \
  --note_category "Nursing/other" \
  --embedding_choice both \
  --visualize_entities \
  --extract_relationships \
  --output_path results \
  --use_medspacy
```

Adjust flags as needed. Logs, outputs, and generated plots will be stored in the `results` directory.

## Project Structure
```
.
├── mimic_nlp.py        # Main NLP pipeline script
├── README.md           # Documentation
├── requirements.txt    # Dependencies
├── output/             # Default output folder
└── ...
```

## Data Access Requirement
This project utilizes clinical text data from the **MIMIC-III** database, which contains de-identified electronic health records from critical care units. **Access to MIMIC-III is restricted** and requires credentialed access through **PhysioNet**. To use this dataset:

1. Register for an account at [PhysioNet](https://physionet.org/)
2. Complete the required training on data usage ethics and privacy
3. Request access to the [MIMIC-III dataset](https://physionet.org/content/mimiciii/1.4/)
4. Once granted access, download the relevant ICU note files for processing

Without valid access, users will not be able to retrieve and analyze clinical notes. Please ensure compliance with all ethical and legal requirements when handling sensitive medical data.

## GPU Utilization
This project leverages GPU acceleration where available to optimize NLP model inference times. The use of ClinicalBERT and other transformer-based models benefits from CUDA-enabled hardware, significantly reducing computation time for entity extraction and embedding generation.

## License
This project is distributed under the MIT License.
