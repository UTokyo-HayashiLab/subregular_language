# Morphological Affix Learning Experiments

This repository provides scripts for preparing morphological datasets, training models, and evaluating their performance.  
All experiments can be run using the commands described below.  

## Installation

First, install the required dependencies:

```bash
pip install numpy scikit-learn matplotlib pandas
```

## Workflow

### 1. Download the Dataset
Download the Morpholex English dataset and save it as an Excel file:

```bash
python download_morpholex_en.py --out Morpholex_en.xlsx
```

### 2. Data Preparation

#### Convert Excel to CSV
```bash
python morpholex_to_affixcsv.py --xlsx Morpholex_en.xlsx --out morpholex_affixes.csv
```

#### Split the Dataset
```bash
python data_prep_morphology.py --input_csv morpholex_affixes.csv --out_dir data_en
```

### 3. Training

Train a model using the **PT** and **LTT** features:

```bash
python train_morphology.py --data_dir data_en --out_dir runs/pt_ltt --features PT LTT --m 2 --k 2
```

### 4. Evaluation and Plotting

Evaluate the trained model and generate plots:

```bash
python eval_plot_morphology.py --data_dir data_en --run_dir runs/pt_ltt --out_dir runs/pt_ltt/plots --m 2 --k 2
```

### 5. Linear Separability Check

To investigate the degree of linear separability, train with multiple values of `C`:

```bash
python train_morphology.py --data_dir data_en --out_dir runs/pt_ltt --features PT LTT --m 2 --k 2   --C 1.0 --path_Cs "0.1,1,10,100,1000"
```

### 6. Re-run Evaluation

Finally, evaluate again and generate updated plots:

```bash
python eval_plot_morphology.py --data_dir data_en --run_dir runs/pt_ltt --out_dir runs/pt_ltt/plots --m 2 --k 2
```

---

## Directory Structure

- **data_en/** : Preprocessed dataset split for training and testing  
- **runs/pt_ltt/** : Training results, models, and plots  

---

## Citation

If you use this code or dataset in your work, please consider citing Morpholex or referencing this repository.  
