# 🚀 CHIMERA Challenge Task 2

**Bcg Response Subtype Prediction In High-Risk Nmibc using Multi-modal Data**  

This repository contains code, models, and instructions for reproducing experiments for **Task 2 of the CHIMERA Challenge**.

---

## 📌 Table of Contents
1. [Overview](#overview)  
2. [Data](#data)  
3. [Full Pipeline Usage](#full-pipeline-usage)  
4. [Results](#results)  
5. [License](#license)  

---

<a name="overview"></a>
## 🧠 Overview
Task 2 focuses on **BCG response subtype prediction** using multi-modal data:  
- **Histopathology slides**  
- **Clinical features**  

<p align="left">
  <img src="framework.png" alt="Framework">
  <br>
  <em>Figure 1: Framework of the proposed method. Patch features are first extracted using the UNI model, and slide-level representations are generated with MADMIL. These representations, together with clinical data, are concatenated. Finally, the slide label is predicted using a linear fully connected classifier.</em>
</p>

---

<a name="data"></a>
## 📂 Data
The CHIMERA Task 2 dataset must be downloaded from the official challenge website.  
Organize the data as follows:  

```text
CHIM_ostu_10x/
├── pt_files/             # Patch feature files for each slide
├── clinicals/              # Clinical data files
clinical_preprocessor.pkl # Preprocessing object for clinical features
```

---

<a name="full-pipeline-usage"></a>
## 🛠 Full Pipeline Usage

Run the complete workflow for Task 2 in sequence:

```bash
# 1️⃣ Patch Extraction
python create_patches_fp.py \
    --source .../bladder-cancer-tissue-biopsy-wsi \
    --source_mask .../tissue-mask \
    --save_dir ./Bladder_10x_ \
    --patch_level 1 \
    --patch_size 224 \
    --step_size 224 \
    --seg \
    --patch

# 2️⃣ Feature Extraction
python extract_features.py \
    --data_h5_dir Bladder_10x \
    --data_slide_dir .../bladder-cancer-tissue-biopsy-wsi \
    --csv_path Bladder_10x/process_list_autogen.csv \
    --feat_dir ./CHIM_ostu_10x/feat_uni \
    --batch_size 256 \
    --slide_ext .tif
    
# Coords Extraction
python coord.py

# 3️⃣ Training
python train.py \
    --model_type madmil \
    --n 2 \
    --exp_code MADMIL_10x_CLIN_2e-1 \
    --reg 2e-1

# 4️⃣ Evaluation
python eval.py \
    --models_exp_code MADMIL_10x_CLIN_2e-1_s2021 \
    --save_exp_code MADMIL_10x_CLIN_2e-1 \
    --model_typ madmil \
    --n 2 
```

---

<a name="results"></a>
## 📊 Results

<a name="license"></a>
## ⚖ License

This repository is licensed under MIT License.
