# 🐦 BirdCLEF+2025 Competition - INT3405-SV Team

---

## 📜 Abstract
This repository contains the working notes and code for the **INT3405-SV team** participating in the **BirdCLEF+2025** competition. The competition focuses on **species identification from 1-minute audio recordings** in the Middle Magdalena Valley of Colombia, covering **birds, amphibians, mammals, and insects**.  

We explored two main approaches for classification:  

1. **CNN on Mel-Spectrograms**  
2. **Transfer Learning on Audio Embeddings**  

Our final ranking was **536th** with a leaderboard ROC-AUC score of **0.817**.

**Keywords:** Transfer Learning, Dataset Annotation, Embeddings, Association Rule Mining, BirdNET, EnCodec, Google Bird Vocalization Classifier.

---

## 🐤 Competition Overview
BirdCLEF+2025 challenges participants to identify vocalizing animals in 1-minute soundscapes recorded at **El Silencio Natural Reserve**. Unlike previous editions, the dataset includes **four taxonomic groups**, with **182 target species**.  

The task is to **predict the presence of each species in 5-second segments** of test soundscapes.

---

## 🔍 Approaches

### 1. CNN on Mel-Spectrograms

#### Motivation
- Transform **1D audio waveforms** into **2D Mel-spectrogram images**, capturing both temporal and spectral features.  
- Enables use of **powerful computer vision architectures** like **ResNet, EfficientNet, ConvNeXt**.  
- Mel scale aligns with **human auditory perception**, improving model sensitivity to relevant frequencies.

#### Preprocessing
- Sample Rate: 32,000 Hz  
- Segment Length: matches test data segments  
- Top dB: suppress background noise  
- n_mels: 256  
- fmin/fmax: aligned with human auditory range  
- FFT parameters: n_fft, hop_length, power, normalization  
- Mel-spectrograms were converted to **3-channel RGB** images for CNN input and cached as `.npy` files.

#### Data Augmentation
- **Mixup:** weighted combination of pairs of spectrograms and labels  
- **Random Horizontal Flip:** changes time dynamics  
- **Random Erasing:** simulates partial signal dropout  

#### Model Architecture
- Backbone: **EfficientNet-B0 / ResNet** (pretrained)  
- Classifier head: replaced with fully connected linear layer  
- Input: `(256 × 256 × 3)` mel-spectrogram  
- Dropout disabled to retain full feature flow  
- Flattening: `AdaptiveAvgPool2d`  

#### Loss Function
- **BCEWithLogitsLoss** for multi-label classification  
- Aligns with **ROC-AUC evaluation metric**  

#### Training Strategy
1. **Supervised Learning**  
   - Warm-up: train classifier head only  
   - Partial Fine-tuning: unfreeze 60% backbone  
   - Full Fine-tuning: unfreeze all layers  
   - Result: **0.736 ROC-AUC**
2. **Semi-Supervised Learning (pseudo-labeling)**  
   - Pseudo-label top confident predictions from unlabeled data  
   - Mixup and other regularizations applied  
   - Result: **0.804 ROC-AUC**
3. **Model Ensembling**  
   - Combine predictions from multiple trained models  
   - Final leaderboard: **0.817 ROC-AUC**

---

### 2. Transfer Learning on Audio Embeddings

#### Motivation
- Pretrained **bird-specific embeddings** outperform generic audio models  
- Captures **frequency-temporal patterns** applicable across species and habitats  
- Lightweight classification head (linear or shallow MLP) allows efficient training

#### Workflow
1. **Environment Setup:** import modules, select dataset, test embedding model  
2. **Embedding Generation:** compute embeddings for unlabeled corpus using SurfPerch model  
3. **Query Selection:** labeled vocalizations from domain experts or CIOC dataset  
4. **Similarity Search:** use k-NN to retrieve most similar clips from corpus  
5. **Manual Annotation:** validate or discard retrieved samples  
6. **Iterative Labeling:** repeat until 20–30 samples per species  
7. **Train Classifier:** linear/MLP on collected embeddings  

> This approach supports rapid model development even with limited labeled data.  
> Due to time and GPU constraints, this method was only tested once, achieving **0.65 ROC-AUC**.  

---

## 📊 Dataset & EDA
- Highly imbalanced labels: some species with 990 samples, others only 2  
- Standard random split is infeasible  
- Used **stratified splitting** to ensure representation of all species in training and validation sets

---

## 🛠️ Tools & Libraries
- **Python**  
- **PyTorch** / **Torchvision**  
- **Librosa** for audio processing  
- **NumPy** / **Pandas** for data manipulation  
- **BirdNET**, **Google Bird Vocalization Classifier**, **EnCodec**  
- **Kaggle API** for dataset access  

---

## 📈 Results & Conclusion
- **Best performance:** Semi-supervised CNN with Mel-spectrograms and model ensembling  
- **Leaderboard score:** 0.817 ROC-AUC  
- **Final ranking:** 536th  
- Transfer-learning approach promising but limited by resources  

> This project demonstrates **practical bioacoustic classification** using CNNs, embeddings, pseudo-labeling, and ensembling, providing a strong foundation for future research on multi-species soundscape analysis.

---

## 🔗 References
1. Ghani, B., Denton, T., Kahl, S. et al. *Global birdsong embeddings enable superior transfer learning for bioacoustic classification.* Sci Rep 13, 22876 (2023). [DOI](https://doi.org/10.1038/s41598-023-49989-z)
2. BirdNET: https://birdnet.cornell.edu/  
3. Google Bird Vocalization Classifier: https://ai.googleblog.com/2023/  
