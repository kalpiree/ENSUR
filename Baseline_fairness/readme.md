## Prerequisites

- **Python**
- **PyTorch**

## Instructions

### Datasets

Use the relevant datasets from the `dataset` folder for each step.

### Baselines

#### Baselines 1 and 2
1. **Pre-training NCF model**:
   - Run: `run_preTrainNCF.py`
   - Description: Pre-trains the NCF model. The pre-trained model will be saved in the `trained-models` folder.

2. **Debiasing user embeddings**:
   - Run: `run_debiasing_userEmbeddings.py`
   - Description: Debiases the user embeddings, which will be saved in the `results` folder.

3. **Fine-tuning NFCF with fairness interventions**:
   - Run: `run_nfcf_career_recommend.py`
   - Description: Fine-tunes the NFCF model with fairness interventions. Evaluation results will be saved in the `results` folder.

#### Baselines 3 and 4
- Run: `ufr.py`
- Description: Executes re-ranking based fairness methods.
