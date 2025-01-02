
# NEGCDTI
NEGCDTI: Neighborhood-Enhanced Graph Contrastive Learning for Drug-Target Interaction Prediction with Hierarchical Representation

##### Dependencies：

- python == 3.8.18
- pytorch == 1.12.1
- torchvision == 0.13.1
- cudatoolkit == 11.3.1
- numpy == 1.24.3
- pandas == 2.0.3
- matplotlib == 3.7.2
- scikit-learn == 1.3.2
- scipy == 1.10.1
- biotite == 0.33.0
- rdkit == 2023.9.1
- timm == 0.9.16

##### Using

1. `Data` stores four datasets.
2. `smile_to_features.py` generates chemical text information. 
3. `smiles_k_gram.py` lets the chemical text be divided into words according to the k-gram method. 
4. `protein_k_gram.py` lets the protein sequences be divided into words according to the k-gram method. 
5. `cluster.py` stores the neighborhood-enhanced graph contrastive learning algorithm.
6. `main.py` trains NEGCDTI model.

##### Training

If you use the data we provide, you can run main.py directly.

For a new dataset, you need to prepare the following files:
1. drugs.xlsx: This file stores the SMILES string information of the drugs.
2. targets.xlsx: This file stores the Fasta sequence information of the targets.
3. dti_mat.xlsx: This file stores the interaction information between drugs and targets.
4. drug_affinity_mat.txt and target_affinity_mat.txt: These files should be generated using the method described in our paper. They store the similarity information 
   of drugs and targets, respectively.
Next, you need to run the following scripts:
1. Run smile_to_features.py to extract chemical textual features from the drug SMILES sequences.
2. Run smiles_k_gram.py to obtain the drug representations.
3. Run protein_k_gram.py to obtain the target representations.
4. Run main.py to train the NEGCDTI model.
