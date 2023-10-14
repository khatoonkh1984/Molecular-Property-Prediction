# Molecular-Property-Prediction
Graph classification tasks for datasets such as BBBP and Protein

In this repository, we employ graph classification tasks for datasets such as BBBP and Protein.

- The BBBP dataset comes from a recent study 52 on the modeling and prediction of barrier permeability. This dataset includes binary labels for over 2000 compounds on their permeability properties, where the goal is to predict whether a compound can cross the blood-brain barrier or not based on its molecular features. The BBBP dataset is stored as a CSV file, where each row represents a compound and each column represents an attribute. The first column is the name of the compound, the second column is a binary label indicating whether the compound can penetrate the blood-brain barrier or not, and the third column is a SMILES string representing the molecular structure of the compound. The BBBP dataset is part of the MoleculeNet benchmark, which is a collection of datasets for testing machine learning methods of molecular properties. For training this dataset, we use a bunch of GCN and linear layers.

- The protein dataset from TUDataset is a collection of graphs that represent the 3D structure of proteins. The dataset contains 1,113 proteins, of which 663 enzymes and 450 non-enzymes. The task is to classify a protein as an enzyme or a non-enzyme based on its graph representation. The nodes in the graphs are amino acids. Also, the edges are spatial proximity. The nodes have three features: the amino acid type, the secondary structure, and the relative solvent accessibility. The model for training this dataset is from this source code:
 https://github.com/cszhangzhen/HGP-SL.


