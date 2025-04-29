import pickle
import pandas as pd

data = pickle.load(open("datasets/gt1H/raw/graph_conv_many_nuc_pipeline.data.1H.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.1.mol_dict.pickle", "rb"))
# print(data)

train_df = data['train_df']

for idx, row in train_df.iterrows():
    rdmol = row['rdmol']
    atoms_symbols = [atom.GetSymbol() for atom in rdmol.GetAtoms()]
    atoms = rdmol.GetAtoms()
    value = row['value'][0] # value is a one element list
    for k, v in value.items():
        print(atoms[k].GetNeighbors()[0].GetSymbol(), v)


# the dataset does have the shift of Non-Carbons atoms