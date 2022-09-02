
from single_cell_scm.build_network import BuildNetwork
import pickle
import os
import pandas as pd
import numpy as np

os.environ['ELSEVIER_API_KEY'] = '270acf549bfed5a87046fb9d91b02ead'

def learn_network(mapping):
    genes = mapping.loc[:, "Gene"].values
    genes = np.append(genes, ["BRAF", "NRAS"])
    model = BuildNetwork(genes)
    model.assemble_genes(BEL=False, reach_server="remote_server")
    model.assemble_pybel()
    model.assemble_pandas_df(mapping)
    return model

def main():

    protein_gene_mapping = pd.read_csv(
        "/home/kohler.d/applications_project/single_cell_scm/data/protein_gene_mapping.tsv",
        sep="\t", header=0, names=["Protein", "Gene"])

    model = learn_network(protein_gene_mapping)

    with open("/scratch/kohler.d/code_output/Leduc_model.pkl", 'wb') as handle:
        pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
