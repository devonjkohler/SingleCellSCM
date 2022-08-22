
from single_cell_scm.build_network import BuildNetwork
import pickle
import os
import pandas as pd

os.environ['ELSEVIER_API_KEY'] = '270acf549bfed5a87046fb9d91b02ead'

def main():

    # protein_data = pd.read_csv("../data/plexDIA_pSCoPE_integrated.csv")
    # protein_data.rename(columns={'Unnamed: 0': 'Protein'}, inplace=True)
    # proteins = protein_data["Protein"].unique()

    protein_gene_mapping = pd.read_csv(
        "/home/kohler.d/applications_project/single_cell_scm/data/protein_gene_mapping.tsv",
        sep="\t", header=0, names=["Protein", "Gene"])

    model = BuildNetwork(protein_gene_mapping.loc[:, "Gene"].values)
    model.assemble_genes(reach_server="remote_server")
    model.assemble_pybel()

    with open("/scratch/kohler.d/code_output/Leduc_model.pkl", 'wb') as handle:
        pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()