
from single_cell_scm.causal_graph import CausalGraph
import pandas as pd
import networkx as nx
import itertools
import pickle

from protmapper import uniprot_client

pd.options.mode.chained_assignment = None  # default='warn'


def main():
    net = pd.read_csv("../data/net_no_cycles.tsv", sep="\t")
    causal_net = CausalGraph(net)
    causal_net.indra_to_dag(["IncreaseAmount", "DecreaseAmount"],
                            drop_bidirect_randomly=True)

    msstats_mel_summarized = pd.read_csv("../data/MSstats_summarized.csv")
    annotation = pd.read_csv("../data/meta.csv", index_col="Unnamed: 0")

    ## Add in info for join
    annotation.loc[:, "channel_match"] = "channel" + annotation.loc[:, "channel"].str.split(".", expand=True)[2]
    msstats_mel_summarized = pd.merge(msstats_mel_summarized, annotation, how="left",
                                      left_on=["Mixture", "Channel", "Condition"],
                                      right_on=["digest", "channel_match", "celltype"])

    msstats_mel_summarized = msstats_mel_summarized.loc[~msstats_mel_summarized["Protein"].str.contains(";")]
    msstats_mel_summarized.loc[:, "Protein"] = msstats_mel_summarized.loc[:, "Protein"].str.split("|", expand=True)[1]

    input_data = msstats_mel_summarized.pivot(columns="Protein", values="Abundance", index="id")
    input_data.columns = [uniprot_client.get_gene_name(x) for x in input_data.columns]

    keep = (input_data.isna().sum() / (input_data.count() + input_data.isna().sum()))[
        (input_data.isna().sum() / (input_data.count() + input_data.isna().sum())) < .34].index

    parsed_input = input_data.loc[:, keep].fillna(input_data.loc[:, keep].mean())

    causal_net.identify_latent_nodes(parsed_input)
    causal_net.create_latent_graph()
    causal_net.find_all_identifiable_pairs()

    with open('leduc_graph.pkl', 'wb') as f:
        # dump the object to the file using pickle
        pickle.dump(causal_net, f)

if __name__ == "__main__":
    main()

# from single_cell_scm.pyro_scm.build_network import BuildNetwork
# import pickle
# import os
# import pandas as pd
# import numpy as np
#
# os.environ['ELSEVIER_API_KEY'] = '270acf549bfed5a87046fb9d91b02ead'
#
# def learn_network(mapping):
#     genes = mapping.loc[:, "Gene"].values
#     genes = np.append(genes, ["BRAF", "NRAS"])
#     model = BuildNetwork(genes)
#     model.assemble_genes()
#     model.assemble_pybel()
#     model.assemble_pandas_df(mapping)
#     return model
#
# def main():
#
#     protein_gene_mapping = pd.read_csv(
#         "/home/kohler.d/applications_project/single_cell_scm/data/protein_gene_mapping.tsv",
#         sep="\t", header=0, names=["Protein", "Gene"])
#
#     model = learn_network(protein_gene_mapping)
#
#     with open("/scratch/kohler.d/code_output/Leduc_model.pkl", 'wb') as handle:
#         pickle.dump(model, handle, pickle.HIGHEST_PROTOCOL)
#
#
# if __name__ == "__main__":
#     main()
