
from indra.tools.gene_network import GeneNetwork
from indra import literature
from indra.sources import reach
from indra.tools import assemble_corpus as ac
from indra.assemblers.pybel.assembler import PybelAssembler
import pybel_jupyter
from pybel.struct.mutation.deletion import remove_filtered_nodes, remove_biological_processes, \
    remove_isolated_list_abundances, remove_non_causal_edges, remove_pathologies
from pybel.struct.mutation.utils import remove_isolated_nodes

import pandas as pd
import numpy as np
import time

class BuildNetwork:

    def __init__(self, gene_list):

        self.gene_list = gene_list

        ## Defined later
        self.network_statements = None
        self.pybel_model = None
        self.pandas_graph = pd.DataFrame()

    def assemble_genes(self, PathwayCommons=True, Pathway_query = "pathsbetween",
                       BEL=True, search_lit=False, reach_server="remote_server"):

        if reach_server in ["local", "remote_server"]:
            ## Collect known statements
            gn = GeneNetwork(list(self.gene_list), basename="cache")  # ,

            biopax_stmts, bel_stmts, literature_stmts = list(), list(), list()

            if PathwayCommons:
                biopax_stmts = gn.get_biopax_stmts(query=Pathway_query)  ## PathwayCommons DB
            if BEL:
                bel_stmts = gn.get_bel_stmts()  ## BEL Large Corpus

            ## Get statements from pubmed literature
            ## TODO: Need to get local REACH working
            if search_lit:
                pmids = literature.pubmed_client.get_ids_for_gene('MTA2') ## TODO: Maybe need for loop here

                # Get all lit
                paper_contents = dict()
                counter = 0
                for pmid in pmids:
                    content, content_type = literature.get_full_text(pmid, 'pmid')
                    if content_type == 'abstract':
                        paper_contents[pmid] = content
                        counter += 1
                    ## Use to speed up process
                    # if counter == 10:
                    #     break

                ## Extract statements from lit
                for pmid, content in paper_contents.items():
                    time.sleep(1)
                    if reach_server == "local":
                        rp = reach.process_text(content, url=reach.local_text_url)
                    elif reach_server == "remote_server":
                        rp = reach.process_text(content)
                    if rp is not None:
                        literature_stmts += rp.statements

            ## Combine all statements and run pre-assembly
            stmts = biopax_stmts + bel_stmts + literature_stmts

            stmts = ac.map_grounding(stmts)
            ## Some indra error on ActiveForm statements idk...
            stmts = ac.map_sequence([x for x in stmts if "ActiveForm" not in str(x)])
            stmts = ac.run_preassembly(stmts)

            self.network_statements = stmts
        else:
            raise ValueError("reach_server must be one of local or remote_server")

    def assemble_pybel(self, save_html=False):

        if self.network_statements is not None:

            raw_pybel_graph = PybelAssembler(self.network_statements)
            pybel_model = raw_pybel_graph.make_model()

            self.pybel_model = pybel_model

            if save_html:
                pybel_jupyter.to_html_file(self.pybel_model, "pybel_graph.html")

        else:
            raise ValueError("No network statements. Please run assemble_genes() function first.")

    def assemble_pandas_df(self, protein_gene_mapping, filter_edges=True, save_filtered_graph=False, keep_self_cycles=False):

        def clean_pandas_data(df):
            ## Clean up BEL statements
            df = df[(~df["From"].str.contains("a\(")) & (~df["To"].str.contains("a\(")) &
                    (~df["To"].str.contains("var\("))].reset_index(drop=True)
            df.loc[:, "From"] = df.loc[:, "From"].apply(lambda x: x.split("! ")[1].replace(")", ""))
            df.loc[:, "To"] = df.loc[:, "To"].apply(lambda x: x.split("! ")[1].replace(")", ""))
            return df

        def add_proteins(df, protein_df):
            joined_df = pd.merge(df, protein_df, left_on="From", right_on="Gene", how="left")
            joined_df.loc[:, "From_Gene"] = joined_df.loc[:, "From"]
            joined_df.loc[:, "From"] = joined_df.loc[:, "Protein"]
            joined_df.drop(columns=["Protein", "Gene"], inplace=True)
            joined_df = pd.merge(joined_df, protein_df, left_on="To", right_on="Gene", how="left")
            joined_df.loc[:, "To_Gene"] = joined_df.loc[:, "To"]
            joined_df.loc[:, "To"] = joined_df.loc[:, "Protein"]
            joined_df.loc[:, "To_latent"] = joined_df.loc[:, "To"].isna()
            joined_df.loc[:, "From_latent"] = joined_df.loc[:, "From"].isna()
            joined_df.drop(columns=["Protein", "Gene"], inplace=True)
            return joined_df

        if self.pybel_model is not None:

            if filter_edges:
                ## Run filtering functions
                remove_biological_processes(self.pybel_model)
                remove_isolated_list_abundances(self.pybel_model)
                remove_non_causal_edges(self.pybel_model)
                remove_pathologies(self.pybel_model)
                remove_isolated_nodes(self.pybel_model)

                if save_filtered_graph:
                    pybel_jupyter.to_html_file(self.pybel_model, "filtered_pybel_graph.html")

            ## Remove some BEL statement stuff
            cleaned_edges = [str(x).replace("(<BEL ", "").replace(" <BEL ", "").replace(">", "").split(",")[:2]
                             for x in self.pybel_model.edges.keys()]

            pd_graph = pd.DataFrame(data=np.array(cleaned_edges), columns=["From", "To"])
            pd_graph = clean_pandas_data(pd_graph)

            ## Add back in protein names
            final_graph = add_proteins(pd_graph, protein_gene_mapping)
            final_graph.drop_duplicates(inplace=True)

            if not keep_self_cycles:
                final_graph = final_graph[final_graph["From"] != final_graph["To"]]

            ## Add gene into protein name and where missing
            final_graph.loc[:, "From"] = np.where(final_graph.loc[:, "From"].isna(), final_graph.loc[:, "From_Gene"],
                                                  final_graph.loc[:, "From"] + "_" + final_graph.loc[:, "From_Gene"])
            final_graph.loc[:, "To"] = np.where(final_graph.loc[:, "To"].isna(), final_graph.loc[:, "To_Gene"],
                                                final_graph.loc[:, "To"] + "_" + final_graph.loc[:, "To_Gene"])

            self.pandas_graph = final_graph

        else:
            raise RuntimeError("Method requires PyBEL graph. Please run assemble_pybel() first.")
