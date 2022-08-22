
from indra.tools.gene_network import GeneNetwork
from indra import literature
from indra.sources import reach
from indra.tools import assemble_corpus as ac
from indra.assemblers.pybel.assembler import PybelAssembler

import time

class BuildNetwork:

    def __init__(self, gene_list):

        self.gene_list = gene_list

        ## Defined later
        self.network_statements = None
        self.pybel_model = None

    def assemble_genes(self, reach_server="remote_server"):

        if reach_server in ["local", "remote_server"]:
            ## Collect known statements
            gn = GeneNetwork(list(gene_list))  # , basename="cache"

            biopax_stmts = gn.get_biopax_stmts()  ## PathwayCommons DB
            bel_stmts = gn.get_bel_stmts()  ## BEL Large Corpus

            ## Get statements from pubmed literature
            pmids = literature.pubmed_client.get_ids_for_gene('MTA2')

            ## Get all lit
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
            literature_stmts = list()
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
            stmts = ac.map_sequence(stmts)
            stmts = ac.run_preassembly(stmts)

            self.network_statements = stmts
        else:
            print("reach_server must be one of local or remote_server")

    def assemble_pybel(self):

        if self.network_statements is not None:

            raw_pybel_graph = PybelAssembler(stmts)
            pybel_model = raw_pybel_graph.make_model()

            self.pybel_model = pybel_model

        else:
            print("No network statements. Please run assemble_genes() function first.")