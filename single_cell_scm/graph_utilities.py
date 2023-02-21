
import pandas as pd
import numpy as np

def drop_bidirect(df):

    ## Find bidirected columns
    df.loc[:, "restart_A"] = np.minimum(df.loc[:, "From"], df.loc[:, "To"])
    df.loc[:, "restart_B"] = np.maximum(df.loc[:, "From"], df.loc[:, "To"])
    bidirect_count = (df.loc[:, "restart_A"] + "_" + df.loc[:, "restart_B"]).value_counts()
    bidirect = pd.DataFrame(bidirect_count[bidirect_count > 1].index)[0].str.split("_", expand = True)
    bidirect.columns = ["Node1", "Node2"]

    for i in range(len(bidirect)):
        sample = np.random.binomial(1, .5) ## sample which direction to remove
        if sample == 0:
            other = 1
        else:
            other = 0

        ## Remove sampled edge
        from_node = bidirect.iloc[i, sample]
        to_node = bidirect.iloc[i, other]
        df = df[-((df["From"] == from_node) & (df["To"] == to_node))]

    return df.loc[:, ["From", "To"]].reset_index(drop=True)