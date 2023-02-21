def generate_model(data, graph):
    ordered_nodes = [i for i in graph.topological_sort()]
    undirected_edges = graph.undirected.edges()

    beta_coefficients = dict()
    confounder_coefficients = dict()
    sigma_coefficients = dict()
    confounder_sigma_coefficients = dict()

    for i in range(len(ordered_nodes)):
        ancesters = graph.ancestors_inclusive(ordered_nodes[i])
        confounders = graph.undirected.edges(ordered_nodes[i])

        sigma_coefficients[ordered_nodes[i]] = pyro.sample(
            "sigma_{}".format(ordered_nodes[i]), dist.Uniform(0., 5.))

        betas = dict()
        if len(ancesters) > 1:
            for upstream in ancesters:
                if upstream != ordered_nodes[i]:
                    betas[upstream] = pyro.sample(
                        "b_{0}_{1}".format(ordered_nodes[i], upstream),
                        dist.Normal(0., 1.))
        beta_coefficients[ordered_nodes[i]] = betas
        if len(confounders) > 0:
            for e in confounders:
                confounder_coefficients[tuple(sorted(e))] = pyro.sample("b_{0}_{1}".format(ordered_nodes[i],
                                                                                           tuple(sorted(e))),
                                                                        dist.Normal(0., 1.))

    for e in undirected_edges:
        confounder_sigma_coefficients[tuple(sorted(e))] = pyro.sample("sigma_{}".format(tuple(sorted(e))),
                                                                      dist.Uniform(0., 5.))

    with pyro.plate("data", len(data)):
        latent_variables = dict()
        observed_variables = dict()

        for e in undirected_edges:
            latent_variables[tuple(sorted(e))] = pyro.sample("latent_{}".format(tuple(sorted(e))),
                                                             dist.Normal(0, confounder_sigma_coefficients[
                                                                 tuple(sorted(e))]))

        for i in range(len(ordered_nodes)):
            ancesters = graph.ancestors_inclusive(ordered_nodes[i])
            confounders = graph.undirected.edges(ordered_nodes[i])
            temp_mean = 0
            if len(confounders) > 0:
                for e in confounders:
                    temp_mean += confounder_coefficients[tuple(sorted(e))] * latent_variables[tuple(sorted(e))]

            if len(ancesters) > 1:
                for upstream in ancesters:
                    if upstream != ordered_nodes[i]:
                        temp_mean += beta_coefficients[ordered_nodes[i]][upstream] * observed_variables[upstream]

            observed_variables[ordered_nodes[i]] = pyro.sample("obs_{}".format(ordered_nodes[i]),
                                                               dist.Normal(temp_mean,
                                                                           sigma_coefficients[ordered_nodes[i]]),
                                                               obs=torch.tensor(
                                                                   data.loc[:, str(ordered_nodes[i])].values))


def generate_guide(data, graph):
    ordered_nodes = [i for i in graph.topological_sort()]
    undirected_edges = graph.undirected.edges()

    confounder_sigma_coefficients = dict()

    for i in range(len(ordered_nodes)):
        ancesters = graph.ancestors_inclusive(ordered_nodes[i])
        confounders = graph.undirected.edges(ordered_nodes[i])

        temp_sigma_loc = pyro.param("sigma_loc_{}".format(ordered_nodes[i]), torch.tensor(1.),
                                    constraint=constraints.positive)
        pyro.sample("sigma_{}".format(ordered_nodes[i]), dist.Normal(temp_sigma_loc, torch.tensor(0.05)))

        if len(ancesters) > 1:
            for upstream in ancesters:
                if upstream != ordered_nodes[i]:
                    temp_b_loc = pyro.param("b_loc_{0}_{1}".format(ordered_nodes[i], upstream), torch.tensor(0.))
                    pyro.sample("b_{0}_{1}".format(ordered_nodes[i], upstream), dist.Normal(temp_b_loc, .1))
        if len(confounders) > 0:
            for e in confounders:
                temp_b_loc = pyro.param("b_loc_{0}_{1}".format(ordered_nodes[i],
                                                               tuple(sorted(e))),
                                        torch.tensor(0.))
                pyro.sample("b_{0}_{1}".format(ordered_nodes[i], tuple(sorted(e))),
                            dist.Normal(temp_b_loc, 1.))

    for e in undirected_edges:
        temp_sigma_loc = pyro.param("sigma_loc_{}".format(tuple(sorted(e))), torch.tensor(1.),
                                    constraint=constraints.positive)
        confounder_sigma_coefficients[tuple(sorted(e))] = pyro.sample("sigma_{}".format(tuple(sorted(e))),
                                                                      dist.Normal(temp_sigma_loc, torch.tensor(0.05)))

    with pyro.plate("data", len(data)):

        for e in undirected_edges:
            pyro.sample("latent_{}".format(tuple(sorted(e))),
                        dist.Normal(0, confounder_sigma_coefficients[tuple(sorted(e))]))


def inf_model(data, graph):
    ordered_nodes = [i for i in graph.topological_sort()]
    undirected_edges = graph.undirected.edges()

    beta_coefficients = dict()
    confounder_coefficients = dict()
    sigma_coefficients = dict()
    confounder_sigma_coefficients = dict()

    for i in range(len(ordered_nodes)):
        ancesters = graph.ancestors_inclusive(ordered_nodes[i])
        confounders = graph.undirected.edges(ordered_nodes[i])

        sigma_coefficients[ordered_nodes[i]] = pyro.get_param_store()[
            "sigma_loc_{}".format(ordered_nodes[i])].detach()

        betas = dict()
        if len(ancesters) > 1:
            for upstream in ancesters:
                if upstream != ordered_nodes[i]:
                    betas[upstream] = pyro.get_param_store()[
                        "b_loc_{0}_{1}".format(ordered_nodes[i], upstream)].detach()
        beta_coefficients[ordered_nodes[i]] = betas
        if len(confounders) > 0:
            for e in confounders:
                confounder_coefficients[tuple(sorted(e))] = pyro.get_param_store()[
                    "b_loc_{0}_{1}".format(ordered_nodes[i], tuple(sorted(e)))].detach()

    for e in undirected_edges:
        confounder_sigma_coefficients[tuple(sorted(e))] = pyro.get_param_store()[
            "sigma_loc_{}".format(tuple(sorted(e)))].detach()

    latent_variables = dict()
    observed_variables = dict()

    for e in undirected_edges:
        latent_variables[tuple(sorted(e))] = pyro.sample("latent_{}".format(tuple(sorted(e))),
                                                         dist.Normal(0,
                                                                     confounder_sigma_coefficients[tuple(sorted(e))]))

    for i in range(len(ordered_nodes)):
        ancesters = graph.ancestors_inclusive(ordered_nodes[i])
        confounders = graph.undirected.edges(ordered_nodes[i])
        temp_mean = 0
        if len(confounders) > 0:
            for e in confounders:
                temp_mean += confounder_coefficients[tuple(sorted(e))] * latent_variables[tuple(sorted(e))]

        if len(ancesters) > 1:
            for upstream in ancesters:
                if upstream != ordered_nodes[i]:
                    temp_mean += beta_coefficients[ordered_nodes[i]][upstream] * observed_variables[upstream]

        observed_variables[ordered_nodes[i]] = pyro.sample("obs_{}".format(ordered_nodes[i]),
                                                           dist.Normal(temp_mean,
                                                                       sigma_coefficients[ordered_nodes[i]]))
    return observed_variables