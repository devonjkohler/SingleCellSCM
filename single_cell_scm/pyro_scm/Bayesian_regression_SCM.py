import pyro
import pyro.distributions as dist
from pyro.optim import SGD
from pyro.infer import Importance, EmpiricalMarginal, SVI, Trace_ELBO
import torch
import torch.distributions.constraints as constraints

import numpy as np
from functools import partial
from collections import defaultdict

import copy


class BayesianSCM:
    def __init__(self, observational_data, x_variables, y_variables):
        self.observational_data = observational_data
        self.X_variables = x_variables
        self.y_variables = y_variables
        self.num_edges = len(y_variables)

        self.losses = None
        self.learned_params = None

        def model(param_priors, noise_priors, obs_data=None):

            ## Create Latent Variables
            param_samples = dict()
            for i in range(len(param_priors)):
                param_samples[list(param_priors)[i]] = pyro.param(list(param_priors)[i],
                                                                  lambda: torch.randn(()))
            ## Sample noise variables
            noise_samples = dict()
            for i in range(len(noise_priors)):
                noise_samples[list(noise_priors)[i]] = pyro.sample(list(noise_priors)[i],
                                                                   list(noise_priors.values())[i])
                # noise_samples[list(noise_priors)[i]] = pyro.sample(list(noise_priors)[i],
                #                                                    lambda: torch.tensor(1.))

            ## Time0
            sampled_vars = dict()
            for i in range(len(y_variables)):
                current_x_vars = copy.copy(x_variables[i])  ## Get independent vars
                ## Make sure independent vars are in a list
                if not isinstance(current_x_vars, list):
                    current_x_vars = [current_x_vars]

                ## construct regression
                if len(current_x_vars) > 0:
                    variables = torch.stack([
                        param_samples["b_{0}_{1}".format(str(y_variables[i]), x)] * sampled_vars[x]
                        for x in current_x_vars])
                    variables = variables.sum(axis=0)
                else:
                    variables = torch.tensor([0.])
                mean = variables + torch.tensor(param_samples["a_{0}".format(str(y_variables[i]))])

                if obs_data is not None:
                    temp_obs = torch.tensor(obs_data[str(y_variables[i])])
                    plate_len = len(temp_obs)
                else:
                    temp_obs = None
                    plate_len = 1
                with pyro.plate("data{0}".format(y_variables[i]), plate_len):
                    sampled_vars[y_variables[i]] = pyro.sample(
                        y_variables[i], dist.Normal(mean, noise_samples["N_{0}".format(str(y_variables[i]))]),
                        obs=temp_obs
                    )

            return sampled_vars

        self.model = model

    # def infer(self, model, noise):
    #     return Importance(model, num_samples=1000).run(noise)
    #
    # def update_noise_importance(self, observed_steady_state, initial_noise):
    #     observation_model = pyro.condition(self.model, observed_steady_state)
    #     posterior = self.infer(observation_model, initial_noise)
    #     updated_noise = {
    #         k: EmpiricalMarginal(posterior, sites=k)
    #         for k in initial_noise.keys()
    #     }
    #     return updated_noise
    #
    # def update_noise_svi(self, observed_steady_state, initial_noise):
    #     # def guide(noise):
    #     #     noise_terms = list(noise.keys())
    #     #     noise_means = [i.mean for i in noise.values()]
    #     #     noise_std = [i.stddev for i in noise.values()]
    #     #     mu_constraints = [constraints.interval(i - i *.5, i + i *.5) if i != 0
    #     #                       else constraints.interval(-1., 1.) for i in noise_means]
    #     #     sigma_constraints = [constraints.interval(.0001, i + i) for i in noise_std]
    #     #     mu = {
    #     #         noise_terms[i] : pyro.param(
    #     #             '{}_mu'.format(noise_terms[i]),
    #     #             torch.tensor(noise_means[i]),
    #     #             constraint=mu_constraints[i]
    #     #         ) for i in range(len(noise_terms))
    #     #     }
    #     #     sigma = {
    #     #         noise_terms[i] : pyro.param(
    #     #             '{}_sigma'.format(noise_terms[i]),
    #     #             torch.tensor(noise_std[i]),
    #     #             constraint=sigma_constraints[i]
    #     #         ) for i in range(len(noise_terms))
    #     #     }
    #     #     for noise in noise_terms:
    #     #         pyro.sample(noise, pyro.distributions.Normal(mu[noise], sigma[noise]))
    #
    #     guide = pyro.infer.autoguide.AutoNormal(self.model)
    #
    #     observation_model = pyro.condition(self.model, observed_steady_state)
    #     pyro.clear_param_store()
    #
    #     svi = SVI(
    #         model=observation_model,
    #         guide=guide,
    #         optim=SGD({"lr": 0.001, "momentum": 0.1}),
    #         loss=Trace_ELBO()
    #     )
    #
    #     losses = list()
    #     num_steps = 250
    #     samples = defaultdict(list)
    #     for t in range(num_steps):
    #         losses.append(svi.step(initial_noise))
    #         # print(losses[-1])
    #         for noise in initial_noise.keys():
    #             mu = '{}_mu'.format(noise)
    #             sigma = '{}_sigma'.format(noise)
    #             samples[mu].append(pyro.param(mu).item())
    #             samples[sigma].append(pyro.param(sigma).item())
    #     means = {k: np.mean(v) for k, v in samples.items()}
    #
    #     updated_noise = dict()
    #     noise_terms = list(initial_noise.keys())
    #     for i in range(len(noise_terms)):
    #         updated_noise[noise_terms[i]] = pyro.distributions.Normal(means['{0}_mu'.format(noise_terms[i])],
    #                                                                   means['{0}_sigma'.format(noise_terms[i])])
    #
    #     return updated_noise

    def learn_parameters(self, param_priors, noise_priors, obs_data=None):

        def guide(param_priors, noise_priors, obs_data=None):
            noise_terms = list(noise_priors.keys())
            mu = {
                noise_terms[i]: pyro.param(
                    '{}_mu'.format(noise_terms[i]),
                    torch.tensor(.001),
                    constraint=constraints.positive
                ) for i in range(len(noise_terms))
            }
            sigma = {
                noise_terms[i]: pyro.param(
                    '{}_sigma'.format(noise_terms[i]),
                    torch.tensor(.0001),
                    constraint=constraints.positive
                ) for i in range(len(noise_terms))
            }
            for noise in noise_terms:
                pyro.sample(noise, pyro.distributions.Normal(mu[noise], sigma[noise]))

        pyro.clear_param_store()
        auto_guide = pyro.infer.autoguide.AutoNormal(self.model)
        adam = pyro.optim.Adam({"lr": 0.01})
        elbo = pyro.infer.Trace_ELBO()
        svi = pyro.infer.SVI(self.model, guide, adam, elbo)

        losses = []
        for step in range(10000):
            loss = svi.step(param_priors, noise_priors, obs_data)
            losses.append(loss)
            if step % 100 == 0:
                print("Elbo loss: {}".format(loss))

        self.losses = losses
        self.learned_params = pyro.get_param_store().items()