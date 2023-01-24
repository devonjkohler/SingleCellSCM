
from sklearn.linear_model import LinearRegression
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

class SCM:
    def __init__(self, observational_data, start_vars, x_variables, y_variables):
        self.observational_data = observational_data
        self.start_vars = start_vars
        self.X_variables = x_variables
        self.y_variables = y_variables
        self.num_edges = len(y_variables)

        ## Train edges
        coefs = dict()
        intercepts = dict()
        scores = dict()
        for i in range(len(y_variables)):
            ## Test if X is multiple or singular
            if isinstance(x_variables[i], list):
                x = observational_data.loc[:, x_variables[i]]
            else:
                x = observational_data.loc[:, x_variables[i]].values.reshape(-1, 1)

            y = observational_data.loc[:, y_variables[i]]
            ln = LinearRegression()
            ln.fit(x, y)
            scores["X:{0}, y:{1}".format(x_variables[i], y_variables[i])] = ln.score(x, y)
            coefs["y_{0}".format(y_variables[i])] = ln.coef_
            intercepts["y_{0}".format(y_variables[i])] = ln.intercept_

        self.coefficients = coefs
        self.intercepts = intercepts
        self.edge_scores = scores

        ## Generate functions for model
        for i in range(len(y_variables)):
            x_model_string = ' + '.join([str(b) + "*" + x for b,x in zip(list(coefs.values())[i],x_variables[i])])
            model_string = str(list(intercepts.values())[i]) + " + " + x_model_string + " + N"
            ## TODO: using exec is dumb.. just add created functions to a list..
            exec("""def edge_{0}({1}, N):
                        return {2}""".format(str(i), ", ".join(x_variables[i]), model_string), globals())

        def model(noise):

            noise_samples = dict()
            for i in range(len(noise)):
                noise_samples[list(noise)[i]] = pyro.sample(list(noise)[i], list(noise.values())[i])

            ## Time0
            sampled_vars = dict()
            for i in range(len(start_vars)):
                sampled_vars[start_vars[i]] = pyro.sample(
                    start_vars[i], dist.Delta(noise_samples["N_{0}".format(str(start_vars[i]))])
                )

            for i in range(len(y_variables)):

                current_x_vars = copy.copy(x_variables[i])
                if not isinstance(current_x_vars, list):
                    current_x_vars = [current_x_vars]
                full_x = [sampled_vars[x] for x in current_x_vars]
                full_x.append(noise_samples["N_{0}".format(str(y_variables[i]))])

                ## TODO: dont need to call global() if swap to list of functions
                sampled_vars[y_variables[i]] = pyro.sample(
                    y_variables[i], dist.Delta(globals()["edge_{}".format(str(i))](*full_x))
                )

            return sampled_vars, noise_samples

        Spike = partial(dist.Normal, scale=torch.tensor(1.0))

        def noisy_model(noise):

            noise_samples = dict()
            for i in range(len(noise)):
                noise_samples[list(noise)[i]] = pyro.sample(list(noise)[i], list(noise.values())[i])

            ## Time0
            sampled_vars = dict()
            for i in range(len(start_vars)):
                sampled_vars[start_vars[i]] = pyro.sample(
                    start_vars[i], Spike(noise_samples["N_{0}".format(str(start_vars[i]))])
                )

            for y in range(len(y_variables)):

                current_x_vars = copy.copy(x_variables[y])
                if not isinstance(current_x_vars, list):
                    current_x_vars = [current_x_vars]
                full_x = [sampled_vars[x] for x in current_x_vars]
                full_x.append(noise_samples["N_{0}".format(str(y_variables[i]))])

                sampled_vars[y_variables[y]] = pyro.sample(
                    y_variables[y], Spike(globals()["edge_{}".format(str(y))](*full_x))
                )

            return sampled_vars, noise_samples

        self.model = model
        self.noisy_model = noisy_model

    def infer(self, model, noise):
        return Importance(model, num_samples=1000).run(noise)

    def update_noise_importance(self, observed_steady_state, initial_noise):
        observation_model = pyro.condition(self.noisy_model, observed_steady_state)
        posterior = self.infer(observation_model, initial_noise)
        updated_noise = {
            k: EmpiricalMarginal(posterior, sites=k)
            for k in initial_noise.keys()
        }
        return updated_noise

    def update_noise_svi(self, observed_steady_state, initial_noise):
        def guide(noise):
            noise_terms = list(noise.keys())
            noise_means = [i.mean for i in noise.values()]
            noise_std = [i.stddev for i in noise.values()]
            mu_constraints = [constraints.interval(i - i *.5, i + i *.5) if i != 0
                              else constraints.interval(-1., 1.) for i in noise_means]
            sigma_constraints = [constraints.interval(.0001, i + i) for i in noise_std]
            mu = {
                noise_terms[i] : pyro.param(
                    '{}_mu'.format(noise_terms[i]),
                    torch.tensor(noise_means[i]),
                    constraint=mu_constraints[i]
                ) for i in range(len(noise_terms))
            }
            sigma = {
                noise_terms[i] : pyro.param(
                    '{}_sigma'.format(noise_terms[i]),
                    torch.tensor(noise_std[i]),
                    constraint=sigma_constraints[i]
                ) for i in range(len(noise_terms))
            }
            for noise in noise_terms:
                pyro.sample(noise, pyro.distributions.Normal(mu[noise], sigma[noise]))

        observation_model = pyro.condition(self.noisy_model, observed_steady_state)
        pyro.clear_param_store()

        svi = SVI(
            model=observation_model,
            guide=guide,
            optim=SGD({"lr": 0.001, "momentum": 0.1}),
            loss=Trace_ELBO()
        )

        losses = list()
        num_steps = 250
        samples = defaultdict(list)
        for t in range(num_steps):
            losses.append(svi.step(initial_noise))
            # print(losses[-1])
            for noise in initial_noise.keys():
                mu = '{}_mu'.format(noise)
                sigma = '{}_sigma'.format(noise)
                samples[mu].append(pyro.param(mu).item())
                samples[sigma].append(pyro.param(sigma).item())
        means = {k: np.mean(v) for k, v in samples.items()}

        updated_noise = dict()
        noise_terms = list(initial_noise.keys())
        for i in range(len(noise_terms)):
            updated_noise[noise_terms[i]] = pyro.distributions.Normal(means['{0}_mu'.format(noise_terms[i])],
                                                                      means['{0}_sigma'.format(noise_terms[i])])

        return updated_noise