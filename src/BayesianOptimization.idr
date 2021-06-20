{--
Copyright 2021 Joel Berkeley

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--}
||| This module defines functionality for Bayesian optimization, the data-efficient optimization of
||| objective functions. Bayesian optimization recommends new points at which to query your
||| objective by placing a probabilistic model over historic data then optimizing an "acquisition
||| function" which quantifies how useful it would be to evaluate the objective at any given set of
||| points.
module BayesianOptimization

import public BayesianOptimization.Acquisition as BayesianOptimization
import public BayesianOptimization.Domain as BayesianOptimization
import public BayesianOptimization.Optimize as BayesianOptimization
import public BayesianOptimization.Util as BayesianOptimization
