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
||| objective by placing a probabilistic model over historic data then, typically, optimizing an
||| _acquisition function_ which quantifies how useful it would be to evaluate the objective at any
||| given set of points.
module BayesianOptimization

import public Data.Stream
import Tensor

import public BayesianOptimization.Acquisition as BayesianOptimization
import public BayesianOptimization.Binary as BayesianOptimization

||| A Bayesian optimization loop as a (potentially infinite) stream of values. The values are
||| typically the observed data, and the models of that data. The loop iteratively finds new points
||| with the specified `tactic` then updates the values with these new points (assuming some
||| implicit objective function).
|||
||| @tactic The tactic which which to recommend new points. This could be optimizing an acquisition
|||   function, for example. Note this is a `Morphism`, not a function.
||| @observer A function which evaluates the optimization objective at the recommended points, then
|||   updates the values (typically data and models).
export
loop : Semigroup data_ =>
  (tactic : Binary data_ model points)
  -> (objective : points -> data_)
  -> (model_update : data_ -> model -> model)
  -> data_ -> model -> Stream (data_, model)
loop tactic objective model_update = curry $ iterate $
  \(dataset, model) => let new_dataset = objective (run tactic dataset model)
                        in (dataset <+> new_dataset, model_update new_dataset model)
