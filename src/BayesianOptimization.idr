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
||| objective by placing a probabilistic model over known data then optimizing an "acquisition
||| function" which quantifies how useful it would be to evaluate the objective at any given set of
||| points.
module BayesianOptimization

import Tensor
import Data.Vect
import Optimize
import Distribution

||| A `ProbabilisticModel` is a mapping from a feature space to a probability distribution over
||| a target space.
interface (Distribution samples targets dist) => ProbabilisticModel (0 features : Shape) (0 targets : Shape) dist model where
  ||| Return the probability distribution over the target space at the specified points in the
  ||| feature space, given the model.
  predict : model -> Tensor (samples :: features) Double -> dist

interface Domain where

||| An `Acquisition` function quantifies how useful it would be to query the objective at a given
||| set of points, towards the goal of optimizing the objective.
public export
Acquisition : Nat -> Shape -> Type
Acquisition batch_size features = Tensor (batch_size :: features) Double -> Tensor [] Double

||| An `AcquisitionOptimizer` returns the points which optimize a given `Acquisition`.
public export
AcquisitionOptimizer : {batch_size : Nat} -> {features : Shape} -> Type
AcquisitionOptimizer = Optimizer $ Tensor (batch_size :: features) Double

-- todo constraint data?
||| Observed query points and objective values
public export
Data : Shape -> Shape -> Type
Data features targets = (Tensor features Double, Tensor targets Double)

||| An `AcquisitionBuilder` constructs an `Acquisition` from historic data and the model over that data.
public export
AcquisitionBuilder : (ProbabilisticModel features targets model dist) => {model : Type} -> (features : Shape) -> (targets : Shape) -> Nat -> Type
AcquisitionBuilder {model} features targets batch_size = Data features targets -> model -> Acquisition batch_size features

||| Construct the acquisition function that estimates the absolute improvement in the best observation
||| if we were to evaluate the objective at a given point.
|||
||| @model The model over the historic data.
||| @best The current best observation.
expected_improvement : (ProbabilisticModel features [] (Gaussian samples [1]) model_t) => (model : model_t) -> (best : Tensor [] Double) -> Acquisition 1 features
--expected_improvement model best at = let normal = predict model at in (best - mean normal) * (cdf normal best) + (?squeeze $ covariance normal) * ?prob

--expected_improvement_by_model : (ProbabilisticModel features [] model) => Data features [] -> model -> Acquisition 1 features
--expected_improvement_by_model (query_points, _) model' at = let best = min $ predict model' (?expand_dims0 query_points) in expected_improvement model best
