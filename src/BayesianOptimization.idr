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
public export 0
ProbabilisticModel : Distribution targets marginal => Shape -> Type
ProbabilisticModel {marginal} features = {0 samples : Nat} ->
        Tensor (samples :: features) Double -> marginal samples

||| An `Acquisition` function quantifies how useful it would be to query the objective at a given  
||| set of points, towards the goal of optimizing the objective.
public export 0
Acquisition : Nat -> Shape -> Type
Acquisition batch_size features = Tensor (batch_size :: features) Double -> Maybe $ Tensor [] Double

||| An `AcquisitionOptimizer` returns the points which optimize a given `Acquisition`.
public export 0
AcquisitionOptimizer : {batch_size : Nat} -> {features : Shape} -> Type
AcquisitionOptimizer = Optimizer $ Tensor (batch_size :: features) Double

-- todo this only allows one point. Should it allow multiple?
||| Observed query points and objective values
public export 0
Data : Shape -> Shape -> Type
Data features targets = {0 samples : Nat} ->   
  (Tensor (samples :: features) Double, Tensor (samples :: targets) Double)

||| An `Empiric` constructs values from historic data and the model over that data.
public export 0
Empiric : Distribution targets marginal => Shape -> Type -> Type
Empiric {targets} {marginal} features out =
  (Data features targets, ProbabilisticModel features {targets} {marginal}) -> out

||| Construct the acquisition function that estimates the absolute improvement in the best
||| observation if we were to evaluate the objective at a given point.
|||
||| @model The model over the historic data.
||| @best The current best observation.
export
expectedImprovement : ProbabilisticModel features {targets=[1]} {marginal=Gaussian [1]} ->
                      (best : Tensor [1] Double) -> Acquisition 1 features
--expectedImprovement predict best at = let marginal = predict at in
--  (best - mean marginal) * (cdf marginal best) + pdf marginal best * (variance marginal) 

-- todo can I get the type checker to infer `targets` and `samples`? It should be able to, given the
-- implementation of `Distribution` for `Gaussian`
||| Build an acquisition function that returns the absolute improvement, expected by the model, in
||| the observation value at each point.
export
expectedImprovementByModel : Empiric features {targets=[1]} {marginal=Gaussian [1]} $
                             Acquisition 1 features
--expectedImprovementByModel ((query_points, _), predict) = let best = min $ predict query_points in
--                                                              expectedImprovement predict best

||| Build an acquisition function that returns the probability that any given point will take a
||| value less than the specified `limit`.
export
probabilityOfFeasibility : (limit : Tensor [] Double) -> Distribution [1] d =>
                           Empiric features {targets=[1]} {marginal=d} $ Acquisition 1 features

||| Build the expected improvement acquisition function in the context of a constraint on the input
||| domain, where points that do not satisfy the constraint do not offer an improvement. The
||| complete acquisition function is built from a constraint acquisition function, which quantifies
||| whether specified points in the input space satisfy the constraint.
export
expectedConstrainedImprovement : Empiric features {targets=[1]} {marginal=Gaussian [1]} $
                                 (Acquisition 1 features -> Acquisition 1 features)

||| A `Connection` encapsulates the machinery to convert an initial representation of data to some
||| arbitrary final value, via another arbitrary intermediate state. The intermediate state can
||| contain just a subset of the original data and thus allows users to allocate different parts of
||| the original data for use in constructing different final values.
|||
||| The primary application in spidr for this is to allow users to allocate individial pairs of
||| data sets and models to `KnowledgeBased`s, without demanding users represent all their data sets
||| and models in any specific way.
public export
data Connection i o = MkConnection (i -> ty) (ty -> o)

export
apply : Connection i o -> i -> o
apply (MkConnection in_ out) = out . in_

export
direct : (i -> o) -> Connection i o
direct = MkConnection (\x => x)

export
Functor (Connection i) where
  map f (MkConnection get g) = MkConnection get $ f . g

export
Applicative (Connection i) where
  pure x = MkConnection (\_ => ()) (\_ => x)
  (MkConnection get g) <*> (MkConnection get' g') =
    MkConnection (\ii => (get ii, get' ii)) (\(t, t') => g t $ g' t')
