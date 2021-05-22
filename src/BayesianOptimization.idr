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
import GaussianProcess

||| A `ProbabilisticModel` is a mapping from a feature space to a probability distribution over
||| a target space.
interface Distribution targets dist => ProbabilisticModel
(0 features : Shape) dist model where
  ||| Return the probability distribution over the target space at the specified points in the
  ||| feature space, given the model.
  predict : {samples : Nat} -> model -> Tensor (samples :: features) Double -> dist samples

interface Domain where

||| An `Acquisition` function quantifies how useful it would be to query the objective at a given  
||| set of points, towards the goal of optimizing the objective.
public export 0
Acquisition : Nat -> Shape -> Type
Acquisition batch_size features = Tensor (batch_size :: features) Double -> Tensor [] Double

||| An `AcquisitionOptimizer` returns the points which optimize a given `Acquisition`.
public export 0
AcquisitionOptimizer : {batch_size : Nat} -> {features : Shape} -> Type
AcquisitionOptimizer = Optimizer $ Tensor (batch_size :: features) Double

-- todo this only allows one point. Should it allow multiple?
||| Observed query points and objective values
public export 0
Data : Shape -> Shape -> Type
Data features targets = (Tensor features Double, Tensor targets Double)
||| An `AcquisitionBuilder` constructs an `Acquisition` from historic data and the model over that
||| data.
public export 0
KnowledgeBased : Shape -> (dist : Nat -> Type) -> Distribution targets dist => Type -> Type
KnowledgeBased features dist out = {0 model : Type} -> ProbabilisticModel features dist model =>
                                (Data features targets, model) -> out

||| Construct the acquisition function that estimates the absolute improvement in the best
||| observation if we were to evaluate the objective at a given point.
|||
||| @model The model over the historic data.
||| @best The current best observation.
export
expectedImprovement : ProbabilisticModel features (Gaussian []) m =>
                      (model : m) -> (best : Tensor [1] Double) -> Acquisition 1 features
-- expectedImprovement model best at = let gaussian = predict model at in
--   (best - mean gaussian) * (cdf gaussian best) + pdf gaussian best * (variance gaussian) 

-- todo can I get the type checker to infer `targets` and `samples`? It should be able to, given the
-- implementation of `Distribution` for `Gaussian`
||| Build an acquisition function that returns the absolute improvement, expected by the model, in
||| the observation value at each point.
export
expectedImprovementByModel : KnowledgeBased features {targets=[]} (Gaussian []) $
                             Acquisition 1 features
--expectedImprovementByModel ((query_points, _), model) at =
--  let best = min $ predict model (?expand_dims0 query_points) in expectedImprovement model best

||| Build an acquisition function that returns the probability that any given point will take a
||| value less than the specified `limit`.
export
probabilityOfFeasibility : (limit : Tensor [] Double) ->
                           KnowledgeBased features {targets=[]} (Gaussian []) $
                           Acquisition 1 features

export
expectedConstrainedImprovement : KnowledgeBased features {targets=[]} (Gaussian []) $
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
data Connection : Type -> Type -> Type where
  ||| Construct a `Connection`.
  MkConnection : (i -> ty) -> (ty -> o) -> Connection i o

export
Functor (Connection i) where
  map f (MkConnection get g) = MkConnection get $ f . g

export
Applicative (Connection i) where
  pure x = MkConnection (\_ => ()) (\_ => x)
  (MkConnection get g) <*> (MkConnection get' g') = MkConnection
    (\ii => (get ii, get' ii)) (\(t, t') => g t $ g' t')

