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
||| Acquisition functions for Bayesian optimization.
module BayesianOptimization.Acquisition

import Control.Monad.Reader
import Control.Monad.Identity
import public Data.Nat
import Distribution
import Tensor
import Data
import Model

||| A `DataModel` packages data with a model over that data.
public export
data DataModel : {features, targets : Shape} -> Type -> Type where
  MkDataModel : m -@ (Dataset features targets -@ DataModel {features, targets} m)

export
Copy model => Copy (DataModel {features, targets} model) where
  copy (MkDataModel m d) = do
    MkBang m <- copy m
    MkBang d <- copy d
    pure $ MkBang $ MkDataModel m d
  discard (MkDataModel m d) = discarding m $ discard d

export
(.model) : DataModel m -@ m
(MkDataModel m d).model = discarding d m

export
(.dataset) : Copy m => DataModel {features, targets} m -@ Dataset features targets
(MkDataModel m d).dataset = discarding m d

||| An `Acquisition` function quantifies how useful it would be to query the objective at a given  
||| set of points, towards the goal of optimizing the objective.
|||
||| @batchSize The number of points in the feature domain that the `Acquisition` evaluates
|||   at once.
||| @features The shape of the feature domain.
public export 0
Acquisition : (0 batchSize : Nat) -> {auto 0 _ : GT batchSize 0} -> (0 features : Shape) -> Type
Acquisition batchSize features = Tensor (batchSize :: features) F64 -@ Tag $ Tensor [] F64

||| Construct the acquisition function that estimates the absolute improvement in the best
||| observation if we were to evaluate the objective at a given point.
|||
||| @model The model over the historic data.
||| @best The current best observation.
export
expectedImprovement :
  ProbabilisticModel features [1] Gaussian m =>
  (1 model : m) ->
  (1 best : Tensor [] F64) ->
  Acquisition 1 features
expectedImprovement model best at = do
  MkBang best <- copy best
  MkBang marginal <- copy !(marginalise model at)
  let best' = broadcast {to = [_, 1]} best
  MkBang pdf <- copy !(pdf marginal best')
  MkBang cdf <- copy !(cdf marginal best')
  let mean = squeeze !(mean {event = [1]} {dim = 1} marginal)
      variance = squeeze !(variance {event = [1]} marginal)
  pure $ (best - mean) * cdf + variance * pdf

||| Build an acquisition function that returns the absolute improvement, expected by the model, in
||| the observation value at each point.
export
expectedImprovementByModel :
  Copy modelType =>
  ProbabilisticModel features [1] Gaussian modelType =>
  DataModel {features, targets = [1]} modelType -@ Tag $ Acquisition 1 features
expectedImprovementByModel (MkDataModel model (MkDataset features targets)) = do
  MkBang model <- copy model
  marginal <- marginalise model (discarding targets features)
  MkBang best <- copy $ squeeze {to = []} !(reduce @{Min} [0] !(mean marginal))
  pure $ expectedImprovement model best

||| Build an acquisition function that returns the probability that any given point will take a
||| value less than the specified `limit`.
export
probabilityOfFeasibility :
  (limit : Tensor [] F64) ->
  ClosedFormDistribution [1] dist =>
  ProbabilisticModel features [1] dist modelType =>
  DataModel modelType -@ Acquisition 1 features
probabilityOfFeasibility limit env at =
  cdf !(marginalise env.model at) (broadcast {to = [_, 1]} limit)

||| Build an acquisition function that returns the negative of the lower confidence bound of the
||| probabilistic model. The variance contribution is weighted by a factor `beta`.
|||
||| @beta The weighting given to the variance contribution.
export
negativeLowerConfidenceBound :
  (beta : Double) ->
  {auto 0 betaNonNegative : beta >= 0 = True} ->
  ProbabilisticModel features [1] Gaussian modelType =>
  DataModel modelType -@ Acquisition 1 features
negativeLowerConfidenceBound beta env at = do
  MkBang marginal <- copy !(marginalise env.model at)
  pure $ squeeze $
    !(mean {event = [1]} marginal) - fromDouble beta * !(variance {event = [1]} marginal)

||| Build the expected improvement acquisition function in the context of a constraint on the input
||| domain, where points that do not satisfy the constraint do not offer an improvement. The
||| complete acquisition function is built from a constraint acquisition function, which quantifies
||| whether specified points in the input space satisfy the constraint.
|||
||| **NOTE** This function is not yet implemented.
export
expectedConstrainedImprovement :
  (limit : Tensor [] F64) ->
  ProbabilisticModel features [1] Gaussian modelType =>
  DataModel modelType -@ Tag (Acquisition 1 features -> Acquisition 1 features)
