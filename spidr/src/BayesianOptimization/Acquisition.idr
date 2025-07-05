{--
Copyright (C) 2021  Joel Berkeley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
--}
module BayesianOptimization.Acquisition

import Control.Monad.Reader
import Control.Monad.Identity
import public Data.Nat
import Distribution
import Tensor
import Data
import Model

%prefix_record_projections off

||| A `DataModel` packages data with a model over that data.
public export
record DataModel modelType {auto probabilisticModel : ProbabilisticModel f t marginal modelType} where
  constructor MkDataModel

  ||| A probabilistic model
  model : modelType

  ||| The data the model is trained on
  dataset : Dataset f t

%prefix_record_projections on

||| An `Acquisition` function quantifies how useful it would be to query the objective at a given  
||| set of points, towards the goal of optimizing the objective.
|||
||| @batchSize The number of points in the feature domain that the `Acquisition` evaluates
|||   at once.
||| @features The shape of the feature domain.
public export 0
Acquisition : (0 batchSize : Nat) -> {auto 0 _ : GT batchSize 0} -> (0 features : Shape) -> Type
Acquisition batchSize features = Tensor (batchSize :: features) F64 -> Tag $ Tensor [] F64

||| Construct the acquisition function that estimates the absolute improvement in the best
||| observation if we were to evaluate the objective at a given point.
|||
||| @model The model over the historic data.
||| @best The current best observation.
export
expectedImprovement :
  ProbabilisticModel features [1] Gaussian m =>
  (model : m) ->
  (best : Tensor [] F64) ->
  Acquisition 1 features
expectedImprovement model best at = do
  best <- tag best
  marginal <- tag =<< marginalise model at
  let best' = broadcast {to = [_, 1]} best
  pdf <- tag =<< pdf marginal best'
  cdf <- tag =<< cdf marginal best'
  let mean = squeeze !(mean {event = [1]} {dim = 1} marginal)
      variance = squeeze !(variance {event = [1]} marginal)
  pure $ (best - mean) * cdf + variance * pdf

||| Build an acquisition function that returns the absolute improvement, expected by the model, in
||| the observation value at each point.
export
expectedImprovementByModel :
  ProbabilisticModel features [1] Gaussian modelType =>
  ReaderT (DataModel modelType) Tag $ Acquisition 1 features
expectedImprovementByModel = MkReaderT $ \env => do
  marginal <- marginalise env.model env.dataset.features
  best <- tag $ squeeze !(reduce @{Min} [0] !(mean {event = [1]} marginal))
  pure $ expectedImprovement env.model best

||| Build an acquisition function that returns the probability that any given point will take a
||| value less than the specified `limit`.
export
probabilityOfFeasibility :
  (limit : Tensor [] F64) ->
  ClosedFormDistribution [1] dist =>
  ProbabilisticModel features [1] dist modelType =>
  ReaderT (DataModel modelType) Tag $ Acquisition 1 features
probabilityOfFeasibility limit =
  asks $ \env, at => do cdf !(marginalise env.model at) (broadcast {to = [_, 1]} limit)

||| Build an acquisition function that returns the negative of the lower confidence bound of the
||| probabilistic model. The variance contribution is weighted by a factor `beta`.
|||
||| @beta The weighting given to the variance contribution.
export
negativeLowerConfidenceBound :
  (beta : Double) ->
  {auto 0 betaNonNegative : beta >= 0 = True} ->
  ProbabilisticModel features [1] Gaussian modelType =>
  ReaderT (DataModel modelType) Tag $ Acquisition 1 features
negativeLowerConfidenceBound beta = asks $ \env, at => do
  marginal <- tag =<< marginalise env.model at
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
  ReaderT (DataModel modelType) Tag (Acquisition 1 features -> Acquisition 1 features)
