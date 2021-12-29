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
module BayesianOptimization.Acquisition

import public Data.Nat
import Distribution
import Tensor
import Data
import Model
import Optimize
import BayesianOptimization.Morphisms
import Error

||| An `Empiric` constructs values from historic data and the model over that data.
|||
||| @features The shape of the feature domain.
||| @out The type of the value constructed by the `Empiric`.
public export 0
Empiric : Distribution targets marginal => (0 features : Shape) -> (0 out : Type) -> Type
Empiric features out = Dataset features targets -> ProbabilisticModel features {marginal} -> out

||| An `Acquisition` function quantifies how useful it would be to query the objective at a given  
||| set of points, towards the goal of optimizing the objective.
|||
||| @batch_size The number of points in the feature domain that the `Acquisition` evaluates
|||   at once.
||| @features The shape of the feature domain.
public export 0
Acquisition : (0 batch_size : Nat) -> {auto 0 _ : GT batch_size 0} -> (0 features : Shape) -> Type
Acquisition batch_size features = Tensor (batch_size :: features) Double -> Tensor [] Double

||| Construct the acquisition function that estimates the absolute improvement in the best
||| observation if we were to evaluate the objective at a given point.
|||
||| @model The model over the historic data.
||| @best The current best observation.
export
expectedImprovement : ProbabilisticModel features {marginal=Gaussian [1]} ->
                      (best : Tensor [] Double) -> Acquisition 1 features
expectedImprovement predict best at =
  let marginal = predict at
      pdf = pdf marginal $ broadcast {to=[_, 1]} best
      variance = squeeze {to=[]} $ variance marginal
      mean = squeeze {to=[]} $ mean marginal
      cdf = cdf marginal $ broadcast {to=[_, 1]} best
   in (best - mean) * cdf + variance * pdf

||| Build an acquisition function that returns the absolute improvement, expected by the model, in
||| the observation value at each point.
export
expectedImprovementByModel : Empiric features {marginal=Gaussian [1]} $ Acquisition 1 features
expectedImprovementByModel (MkDataset query_points _) predict at =
  let best = squeeze $ reduce_min 0 $ mean $ predict query_points
   in expectedImprovement predict best at

||| Build an acquisition function that returns the probability that any given point will take a
||| value less than the specified `limit`.
export
probabilityOfFeasibility : (limit : Tensor [] Double) -> ClosedFormDistribution [1] d =>
                           Empiric features {marginal=d} $ Acquisition 1 features
probabilityOfFeasibility limit _ predict at = cdf (predict at) $ broadcast {to=[_, 1]} limit

||| Build an acquisition function that returns the negative of the lower confidence bound of the
||| probabilistic model. The variance contribution is weighted by a factor `beta`.
|||
||| @beta The weighting given to the variance contribution. If negative, this function will return
|||   `Nothing`.
export
negativeLowerConfidenceBound : (beta : Double) ->
  Either ValueError $ Empiric features {marginal=Gaussian [1]} $ Acquisition 1 features
negativeLowerConfidenceBound beta =
  if beta < 0
  then Left $ MkValueError $ "beta should be greater than or equal to zero, got " ++ show beta
  else Right $ \_, predict, at =>
    let marginal = predict at
     in squeeze $ mean marginal - (const {shape=[]} beta) * (variance marginal)

||| Build the expected improvement acquisition function in the context of a constraint on the input
||| domain, where points that do not satisfy the constraint do not offer an improvement. The
||| complete acquisition function is built from a constraint acquisition function, which quantifies
||| whether specified points in the input space satisfy the constraint.
export
expectedConstrainedImprovement : (limit : Tensor [] Double) ->
  Empiric features {marginal=Gaussian [1]} $ (Acquisition 1 features -> Acquisition 1 features)
