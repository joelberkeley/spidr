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
import Model
import Optimize
import BayesianOptimization.Util
import BayesianOptimization.Domain
import Util

||| An `Acquisition` function quantifies how useful it would be to query the objective at a given  
||| set of points, towards the goal of optimizing the objective.
public export 0
Acquisition : (batch_size : Nat) -> {auto prf : GT batch_size 0} -> Shape -> Type
Acquisition batch_size features = Tensor (batch_size :: features) Double -> Tensor [] Double

||| Construct the acquisition function that estimates the absolute improvement in the best
||| observation if we were to evaluate the objective at a given point.
|||
||| @model The model over the historic data.
||| @best The current best observation.
export
expectedImprovement : ProbabilisticModel features {targets=[1]} {marginal=Gaussian [1]} ->
                      (best : Tensor [] Double) -> Acquisition 1 features
expectedImprovement predict best at =
  let marginal = predict at
      pdf = pdf marginal $ broadcast {to=[1, 1]} best
      variance = squeeze {from=[1, 1]} {to=[]} $ variance marginal
      mean = squeeze {from=[1, 1]} {to=[]} $ mean marginal
      cdf = cdf marginal $ broadcast {to=[1, 1]} best
   in (best - mean) * cdf + variance * pdf

||| Build an acquisition function that returns the absolute improvement, expected by the model, in
||| the observation value at each point.
export
expectedImprovementByModel :
  Empiric features {targets=[1]} {marginal=Gaussian [1]} $ Acquisition 1 features
expectedImprovementByModel ((query_points, _), predict) at =
  let best = squeeze {from=[1]} $ reduce_min 0 $ mean $ predict query_points
   in expectedImprovement predict best at

||| Build an acquisition function that returns the probability that any given point will take a
||| value less than the specified `limit`.
export
probabilityOfFeasibility : (limit : Tensor [] Double) -> ClosedFormDistribution [1] m =>
                           Empiric features {targets=[1]} {marginal=m} $ Acquisition 1 features
probabilityOfFeasibility limit (_, predict) at = cdf (predict at) $ broadcast {to=[1, 1]} limit

||| Build an acquisition function that returns the negative of the lower confidence bound of the
||| probabilistic model. The variance contribution is weighted by a factor `beta`.
|||
||| @beta The weighting given to the variance contribution. If negative, this function will return
|||   `Nothing`.
export
negativeLowerConfidenceBound : (beta : Double) ->
  Either ValueError $ Empiric features {targets=[1]} {marginal=Gaussian [1]} $
  Acquisition 1 features
negativeLowerConfidenceBound beta =
  if beta < 0
  then Left $ MkValueError $ "beta should be greater than or equal to zero, got " ++ show beta
  else Right impl where
    impl : Empiric features {targets=[1]} {marginal=Gaussian [1]} $ Acquisition 1 features
    impl (_, predict) at = let marginal = predict at
                               mean = squeeze {from=[1, 1]} {to=[]} $ mean marginal
                               variance = squeeze {from=[1, 1]} {to=[]} $ variance marginal
                            in mean - variance * const {shape=[]} beta

||| Build the expected improvement acquisition function in the context of a constraint on the input
||| domain, where points that do not satisfy the constraint do not offer an improvement. The
||| complete acquisition function is built from a constraint acquisition function, which quantifies
||| whether specified points in the input space satisfy the constraint.
export
expectedConstrainedImprovement : Empiric features {targets=[1]} {marginal=Gaussian [1]} $
                                 (Acquisition 1 features -> Acquisition 1 features)

||| The state type used in the trust region algorithm.
export
record TrustRegionState (dim : Nat) where
    constructor MkTrustRegionState
    localDomain : ContinuousDomain [dim]
    maxDiagonal : Tensor [dim] Double
    best : Tensor [1] Double
    isGlobal : Tensor [] Bool

||| The initial trust region state for a given global domain.
export
init : {d : Nat} -> ContinuousDomain [S d] -> TrustRegionState (S d)
init domain = let maxDiag = (upper domain - lower domain) / (the (Tensor [S d] Double) ?scaling)
               in MkTrustRegionState domain maxDiag ?best (const True)

||| The trust region algorithm for intelligently choosing a local search domain from empirical data
||| and a previous search step.
export
trustRegion : {d : Nat} -> (beta : Double) -> (kappa : Double) -> ContinuousDomain [S d] ->
  Distribution [1] m => Empiric [S d] {targets=[1]} {marginal=m} $
  State (TrustRegionState (S d)) $ ContinuousDomain [S d]
trustRegion beta kappa global_domain ((qp, obs), predict)
            (MkTrustRegionState prev_local_domain prev_max_diag prev_best prev_is_global) =
  let best = reduce_min 0 obs
      tr_vol = reduce_prod 0 $ upper prev_local_domain - lower prev_local_domain
      is_success = squeeze $ best < (prev_best - (const {shape=[]} kappa) * tr_vol)
      max_diag = if ?prev_is_global' then prev_max_diag else
                 let beta = const {shape=[]} beta in
                 if ?is_success' then prev_max_diag / beta else beta * prev_max_diag
      is_global = is_success || not prev_is_global
      local_domain = if ?is_global' then global_domain else
                     let xmin = ?xmin_rhs in MkContinuousDomain ?upper ?lower
   in (MkTrustRegionState local_domain max_diag best is_global, local_domain)
