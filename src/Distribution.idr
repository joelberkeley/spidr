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
||| This module contains definitions and implementations for probability distributions.
module Distribution

import Tensor
import Data.Vect

||| A joint, or multivariate distribution over a tensor of floating point values. Every sub-event
||| is assumed to have the same shape.
export
interface Distribution (dim : Nat) (event_shape : Shape) dist where
  ||| The mean of the distribution
  mean : dist -> Tensor (dim :: event_shape) Double

  ||| The covariance, or correlation between sub-events
  covariance : dist -> Tensor (dim :: dim :: event_shape) Double

||| A joint Gaussian distribution.
public export
data Gaussian : (dim : Nat) -> (event_shape : Shape) -> Type where
  ||| @mean The Gaussian mean.
  ||| @covariance The Gaussian covariance.
  MkGaussian : (mean : Tensor (dim :: event_shape) Double) -> (covariance : Tensor (dim :: dim :: event_shape) Double) -> Gaussian dim event_shape

{dim : Nat} -> {event_shape : Shape} -> Distribution dim event_shape (Gaussian dim event_shape) where
  mean (MkGaussian mean' _) = mean'
  covariance  (MkGaussian _ cov) = cov

||| The cumulative distribution function of the Gaussian at the specified point (that is, the
||| probability the random variable takes a value less than or equal to the given point).
export
cdf : Gaussian 1 event_shape -> Tensor (1 :: event_shape) Double -> Tensor [] Double
