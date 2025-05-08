{--
Copyright (C) 2025  Joel Berkeley

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
||| Probability distributions.
module Distribution

import Data.Nat
import Tensor
import Constants

||| A joint, or multivariate distribution over a tensor of floating point values, where the first
||| two central moments (mean and covariance) are known. Every sub-event is assumed to have the
||| same shape.
|||
||| @dist Constructs the distribution from the shape of each sub-event and the number of events in
|||   the distribution.
public export
interface Distribution (0 dist : (0 event : Shape) -> (0 dim : Nat) -> Type) where
  ||| The mean of the distribution.
  mean : dist event dim -> Tag $ Tensor (dim :: event) F64

  ||| The covariance, or correlation, between sub-events.
  cov : dist event dim -> Tag $ Tensor (dim :: dim :: event) F64

||| The variance of a single random variable.
export
variance : {event : _} -> Distribution dist => dist event 1 -> Tag $ Tensor (1 :: event) F64
variance dist = squeeze {from = (1 :: 1 :: event)} <$> cov dist

||| A joint, or multivariate distribution over a tensor of floating point values, where the density
||| function and corresponding cumulative density function are known (either analytically or via
||| approximation). Every sub-event is assumed to have the same shape.
|||
||| @event The shape of each sub-event.
||| @dist Constructs the distribution from the shape of each sub-event and the number of events in
|||   the distribution.
public export
interface Distribution dist  =>
  ClosedFormDistribution (0 event : Shape)
    (0 dist : (0 event : Shape) -> (0 dim : Nat) -> Type) where
      ||| The probability density function of the distribution at the specified point.
      pdf : dist event (S d) -> Tensor (S d :: event) F64 -> Tag $ Tensor [] F64

      ||| The cumulative distribution function of the distribution at the specified point (that is,
      ||| the probability the random variable takes a value less than or equal to the given point).
      cdf : dist event (S d) -> Tensor (S d :: event) F64 -> Tag $ Tensor [] F64

||| A joint Gaussian distribution.
|||
||| @event The shape of each sub-event.
||| @dim The number of sub-events.
public export
data Gaussian : (0 event : Shape) -> (0 dim : Nat) -> Type where
  ||| @mean The mean of the events.
  ||| @cov The covariance between events.
  MkGaussian : {d : Nat} -> (mean : Tensor (S d :: event) F64) ->
               (cov : Tensor (S d :: S d :: event) F64) ->
               Gaussian event (S d)

export
Taggable (Gaussian event dim) where
  tag (MkGaussian mean cov) = [| MkGaussian (tag mean) (tag cov) |]

export
Distribution Gaussian where
  mean (MkGaussian mean' _) = pure mean'
  cov (MkGaussian _ cov') = pure cov'

||| **NOTE** `cdf` is implemented only for univariate `Gaussian`.
export
ClosedFormDistribution [1] Gaussian where
  pdf (MkGaussian {d} mean cov) x = do
    cholCov <- tag $ cholesky $ squeeze {to = [S d, S d]} cov
    tri <- tag $ cholCov |\ squeeze (x - mean)
    let exponent = - tri @@ tri / 2.0
    covSqrtDet <- reduce @{Prod} [0] (diag cholCov)
    let denominator = fromDouble (pow (2.0 * pi) (cast (S d) / 2.0)) * covSqrtDet
    pure (exp exponent / denominator)

  cdf (MkGaussian {d = S _} _ _) _ = ?multivariateGaussianCDF
  cdf (MkGaussian {d = 0} mean cov) x =
    pure $ (1.0 + erf (squeeze (x - mean) / (sqrt (squeeze cov * 2.0)))) / 2.0
