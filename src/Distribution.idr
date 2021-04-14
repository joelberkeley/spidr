module Distribution

import Tensor
import Data.Vect

||| assumes multivariate, where every sub-event has the same shape
export
interface Distribution (dim : Nat) (event_shape : Shape) dist where
  mean : dist -> Tensor (dim :: event_shape) Double
  covariance : dist -> Tensor (dim :: dim :: event_shape) Double
  cdf : dist -> Tensor (dim :: event_shape) Double -> Tensor [] Double

public export
data Gaussian : (dim : Nat) -> (event_shape : Shape) -> Type where
  MkGaussian : (mean : Tensor (dim :: event_shape) Double) -> (covariance : Tensor (dim :: dim :: event_shape) Double) -> Gaussian dim event_shape

{dim : Nat} -> {event_shape : Shape} -> Distribution dim event_shape (Gaussian dim event_shape) where
  mean (MkGaussian mean' _) = mean'
  covariance  (MkGaussian _ cov) = cov
  cdf (MkGaussian mean cov) = ?cdf'

