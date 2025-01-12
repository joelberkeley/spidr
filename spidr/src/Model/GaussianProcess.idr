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
||| Gaussian process inference.
module Model.GaussianProcess

import Control.Relation
import Data.Nat

import Constants
import Distribution
import Data
import Model
import Model.Kernel
import Model.MeanFunction
import Optimize
import Tensor

||| A Gaussian process is a collection of random variables, any finite number of which have joint
||| Gaussian distribution. It can be viewed as a function from a feature space to a joint Gaussian
||| distribution over a target space.
|||
||| @features The shape of the feature domain.
public export
data GaussianProcess : (0 features : Shape) -> Type where
  ||| Construct a `GaussianProcess` as a pair of mean function and kernel.
  MkGP : MeanFunction features -> Kernel features -> GaussianProcess features

posterior :
  GaussianProcess features ->
  Tensor [] F64 ->
  {s : _} -> (Tensor ((S s) :: features) F64, Tensor [S s] F64) ->
  Tag $ GaussianProcess features
posterior (MkGP priorMeanf priorKernel) noise (xTrain, yTrain) = do
  l <- tag $ cholesky $ !(priorKernel xTrain xTrain) + noise * identity
  let alpha = l.T \| (l |\ yTrain)

      posteriorMeanf : MeanFunction features
      posteriorMeanf x = pure $ !(priorMeanf x) + !(priorKernel x xTrain) @@ alpha

      posteriorKernel : Kernel features
      posteriorKernel x x' = pure
        $ !(priorKernel x x') - (l |\ !(priorKernel xTrain x)).T @@ (l |\ !(priorKernel xTrain x'))

  pure $ MkGP posteriorMeanf posteriorKernel

logMarginalLikelihood :
  GaussianProcess features ->
  Tensor [] F64 ->
  {s : _} -> (Tensor ((S s) :: features) F64, Tensor [S s] F64) ->
  Tag $ Tensor [] F64
logMarginalLikelihood (MkGP _ kernel) noise (x, y) = do
  l <- tag $ cholesky (!(kernel x x) + noise * identity)
  let alpha = l.T \| (l |\ y)
  pure $ - y @@ alpha / 2.0 - !(trace (log l)) - fromDouble (cast (S s)) * log (2.0 * pi) / 2.0

||| A trainable model implementing vanilla Gaussian process regression. That is, regression with a
||| Gaussian process as conjugate prior for homoscedastic Gaussian likelihoods. See the following
||| for details:
|||
||| Gaussian Processes for Machine Learning
||| Carl Edward Rasmussen and Christopher K. I. Williams
||| The MIT Press, 2006. ISBN 0-262-18253-X.
|||
||| or
|||
||| Pattern Recognition and Machine Learning, Christopher M. Bishop
public export
data ConjugateGPRegression : (0 features : Shape) -> Type where
  ||| @gpFromHyperparameters Constructs a Gaussian process from the hyperparameters (presented as
  |||   a vector)
  ||| @hyperparameters The hyperparameters (excluding noise) presented as a vector.
  ||| @noise The likehood amplitude, or observation noise.
  MkConjugateGPR :
    {p : _} ->
    (gpFromHyperparameters : Tensor [p] F64 -> Tag $ GaussianProcess features) ->
    (hyperparameters : Tensor [p] F64) ->
    (noise : Tensor [] F64) ->
    ConjugateGPRegression features

||| A probabilistic model from feature values to a distribution over latent target values.
export
[Latent] ProbabilisticModel features [1] Gaussian (ConjugateGPRegression features) where
  marginalise (MkConjugateGPR mkGP gpParams _) x = do
    MkGP meanf kernel <- mkGP gpParams
    [| MkGaussian (expand 1 <$> meanf x) (expand 2 <$> kernel x x) |]

||| A probabilistic model from feature values to a distribution over observed target values.
export
[Observed] ProbabilisticModel features [1] Gaussian (ConjugateGPRegression features) where
  marginalise gpr@(MkConjugateGPR _ _ noise) x = do
    MkGaussian latentMean latentCov <- marginalise @{Latent} gpr x
    pure $ MkGaussian latentMean $ latentCov + broadcast (expand 2 (noise * identity {n = S n}))

||| Fit the Gaussian process and noise to the specified data.
export
fit : (forall n . Tensor [n] F64 -> Optimizer $ Tensor [n] F64)
  -> Dataset features [1]
  -> ConjugateGPRegression features
  -> Tag $ ConjugateGPRegression features
fit optimizer (MkDataset x y) (MkConjugateGPR {p} mkPrior gpParams noise) = do
  let objective : Tensor [S p] F64 -> Tag $ Tensor [] F64
      objective params = do
        let priorParams = slice [1.to (S p)] params
        logMarginalLikelihood !(mkPrior priorParams) (slice [at 0] params) (x, squeeze y)

  params <- optimizer (concat 0 (expand 0 noise) gpParams) objective

  let mkPosterior : Tensor [p] F64 -> Tag $ GaussianProcess features
      mkPosterior params' = posterior !(mkPrior params') (squeeze noise) (x, squeeze y)

  pure $ MkConjugateGPR mkPosterior (slice [1.to (S p)] params) (slice [at 0] params)

    where
    %hint
    reflexive : {n : _} -> LTE n n
    reflexive = Relation.reflexive {ty = Nat}
