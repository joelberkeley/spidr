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
module BayesianOptimization.Util

import Distribution
import Tensor
import Model

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
