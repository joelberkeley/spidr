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

||| Objective query points and either corresponding objective values or metadata.
public export 0
Data : {0 samples : Nat} -> Shape -> Shape -> Type
Data features targets =
  (Tensor (samples :: features) Double, Tensor (samples :: targets) Double)

||| An `Empiric` constructs values from historic data and the model over that data.
public export 0
Empiric : (features, targets : Shape) -> (marginal : Nat -> Type) -> Type -> Type
Empiric features targets marginal out = forall s .
  (Data {samples=S s} features targets, ProbabilisticModel features marginal) -> out

infix 9 >>>

||| A `Connection` encapsulates the machinery to convert an initial representation of data to some
||| arbitrary final value, via another arbitrary intermediate state. The intermediate state can
||| contain just a subset of the original data and thus allows users to delegate different parts of
||| the original data for use in constructing different final values.
|||
||| The primary application in spidr for this is to allow users to allocate individial pairs of
||| data sets and models to `Empiric`s, without demanding users represent all their data sets and
||| models in any specific way.
public export
data Connection i o = (>>>) (i -> ty) (ty -> o)

||| Convert the `Connection` to a function.
export
run : Connection i o -> i -> o
run (get >>> g) = g . get

||| Create a `Connection` with no intermediate state.
export
direct : (i -> o) -> Connection i o
direct = (>>>) (\x => x)

export
Functor (Connection i) where
  map f (get >>> g) = get >>> (f . g)

export
Applicative (Connection i) where
  pure x = (\_ => ()) >>> (\_ => x)
  (get >>> g) <*> (get' >>> g') =
    (\ii => (get ii, get' ii)) >>> (\(t, t') => g t $ g' t')
