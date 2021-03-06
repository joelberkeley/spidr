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
||| This module contains definitions for and implementations of kernel functions, intended
||| particularly for use in Gaussian processes.
module Model.Kernel

import Tensor
import Data.Nat
import Util

||| A `Kernel` function maps pairs of points in a feature space to the covariance between those two
||| points in some target space.
public export
Kernel : (features : Shape) -> Type
Kernel features = {sk, sk' : _} ->
  Tensor (sk :: features) Double ->
  Tensor (sk' :: features) Double ->
  Tensor [sk, sk'] Double

||| The radial basis function, or squared exponential kernel.
export
rbf : {d : Nat} -> Tensor [] Double -> Kernel [S d]
rbf length_scale x x' = let xs = broadcast {to=[sk, sk', S d]} $ expand 1 x
                            xs' = broadcast {to=[sk, sk', S d]} $ expand 0 x'
                            two = const {shape=[]} 2.0
                            l2_norm = reduce_sum 2 $ (xs' - xs) ^ two
                         in exp (- l2_norm / (two * length_scale ^ two))
