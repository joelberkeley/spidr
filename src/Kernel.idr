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
module Kernel

import Data.Vect
import Tensor

public export
Kernel : (features : Shape) -> Type
Kernel features = {sk, sk' : Nat} -> Tensor (sk :: features) Double -> Tensor (sk' :: features) Double -> Tensor [sk, sk'] Double
-- Kernel samples features = Tensor (samples :: features) Double -> Tensor (samples :: features) Double -> Tensor [samples, samples] Double

-- avoid manipulating the underlying array directly. This code is essentially
-- user code and shouldn't have knowledge of the underyling Tensor representation

-- export
-- radial_basis_function : Tensor [] Double -> Kernel [domain_dim]
-- radial_basis_function {samples} {domain_dim} variance x x' = let squares = the (Tensor [samples, domain_dim] Double) (map (pow 2.0) (x - x')) in
--                                                                  exp_ew $ foldr1 {leading=[samples]} {tail=[]} (+) 1 $ squares / (2.0 * variance)

export
linear : Kernel features
