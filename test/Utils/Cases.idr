{--
Copyright 2022 Joel Berkeley

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
module Utils.Cases

import public Data.SOP
import Data.Bounded
import Data.List
import public Hedgehog

import Literal
import Types
import Util

namespace Double
  export
  nan, inf : Double
  nan = 0.0 / 0.0
  inf = 1.0 / 0.0

namespace Literal
  export
  nan, inf : Literal [] Double
  nan = Scalar (0.0 / 0.0)
  inf = Scalar (1.0 / 0.0)

export
fixedProperty : PropertyT () -> Property
fixedProperty = withTests 1 . property

maxRank : Nat
maxRank = 5

maxDim : Nat
maxDim = 10

export
dims : Gen Nat
dims = nat $ linear 0 maxDim

export
shapes : Gen Shape
shapes = list (linear 0 maxRank) dims

export covering
literal : (shape : Shape) -> Gen a -> Gen (Literal shape a)
literal [] gen = map Scalar gen
literal (0 :: _) gen = pure []
literal (S d :: ds) gen = [| literal ds gen :: literal (d :: ds) gen |]

pow : Prelude.Num ty => ty -> Nat -> ty
pow x Z = x
pow x (S k) = x * pow x k

int32Bound : Int32
int32Bound = pow 2 10

export
int32s : Gen Int32
int32s = int32 $ linear (-int32Bound) int32Bound

export
nats : Gen Nat
nats = nat $ linear 0 (pow 2 10)

doubleBound : Double
doubleBound = 9999

export
finiteDoubles : Gen Double
finiteDoubles = double $ exponentialDoubleFrom (-doubleBound) 0 doubleBound

export
doubles : Gen Double
doubles = frequency [(1, finiteDoubles), (3, element [-inf, inf, nan])]

export
doublesWithoutNan : Gen Double
doublesWithoutNan = frequency [(1, finiteDoubles), (3, element [-inf, inf])]

export
Show a => Show (InBounds {a} k xs) where
  show InFirst = "InFirst"
  show (InLater prf) = "InLater (\{show prf})"

export
index : (xs : List a) -> {auto 0 nonEmpty : NonEmpty xs} -> Gen (k ** InBounds k xs)
index [] impossible
index (x :: []) = pure (0 ** InFirst)
index (x :: x' :: xs) = do
  (n ** prf) <- index (x' :: xs)
  b <- bool
  pure $ if b then (n ** inBoundsCons (x' :: xs) prf) else (S n ** InLater prf)
