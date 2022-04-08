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
module Utils.Property

import public Data.SOP
import Data.Bounded
import public Hedgehog

import Literal
import Types

import Utils.Example

maxRank : Nat
maxRank = 5

maxDim : Nat
maxDim = 10

export
shapes : Gen Shape
shapes = list (linear 0 maxRank) (nat $ linear 0 maxDim)

export covering
literal : (shape : Shape) -> Gen a -> Gen (Literal shape a)
literal [] gen = map Scalar gen
literal (0 :: _) gen = pure []
literal (S d :: ds) gen = [| literal ds gen :: literal (d :: ds) gen |]

pow : Prelude.Num ty => ty -> Nat -> ty
pow x Z = x
pow x (S k) = x * pow x k

intBound : Int
intBound = pow 2 10

export
ints : Gen Int
ints = int $ linear (-intBound) intBound

doubleBound : Double
doubleBound = 9999

export
doubles : Gen Double
doubles = frequency [
    (1, double $ exponentialDoubleFrom (-doubleBound) 0 doubleBound),
    (3, element [-1 / 0, 1 / 0, 0 / 0])
  ]

infix 1 ==~

export covering
(==~) : Monad m => {shape : _} -> Literal shape Double -> Literal shape Double -> TestT m ()
(==~) x y = diff x sufficientlyEq' y
  where
  sufficientlyEq' : {shape : _} -> Literal shape Double -> Literal shape Double -> Bool
  sufficientlyEq' x y = all [| sufficientlyEq x y |]
