{--
Copyright (C) 2023  Joel Berkeley

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
module Unit.TestTensor.Sampling

import System

import Device
import Tensor

import Utils
import Utils.Comparison
import Utils.Cases

range : (n : Nat) -> Literal [n] Nat
range n = cast (Vect.range n)

product1 : (x : Nat) -> product (the (List Nat) [x]) = x
product1 x = rewrite plusZeroRightNeutral x in Refl

iidKolmogorovSmirnov :
  {shape : _} -> Tensor shape F64 -> (Tensor shape F64 -> Tensor shape F64) -> Tag $ Tensor [] F64
iidKolmogorovSmirnov samples cdf = do
  let n : Nat
      n = product shape

      indices : Tensor [n] F64 := castDtype $ tensor {dtype = U64} (range n)
      sampleSize : Tensor [] F64 := castDtype $ tensor {dtype = U64} (Scalar n)
      samplesFlat = reshape {sizesEqual = sym (product1 n)} {to = [n]} $ cdf samples
      deviationFromCDF : Tensor [n] F64 = indices / sampleSize - !(sort (<) 0 samplesFlat)
  reduce @{Max} [0] (abs deviationFromCDF)

Prelude.Ord a => Prelude.Ord (Literal [] a) where
  compare (Scalar x) (Scalar y) = compare x y

uniform : Device => Property
uniform = withTests 20 . property $ do
  bound <- forAll (literal [5] finiteDoubles)
  bound' <- forAll (literal [5] finiteDoubles)
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let ksTest = do
    let bound = tensor bound
        bound' = tensor bound'
        bound' = select (bound' == bound) (bound' + fill 1.0e-9) bound'
        key = tensor key
        seed = tensor seed
    samples <- evalStateT seed !(uniform key (broadcast bound) (broadcast bound'))

    let uniformCdf : Tensor [2000, 5] F64 -> Tensor [2000, 5] F64
        uniformCdf x = (x - broadcast bound) / (broadcast $ bound' - bound)

    iidKolmogorovSmirnov samples uniformCdf

  diff (unsafeEval ksTest) (<) 0.015

uniformForNonFiniteBounds : Device => Property
uniformForNonFiniteBounds = property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let bound = tensor [0.0, 0.0, 0.0, -inf, -inf, -inf, inf, inf, nan]
      bound' = tensor [-inf, inf, nan, -inf, inf, nan, inf, nan, nan]
      key = tensor key
      seed = tensor seed
      samples = do evalStateT seed !(uniform key (broadcast bound) (broadcast bound'))
      expected = tensor [-inf, inf, nan, -inf, nan, nan, inf, nan, nan]

  samples ===# pure expected

minNormal : Literal [] Double
minNormal = 2.23e-308  -- see https://en.wikipedia.org/wiki/IEEE_754

uniformForFiniteEqualBounds : Device => Property
uniformForFiniteEqualBounds = withTests 20 . property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let bound = tensor [min @{Finite}, -1.0, -minNormal, 0.0, minNormal, 1.0, max @{Finite}]
      samples = do evalStateT (tensor seed) !(uniform (tensor key) bound bound)

  samples ===# pure bound

uniformSeedIsUpdated : Device => Property
uniformSeedIsUpdated = withTests 20 . property $ do
  bound <- forAll (literal [10] doubles)
  bound' <- forAll (literal [10] doubles)
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let [seed, seed', seed'', sample, sample'] = unsafeEval $ do
        let bound = tensor bound
            bound' = tensor bound'
            key = tensor key
            seed = tensor seed

        rng <- uniform key {shape = [10]} (broadcast bound) (broadcast bound')
        (seed', sample) <- runStateT seed rng
        (seed'', sample') <- runStateT seed' rng
        pure [seed, seed', seed'', sample, sample']

  diff seed' (/=) seed
  diff seed'' (/=) seed'
  diff sample' (/=) sample

uniformIsReproducible : Device => Property
uniformIsReproducible = withTests 20 . property $ do
  bound <- forAll (literal [10] doubles)
  bound' <- forAll (literal [10] doubles)
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let [sample, sample'] = unsafeEval $ do
        let bound = tensor bound
            bound' = tensor bound'
            key = tensor key
            seed = tensor seed

        rng <- uniform {shape = [10]} key (broadcast bound) (broadcast bound')
        sample <- evalStateT seed rng
        sample' <- evalStateT seed rng
        pure [sample, sample']

  sample ==~ sample'

normal : Device => Property
normal = withTests 20 . property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let normalCdf : {shape : _} -> Tensor shape F64 -> Tensor shape F64
      normalCdf x = (fill 1.0 + erf (x / (sqrt $ fill 2.0))) / fill 2.0

      key := tensor key
      seed = tensor seed

      ksTest = do
        samples <- the (Tag $ Tensor [100, 100] F64) $ evalStateT seed (normal key)
        iidKolmogorovSmirnov samples normalCdf

  diff (unsafeEval ksTest) (<) 0.02

normalSeedIsUpdated : Device => Property
normalSeedIsUpdated = withTests 20 . property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let [seed, seed', seed'', sample, sample'] = unsafeEval $ do
        let key = tensor key
            seed = tensor seed
            rng = normal key {shape = [10]}
        (seed', sample) <- runStateT seed rng
        (seed'', sample') <- runStateT seed' rng
        pure [seed, seed', seed'', sample, sample']

  diff seed' (/=) seed
  diff seed'' (/=) seed'
  diff sample' (/=) sample

normalIsReproducible : Device => Property
normalIsReproducible = withTests 20 . property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let [sample, sample'] = unsafeEval $ do
        let key = tensor key
            seed = tensor seed
            rng = normal {shape = [10]} key
        sample <- evalStateT seed rng
        sample' <- evalStateT seed rng
        pure [sample, sample']

  sample ==~ sample'

export
all : Device => List (PropertyName, Property)
all = [
      ("uniform", uniform)
    , ("uniform for infinite and NaN bounds", uniformForNonFiniteBounds)
    , ("uniform is not NaN for finite equal bounds", uniformForFiniteEqualBounds)
    , ("uniform updates seed", uniformSeedIsUpdated)
    , ("uniform produces same samples for same seed", uniformIsReproducible)
    , ("normal", normal)
    , ("normal updates seed", normalSeedIsUpdated)
    , ("normal produces same samples for same seed", normalIsReproducible)
  ]
