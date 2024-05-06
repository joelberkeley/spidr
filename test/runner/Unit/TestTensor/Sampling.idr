{--
Copyright 2023 Joel Berkeley

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
module Unit.TestTensor.Sampling

import System

import Device
import Literal
import Tensor

import Utils
import Utils.Comparison
import Utils.Cases

range : (n : Nat) -> Literal [n] Nat
range n = cast (Vect.range n)

product1 : (x : Nat) -> product (the (List Nat) [x]) = x
product1 x = rewrite plusZeroRightNeutral x in Refl

partial
iidKolmogorovSmirnov :
  {shape : _} -> Tensor shape F64 -> (Tensor shape F64 -> Graph $ Tensor shape F64) -> Graph $ Tensor [] F64
iidKolmogorovSmirnov samples cdf = do
  let n : Nat
      n = product shape

  let indices : Graph $ Tensor [n] F64 = castDtype !(tensor {dtype=U64} (range n))
      sampleSize : Graph $ Tensor [] F64 = castDtype !(tensor {dtype=U64} (Scalar n))
  samplesFlat <- reshape {sizesEqual=sym (product1 n)} {to=[n]} !(cdf samples)
  deviationFromCDF <- the (Graph $ Tensor [n] F64) $ indices / sampleSize - sort (<) 0 samplesFlat
  reduce @{Max} [0] !(abs deviationFromCDF)

Prelude.Ord a => Prelude.Ord (Literal [] a) where
  compare (Scalar x) (Scalar y) = compare x y

partial
uniform : Device => Property
uniform = withTests 20 . property $ do
  bound <- forAll (literal [5] finiteDoubles)
  bound' <- forAll (literal [5] finiteDoubles)
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let ksTest = do
    let bound = tensor bound
        bound' = tensor bound'
        bound' = select !(bound' == bound) !(bound' + fill 1.0e-9) !bound'
    key <- tensor key
    seed <- tensor seed
    samples <- evalStateT seed !(uniform key !(broadcast !bound) !(broadcast !bound'))

    let uniformCdf : Tensor [2000, 5] F64 -> Graph $ Tensor [2000, 5] F64
        uniformCdf x = Tensor.(/) (pure x - broadcast !bound) (broadcast !(bound' - bound))

    iidKolmogorovSmirnov samples uniformCdf

  diff (unsafeEval ksTest) (<) 0.015

partial
uniformForNonFiniteBounds : Device => Property
uniformForNonFiniteBounds = property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let samples = do
    bound <- tensor [0.0, 0.0, 0.0, -inf, -inf, -inf, inf, inf, nan]
    bound' <- tensor [-inf, inf, nan, -inf, inf, nan, inf, nan, nan]
    key <- tensor key
    seed <- tensor seed
    evalStateT seed !(uniform key !(broadcast bound) !(broadcast bound'))

  samples ===# tensor [-inf, inf, nan, -inf, nan, nan, inf, nan, nan]

partial
uniformForFiniteEqualBounds : Device => Property
uniformForFiniteEqualBounds = withTests 20 . property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let bound = tensor [min @{Finite}, -1.0, -1.0e-307, 0.0, 1.0e-307, 1.0, max @{Finite}]
      samples = do evalStateT !(tensor seed) !(uniform !(tensor key) !bound !bound)

  samples ===# bound

partial
uniformSeedIsUpdated : Device => Property
uniformSeedIsUpdated @{device} = withTests 20 . property $ do
  bound <- forAll (literal [10] doubles)
  bound' <- forAll (literal [10] doubles)
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let [seed, seed', seed'', sample, sample'] = unsafePerformIO $ eval device $ do
        bound <- tensor bound
        bound' <- tensor bound'
        key <- tensor key
        seed <- tensor seed

        rng <- uniform key {shape=[10]} !(broadcast bound) !(broadcast bound')
        (seed', sample) <- runStateT seed rng
        (seed'', sample') <- runStateT seed' rng
        pure [seed, seed', seed'', sample, sample']

  diff seed' (/=) seed
  diff seed'' (/=) seed'
  diff sample' (/=) sample

partial
uniformIsReproducible : Device => Property
uniformIsReproducible @{device} = withTests 20 . property $ do
  bound <- forAll (literal [10] doubles)
  bound' <- forAll (literal [10] doubles)
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let [sample, sample'] = unsafePerformIO $ eval device $ do
        bound <- tensor bound
        bound' <- tensor bound'
        key <- tensor key
        seed <- tensor seed

        rng <- uniform {shape=[10]} key !(broadcast bound) !(broadcast bound')
        sample <- evalStateT seed rng
        sample' <- evalStateT seed rng
        pure [sample, sample']

  sample ==~ sample'

partial
normal : Device => Property
normal = withTests 20 . property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let ksTest = do
        key <- tensor key
        seed <- tensor seed

        samples <- the (Graph $ Tensor [100, 100] F64) $ evalStateT seed (normal key)

        let normalCdf : {shape : _} -> Tensor shape F64 -> Graph $ Tensor shape F64
            normalCdf x = do (fill 1.0 + erf !(pure x / (sqrt !(fill 2.0)))) / fill 2.0

        iidKolmogorovSmirnov samples normalCdf

  diff (unsafeEval ksTest) (<) 0.02

partial
normalSeedIsUpdated : Device => Property
normalSeedIsUpdated @{device} = withTests 20 . property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let [seed, seed', seed'', sample, sample'] = unsafePerformIO $ eval device $ do
        key <- tensor key
        seed <- tensor seed
        let rng = normal key {shape=[10]}
        (seed', sample) <- runStateT seed rng
        (seed'', sample') <- runStateT seed' rng
        pure [seed, seed', seed'', sample, sample']

  diff seed' (/=) seed
  diff seed'' (/=) seed'
  diff sample' (/=) sample

partial
normalIsReproducible : Device => Property
normalIsReproducible @{device} = withTests 20 . property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let [sample, sample'] = unsafePerformIO $ eval device $ do
        key <- tensor key
        seed <- tensor seed

        let rng = normal {shape=[10]} key
        sample <- evalStateT seed rng
        sample' <- evalStateT seed rng
        pure [sample, sample']

  sample ==~ sample'

export partial
all : Device => List (PropertyName, Property)
all = [
      ("uniform", uniform)
    , ("uniform for infinite and NaN bounds", uniformForNonFiniteBounds)
    , ("uniform is not NaN for finite equal bounds", uniformForFiniteEqualBounds)
    -- uses tuple , ("uniform updates seed", uniformSeedIsUpdated)
    -- uses tuple , ("uniform produces same samples for same seed", uniformIsReproducible)
    , ("normal", normal)
    -- uses tuple , ("normal updates seed", normalSeedIsUpdated)
    -- uses tuple , ("normal produces same samples for same seed", normalIsReproducible)
  ]