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
  {shape : _} -> Tensor shape F64 -> (Tensor shape F64 -> Ref $ Tensor shape F64) -> Ref $ Tensor [] F64
iidKolmogorovSmirnov samples cdf = do
  let n : Nat
      n = product shape

  let indices : Ref $ Tensor [n] F64 = castDtype !(fromLiteral {dtype=U64} (range n))
      sampleSize : Ref $ Tensor [] F64 = castDtype !(fromLiteral {dtype=U64} (Scalar n))
  samplesFlat <- reshape {sizesEqual=sym (product1 n)} {to=[n]} !(cdf samples)
  deviationFromCDF <- the (Ref $ Tensor [n] F64) $ indices / sampleSize - sort (<) 0 samplesFlat
  reduce @{Max} [0] !(abs deviationFromCDF)

Prelude.Ord a => Prelude.Ord (Literal [] a) where
  compare (Scalar x) (Scalar y) = compare x y

partial
uniform : Property
uniform = withTests 20 . property $ do
  bound <- forAll (literal [5] finiteDoubles)
  bound' <- forAll (literal [5] finiteDoubles)
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let ksTest = do
    let bound = fromLiteral bound
        bound' = fromLiteral bound'
        bound' = select !(bound' == bound) !(bound' + fill 1.0e-9) !bound'
    key <- fromLiteral key
    seed <- fromLiteral seed
    samples <- evalStateT seed !(uniform key !(broadcast !bound) !(broadcast !bound'))

    let uniformCdf : Tensor [2000, 5] F64 -> Ref $ Tensor [2000, 5] F64
        uniformCdf x = Tensor.(/) (pure x - broadcast !bound) (broadcast !(bound' - bound))

    iidKolmogorovSmirnov samples uniformCdf

  diff (toLiteral ksTest) (<) 0.015

partial
uniformForNonFiniteBounds : Property
uniformForNonFiniteBounds = property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let samples = do
    bound <- fromLiteral [0.0, 0.0, 0.0, -inf, -inf, -inf, inf, inf, nan]
    bound' <- fromLiteral [-inf, inf, nan, -inf, inf, nan, inf, nan, nan]
    key <- fromLiteral key
    seed <- fromLiteral seed
    evalStateT seed !(uniform key !(broadcast bound) !(broadcast bound'))

  samples ===# fromLiteral [-inf, inf, nan, -inf, nan, nan, inf, nan, nan]

partial
uniformForFiniteEqualBounds : Property
uniformForFiniteEqualBounds = withTests 20 . property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let bound = fromLiteral [min @{Finite}, -1.0, -1.0e-308, 0.0, 1.0e-308, 1.0, max @{Finite}]
      samples = do evalStateT !(fromLiteral seed) !(uniform !(fromLiteral key) !bound !bound)

  samples ===# bound

partial
uniformSeedIsUpdated : Property
uniformSeedIsUpdated = withTests 20 . property $ do
  bound <- forAll (literal [10] doubles)
  bound' <- forAll (literal [10] doubles)
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let everything = do
        bound <- fromLiteral bound
        bound' <- fromLiteral bound'
        key <- fromLiteral key
        seed <- fromLiteral seed

        rng <- uniform key {shape=[10]} !(broadcast bound) !(broadcast bound')
        (seed', sample) <- runStateT seed rng
        (seed'', sample') <- runStateT seed' rng
        seeds <- concat 0 !(concat 0 seed seed') seed''
        samples <- concat 0 !(expand 0 sample) !(expand 0 sample')
        pure (seeds, samples)

      [seed, seed', seed''] = toLiteral (do (seeds, _) <- everything; pure seeds)
      [sample, sample'] = toLiteral (do (_, samples) <- everything; pure samples)

  diff seed' (/=) seed
  diff seed'' (/=) seed'
  diff sample' (/=) sample

partial
uniformIsReproducible : Property
uniformIsReproducible = withTests 20 . property $ do
  bound <- forAll (literal [10] doubles)
  bound' <- forAll (literal [10] doubles)
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let [sample, sample'] = toLiteral $ do
        bound <- fromLiteral bound
        bound' <- fromLiteral bound'
        key <- fromLiteral key
        seed <- fromLiteral seed

        rng <- uniform {shape=[10]} key !(broadcast bound) !(broadcast bound')
        sample <- evalStateT seed rng
        sample' <- evalStateT seed rng
        concat 0 !(expand 0 sample) !(expand 0 sample')

  sample ==~ sample'

partial
normal : Property
normal = withTests 20 . property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let ksTest = do
        key <- fromLiteral key
        seed <- fromLiteral seed

        samples <- the (Ref $ Tensor [100, 100] F64) $ evalStateT seed (normal key)

        let normalCdf : {shape : _} -> Tensor shape F64 -> Ref $ Tensor shape F64
            normalCdf x = do (fill 1.0 + erf !(pure x / (sqrt !(fill 2.0)))) / fill 2.0

        iidKolmogorovSmirnov samples normalCdf

  diff (toLiteral ksTest) (<) 0.02

partial
normalSeedIsUpdated : Property
normalSeedIsUpdated = withTests 20 . property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let everything = do
        key <- fromLiteral key
        seed <- fromLiteral seed
        let rng = normal key {shape=[10]}
        (seed', sample) <- runStateT seed rng
        (seed'', sample') <- runStateT seed' rng
        seeds <- concat 0 !(concat 0 seed seed') seed''
        samples <- concat 0 !(expand 0 sample) !(expand 0 sample')
        pure (seeds, samples)

      [seed, seed', seed''] = toLiteral (do (seeds, _) <- everything; pure seeds)
      [sample, sample'] = toLiteral (do (_, samples) <- everything; pure samples)

  diff seed' (/=) seed
  diff seed'' (/=) seed'
  diff sample' (/=) sample

partial
normalIsReproducible : Property
normalIsReproducible = withTests 20 . property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let [sample, sample'] = toLiteral $ do
        key <- fromLiteral key
        seed <- fromLiteral seed

        let rng = normal {shape=[10]} key
        sample <- evalStateT seed rng
        sample' <- evalStateT seed rng
        concat 0 !(expand 0 sample) !(expand 0 sample')

  sample ==~ sample'

export partial
all : List (PropertyName, Property)
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
