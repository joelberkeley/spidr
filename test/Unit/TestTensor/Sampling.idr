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
  {shape : _} -> Tensor shape dtype -> (Tensor shape dtype -> Ref $ Tensor shape F64) -> Ref $ Tensor [] F64
iidKolmogorovSmirnov samples cdf = do
  let n : Nat
      n = product shape

  let indices : Ref $ Tensor [n] F64 = castDtype !(tensor {dtype=U64} (range n))
      sampleSize : Ref $ Tensor [] F64 = castDtype !(tensor {dtype=U64} (Scalar n))
  samplesFlat <- reshape {sizesEqual=sym (product1 n)} {to=[n]} !(cdf samples)
  deviationFromCDF <- the (Ref $ Tensor [n] F64) $ indices / sampleSize - sort (<) 0 samplesFlat
  reduce @{Max} [0] !(abs deviationFromCDF)

Prelude.Ord a => Prelude.Ord (Literal [] a) where
  compare (Scalar x) (Scalar y) = compare x y

namespace F64
  export partial
  uniform : Property
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

      let uniformCdf : Tensor [2000, 5] F64 -> Ref $ Tensor [2000, 5] F64
          uniformCdf x = Tensor.(/) (pure x - broadcast !bound) (broadcast !(bound' - bound))

      iidKolmogorovSmirnov samples uniformCdf

    diff (unsafeEval ksTest) (<) 0.015

  export partial
  uniformForNonFiniteBounds : Property
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

  export partial
  uniformForFiniteEqualBounds : Property
  uniformForFiniteEqualBounds = withTests 20 . property $ do
    key <- forAll (literal [] nats)
    seed <- forAll (literal [1] nats)

    let bound = tensor [min @{Finite}, -1.0, -1.0e-308, 0.0, 1.0e-308, 1.0, max @{Finite}]
        samples = do evalStateT !(tensor seed) !(uniform !(tensor key) !bound !bound)

    samples ===# bound

  export partial
  uniformSeedIsUpdated : Property
  uniformSeedIsUpdated = withTests 20 . property $ do
    bound <- forAll (literal [10] doubles)
    bound' <- forAll (literal [10] doubles)
    key <- forAll (literal [] nats)
    seed <- forAll (literal [1] nats)

    let everything = do
          bound <- tensor bound
          bound' <- tensor bound'
          key <- tensor key
          seed <- tensor seed

          rng <- uniform key {shape=[10]} !(broadcast bound) !(broadcast bound')
          (seed', sample) <- runStateT seed rng
          (seed'', sample') <- runStateT seed' rng
          seeds <- concat 0 !(concat 0 seed seed') seed''
          samples <- concat 0 !(expand 0 sample) !(expand 0 sample')
          pure (seeds, samples)

        [seed, seed', seed''] = unsafeEval (do (seeds, _) <- everything; pure seeds)
        [sample, sample'] = unsafeEval (do (_, samples) <- everything; pure samples)

    diff seed' (/=) seed
    diff seed'' (/=) seed'
    diff sample' (/=) sample

  export partial
  uniformIsReproducible : Property
  uniformIsReproducible = withTests 20 . property $ do
    bound <- forAll (literal [10] doubles)
    bound' <- forAll (literal [10] doubles)
    key <- forAll (literal [] nats)
    seed <- forAll (literal [1] nats)

    let [sample, sample'] = unsafeEval $ do
          bound <- tensor bound
          bound' <- tensor bound'
          key <- tensor key
          seed <- tensor seed

          rng <- uniform {shape=[10]} key !(broadcast bound) !(broadcast bound')
          sample <- evalStateT seed rng
          sample' <- evalStateT seed rng
          concat 0 !(expand 0 sample) !(expand 0 sample')

    sample ==~ sample'

Show (Compare p xs ys) where
  show _ = "Compare (contents omitted)"

orderedPair : (shape : Shape) -> Gen (xs : Literal shape Nat ** ys ** Compare LT xs ys)
orderedPair [] = [| lits nats nats |] where
  ord : (n, m : Nat) -> LT n (S (n + m))
  ord n m = LTESucc (lteAddRight n)

  lits : Nat -> Nat -> (n : Literal [] Nat ** m ** Compare LT n m)
  lits n m = (Scalar n ** Scalar (S (n + m)) ** Scalar (ord n m))

orderedPair (0 :: _) = pure ([] ** [] ** [])
orderedPair (S d :: ds) = do
  (x ** y ** ord) <- orderedPair ds
  (xs ** ys ** ords) <- orderedPair (d :: ds)
  pure (x :: xs ** y :: ys ** ord :: ords)

namespace U64
  export partial
  uniform : Property
  uniform = withTests 20 . property $ do
    (lower ** upper ** ordered) <- forAll (orderedPair [10])
    key <- forAll (literal [] nats)
    seed <- forAll (literal [1] nats)

    let uniformCdf : Tensor [1, 10] U64 -> Ref $ Tensor [1, 10] F64
        uniformCdf x = do
          lower <- castDtype !(tensor {dtype = U64} [lower])
          let upper = castDtype !(tensor {dtype = U64} [upper])
          (castDtype x - pure lower) / (upper - pure lower)

    let ksTest := do
      rand <- U64.uniform !(tensor key) [lower] [upper]
      samples <- evalStateT !(tensor seed) rand
      iidKolmogorovSmirnov samples uniformCdf

    -- samples ===# fill 0
    diff (unsafeEval ksTest) (<) 0.01
{-
  export covering
  uniformBoundsAreInclusive : Property
  uniformBoundsAreInclusive = property $ do
    bound <- forAll (literal [100] nats)
    key <- forAll (literal [] nats)
    seed <- forAll (literal [1] nats)

    let bound = fromLiteral bound
        bound' = bound + fill 2
        key = fromLiteral key
        seed = fromLiteral seed

        samples = evalState seed (U64.uniform key bound bound')

    diff (toLiteral samples) (\x, y => any id [| x == y |]) (toLiteral bound)
    diff (toLiteral samples) (\x, y => any id [| x == y |]) (toLiteral bound')

  export covering
  uniformSeedIsUpdated : Property
  uniformSeedIsUpdated = withTests 20 . property $ do
    bound <- forAll (literal [10] nats)
    bound' <- forAll (literal [10] nats)
    key <- forAll (literal [] nats)
    seed <- forAll (literal [1] nats)

    let bound = fromLiteral bound
        bound' = fromLiteral bound'
        key = fromLiteral key
        seed = fromLiteral seed

        rng = U64.uniform key {shape=[10]} (broadcast bound) (broadcast bound')
        (seed', sample) = runState seed rng
        (seed'', sample') = runState seed' rng

    diff (toLiteral seed') (/=) (toLiteral seed)
    diff (toLiteral seed'') (/=) (toLiteral seed')
    diff (toLiteral sample') (/=) (toLiteral sample)

  export covering
  uniformIsReproducible : Property
  uniformIsReproducible = withTests 20 . property $ do
    bound <- forAll (literal [10] nats)
    bound' <- forAll (literal [10] nats)
    key <- forAll (literal [] nats)
    seed <- forAll (literal [1] nats)

    let bound = fromLiteral bound
        bound' = fromLiteral bound'
        key = fromLiteral key
        seed = fromLiteral seed

        rng = U64.uniform {shape=[10]} key (broadcast bound) (broadcast bound')
        sample = evalState seed rng
        sample' = evalState seed rng

    sample ===# sample'
-}

partial
normal : Property
normal = withTests 20 . property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let ksTest = do
        key <- tensor key
        seed <- tensor seed

        samples <- the (Ref $ Tensor [100, 100] F64) $ evalStateT seed (normal key)

        let normalCdf : {shape : _} -> Tensor shape F64 -> Ref $ Tensor shape F64
            normalCdf x = do (fill 1.0 + erf !(pure x / (sqrt !(fill 2.0)))) / fill 2.0

        iidKolmogorovSmirnov samples normalCdf

  diff (unsafeEval ksTest) (<) 0.02

partial
normalSeedIsUpdated : Property
normalSeedIsUpdated = withTests 20 . property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let everything = do
        key <- tensor key
        seed <- tensor seed
        let rng = normal key {shape=[10]}
        (seed', sample) <- runStateT seed rng
        (seed'', sample') <- runStateT seed' rng
        seeds <- concat 0 !(concat 0 seed seed') seed''
        samples <- concat 0 !(expand 0 sample) !(expand 0 sample')
        pure (seeds, samples)

      [seed, seed', seed''] = unsafeEval (do (seeds, _) <- everything; pure seeds)
      [sample, sample'] = unsafeEval (do (_, samples) <- everything; pure samples)

  diff seed' (/=) seed
  diff seed'' (/=) seed'
  diff sample' (/=) sample

partial
normalIsReproducible : Property
normalIsReproducible = withTests 20 . property $ do
  key <- forAll (literal [] nats)
  seed <- forAll (literal [1] nats)

  let [sample, sample'] = unsafeEval $ do
        key <- tensor key
        seed <- tensor seed

        let rng = normal {shape=[10]} key
        sample <- evalStateT seed rng
        sample' <- evalStateT seed rng
        concat 0 !(expand 0 sample) !(expand 0 sample')

  sample ==~ sample'

export partial
all : List (PropertyName, Property)
all = [
      ("uniform F64", F64.uniform)
    , ("uniform F64 for infinite and NaN bounds", uniformForNonFiniteBounds)
    , ("uniform F64 is not NaN for finite equal bounds", uniformForFiniteEqualBounds)
    , ("uniform F64 updates seed", F64.uniformSeedIsUpdated)
    , ("uniform F64 produces same samples for same seed", F64.uniformIsReproducible)
    , ("uniform U64", U64.uniform)
    -- , ("uniform U64 bounds are inclusive", uniformBoundsAreInclusive)
    -- , ("uniform U64 updates seed", U64.uniformSeedIsUpdated)
    -- , ("uniform U64 produces same samples for same seed", U64.uniformIsReproducible)
    , ("normal", normal)
    , ("normal updates seed", normalSeedIsUpdated)
    , ("normal produces same samples for same seed", normalIsReproducible)
  ]
