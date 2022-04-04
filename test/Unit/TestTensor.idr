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
module Unit.TestTensor

import Data.Nat
import Data.Vect
import System

import Literal

import Tensor

import Utils

covering
test_fromLiteral_toLiteral : Property
test_fromLiteral_toLiteral = property $ do
  shape <- forAll shapes

  x <- forAll (literal shape doubles)
  x === toLiteral (fromLiteral {dtype=F64} x)

  x <- forAll (literal shape ints)
  x === toLiteral (fromLiteral {dtype=S32} x)

  x <- forAll (literal shape bool)
  x === toLiteral (fromLiteral {dtype=PRED} x)

test_show_graph : IO ()
test_show_graph = do
  assert "show @{Graph} for scalar" $ show @{Graph {dtype=S32}} 1 == "S32[] fromLiteral"

  assert "show @{Graph} for scalar addition" $ show @{Graph {dtype=S32}} (1 + 2) ==
    """
    S32[] (+)
      S32[] fromLiteral
      S32[] fromLiteral
    """

  let x = fromLiteral {dtype=F64} [1.3, 2.0, -0.4]
  assert "show @{Graph} for vector F64" $ show @{Graph} x == "F64[3] fromLiteral"

  let x = fromLiteral {dtype=S32} [[0, 0, 0], [0, 0, 0]]
      y = fromLiteral [[0], [0], [0]]
  assert "show @{Graph} for differing argument shapes" $ show @{Graph} (x @@ y) ==
    """
    S32[2, 1] (@@)
      S32[2, 3] fromLiteral
      S32[3, 1] fromLiteral
    """

  let x = fromLiteral {dtype=S32} [0, 0]
      y = fromLiteral [[0, 0], [0, 0]]
  assert "show @{Graph} for non-trivial cond" $
    show @{Graph} (cond (fromLiteral True) (fromLiteral [0, 0] *) x diag y) ==
      """
      S32[2] cond
        PRED[] fromLiteral
        S32[2] (*)
          S32[2] fromLiteral
          S32[2] parameter
        S32[2] fromLiteral
        S32[2] diag
          S32[2, 2] parameter
        S32[2, 2] fromLiteral
      """

test_show_graphxla : IO ()
test_show_graphxla = do
  assert "show @{XLA} for scalar" $ show @{XLA {dtype=S32}} 1 == "constant, shape=[], metadata={:0}"

  assert "show @{XLA} for scalar addition" $ show @{XLA {dtype=S32}} (1 + 2) ==
    """
    add, shape=[], metadata={:0}
      constant, shape=[], metadata={:0}
      constant, shape=[], metadata={:0}
    """

  let x = fromLiteral {dtype=F64} [1.3, 2.0, -0.4]
  assert "show @{XLA} for vector F64" $ show @{XLA} x == "constant, shape=[3], metadata={:0}"

test_reshape : IO ()
test_reshape = do
  assertAll "reshape add dims scalar" $ reshape 3 == fromLiteral {dtype=S32} [3]

  let x = fromLiteral {dtype=S32} [3, 4, 5]
      flipped = fromLiteral [[3], [4], [5]]
  assertAll "reshape flip dims vector" $ reshape x == flipped

  let x = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]
      flipped = fromLiteral [[3, 4], [5, 6], [7, 8]]
  assertAll "reshape flip dims array" $ reshape x == flipped

  let with_extra_dim = fromLiteral {dtype=S32} [[[3, 4, 5]], [[6, 7, 8]]]
  assertAll "reshape add dimension array" $ reshape x == with_extra_dim

  let flattened = fromLiteral {dtype=S32} [3, 4, 5, 6, 7, 8]
  assertAll "reshape as flatten array" $ reshape x == flattened

test_slice : IO ()
test_slice = do
  let x = fromLiteral {dtype=S32} [3, 4, 5]
  assertAll "slice vector 0 0" $ slice 0 0 0 x == fromLiteral []
  assertAll "slice vector 0 1" $ slice 0 0 1 x == fromLiteral [3]
  assertAll "slice vector 0 2" $ slice 0 0 2 x == fromLiteral [3, 4]
  assertAll "slice vector 0 3" $ slice 0 0 3 x == fromLiteral [3, 4, 5]
  assertAll "slice vector 1 1" $ slice 0 1 1 x == fromLiteral []
  assertAll "slice vector 1 2" $ slice 0 1 2 x == fromLiteral [4]
  assertAll "slice vector 1 3" $ slice 0 1 3 x == fromLiteral [4, 5]
  assertAll "slice vector 2 2" $ slice 0 2 2 x == fromLiteral []
  assertAll "slice vector 2 2" $ slice 0 2 3 x == fromLiteral [5]

  let x = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]
  assertAll "slice array 0 0 1" $ slice 0 0 1 x == fromLiteral [[3, 4, 5]]
  assertAll "slice array 0 1 1" $ slice 0 1 1 x == fromLiteral []
  assertAll "slice array 1 2 2" $ slice 1 2 2 x == fromLiteral [[], []]
  assertAll "slice array 1 1 3" $ slice 1 1 3 x == fromLiteral [[4, 5], [7, 8]]

test_index : IO ()
test_index = do
  let x = fromLiteral {dtype=S32} [3, 4, 5]
  assertAll "index vector 0" $ index 0 0 x == fromLiteral 3
  assertAll "index vector 1" $ index 0 1 x == fromLiteral 4
  assertAll "index vector 2" $ index 0 2 x == fromLiteral 5

  let x = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]
  assertAll "index array 0 0" $ index 0 0 x == fromLiteral [3, 4, 5]
  assertAll "index array 0 1" $ index 0 1 x == fromLiteral [6, 7, 8]
  assertAll "index array 1 0" $ index 1 0 x == fromLiteral [3, 6]
  assertAll "index array 1 1" $ index 1 1 x == fromLiteral [4, 7]
  assertAll "index array 1 2" $ index 1 2 x == fromLiteral [5, 8]

test_split : IO ()
test_split = do
  let vector = fromLiteral {dtype=S32} [3, 4, 5]

  let (l, r) = split 0 0 vector
  assertAll "split vector 0 left" $ l == fromLiteral []
  assertAll "split vector 0 right" $ r == fromLiteral [3, 4, 5]

  let (l, r) = split 0 1 vector
  assertAll "split vector 1 left" $ l == fromLiteral [3]
  assertAll "split vector 1 right" $ r == fromLiteral [4, 5]

  let (l, r) = split 0 2 vector
  assertAll "split vector 2 left" $ l == fromLiteral [3, 4]
  assertAll "split vector 2 right" $ r == fromLiteral [5]

  let (l, r) = split 0 3 vector
  assertAll "split vector 3 left" $ l == fromLiteral [3, 4, 5]
  assertAll "split vector 3 right" $ r == fromLiteral []

  let arr = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]

  let (l, r) = split 0 0 arr
  assertAll "split array 0 0 left" $ l == fromLiteral []
  assertAll "split array 0 0 right" $ r == fromLiteral [[3, 4, 5], [6, 7, 8]]

  let (l, r) = split 0 1 arr
  assertAll "split array 0 1 left" $ l == fromLiteral [[3, 4, 5]]
  assertAll "split array 0 1 right" $ r == fromLiteral [[6, 7, 8]]

  let (l, r) = split 0 2 arr
  assertAll "split array 0 2 left" $ l == fromLiteral [[3, 4, 5], [6, 7, 8]]
  assertAll "split array 0 2 right" $ r == fromLiteral []

  let (l, r) = split 1 0 arr
  assertAll "split array 1 0 left" $ l == fromLiteral [[], []]
  assertAll "split array 1 0 right" $ r == fromLiteral [[3, 4, 5], [6, 7, 8]]

  let (l, r) = split 1 1 arr
  assertAll "split array 1 1 left" $ l == fromLiteral [[3], [6]]
  assertAll "split array 1 1 right" $ r == fromLiteral [[4, 5], [7, 8]]

  let (l, r) = split 1 2 arr
  assertAll "split array 1 2 left" $ l == fromLiteral [[3, 4], [6, 7]]
  assertAll "split array 1 2 right" $ r == fromLiteral [[5], [8]]

  let (l, r) = split 1 3 arr
  assertAll "split array 1 3 left" $ l == fromLiteral [[3, 4, 5], [6, 7, 8]]
  assertAll "split array 1 3 right" $ r == fromLiteral [[], []]

test_concat : IO ()
test_concat = do
  let vector = fromLiteral {dtype=S32} [3, 4, 5]

  let l = fromLiteral {shape=[0]} []
      r = fromLiteral [3, 4, 5]
  assertAll "concat vector" $ concat 0 l r == vector

  let l = fromLiteral [3]
      r = fromLiteral [4, 5]
  assertAll "concat vector" $ concat 0 l r == vector

  let l = fromLiteral [3, 4]
      r = fromLiteral [5]
  assertAll "concat vector" $ concat 0 l r == vector

  let l = fromLiteral [3, 4, 5]
      r = fromLiteral {shape=[0]} []
  assertAll "concat vector" $ concat 0 l r == vector

  let arr = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]

  let l = fromLiteral {shape=[0, 3]} []
      r = fromLiteral [[3, 4, 5], [6, 7, 8]]
  assertAll "concat array 0" $ concat 0 l r == arr

  let l = fromLiteral [[3, 4, 5]]
      r = fromLiteral [[6, 7, 8]]
  assertAll "concat array 0" $ concat 0 l r == arr

  let l = fromLiteral [[3, 4, 5], [6, 7, 8]]
      r = fromLiteral {shape=[0, 3]} []
  assertAll "concat array 0" $ concat 0 l r == arr

  let l = fromLiteral {shape=[2, 0]} [[], []]
      r = fromLiteral [[3, 4, 5], [6, 7, 8]]
  assertAll "concat array 1" $ concat 1 l r == arr

  let l = fromLiteral [[3], [6]]
      r = fromLiteral [[4, 5], [7, 8]]
  assertAll "concat array 1" $ concat 1 l r == arr

  let l = fromLiteral [[3, 4], [6, 7]]
      r = fromLiteral [[5], [8]]
  assertAll "concat array 1" $ concat 1 l r == arr

  let l = fromLiteral [[3, 4, 5], [6, 7, 8]]
      r = fromLiteral {shape=[2, 0]} [[], []]
  assertAll "concat array 1" $ concat 1 l r == arr

test_diag : IO ()
test_diag = do
  let x = fromLiteral {dtype=S32} []
  assertAll "diag empty" $ diag x == fromLiteral []

  let x = fromLiteral {dtype=S32} [[3]]
  assertAll "diag 1" $ diag x == fromLiteral [3]

  let x = fromLiteral {dtype=S32} [[1, 2], [3, 4]]
  assertAll "diag 2" $ diag x == fromLiteral [1, 4]

test_triangle : IO ()
test_triangle = do
  let x = fromLiteral {dtype=S32} []
  assertAll "triangle upper empty" $ triangle Upper x == fromLiteral []
  assertAll "triangle lower empty" $ triangle Lower x == fromLiteral []

  let x = fromLiteral {dtype=S32} [[3]]
  assertAll "triangle upper 1" $ triangle Upper x == fromLiteral [[3]]
  assertAll "triangle lower 1" $ triangle Lower x == fromLiteral [[3]]

  let x = fromLiteral {dtype=S32} [[1, 2], [3, 4]]
  assertAll "triangle upper 2" $ triangle Upper x == fromLiteral [[1, 2], [0, 4]]
  assertAll "triangle lower 2" $ triangle Lower x == fromLiteral [[1, 0], [3, 4]]

  let x = fromLiteral {dtype=S32} [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  assertAll "triangle upper 3" $ triangle Upper x == fromLiteral [[1, 2, 3], [0, 5, 6], [0, 0, 9]]
  assertAll "triangle lower 3" $ triangle Lower x == fromLiteral [[1, 0, 0], [4, 5, 0], [7, 8, 9]]

test_identity : IO ()
test_identity = do
  assertAll "identity 0 S32" $ identity == fromLiteral {dtype=S32} []
  assertAll "identity 1 S32" $ identity == fromLiteral {dtype=S32} [[1]]
  assertAll "identity 2 S32" $ identity == fromLiteral {dtype=S32} [[1, 0], [0, 1]]
  assertAll "identity 4 S32" $
    identity == fromLiteral {dtype=S32} [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

  assertAll "identity 0 F64" $ identity == fromLiteral {dtype=F64} []
  assertAll "identity 1 F64" $ identity == fromLiteral {dtype=F64} [[1.0]]
  assertAll "identity 2 F64" $ identity == fromLiteral {dtype=F64} [[1.0, 0.0], [0.0, 1.0]]
  assertAll "identity 4 F64" $
    identity == fromLiteral {dtype=F64} [
        [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]
      ]

test_expand : IO ()
test_expand = do
  assertAll "expand add dims scalar" $ expand 0 3 == fromLiteral {dtype=S32} [3]

  let x = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]
      with_extra_dim = fromLiteral [[[3, 4, 5]], [[6, 7, 8]]]
  assertAll "expand add dimension array" $ expand 1 x == with_extra_dim

test_broadcast : IO ()
test_broadcast = do
  assertAll "broadcast scalar to itself" $ broadcast {to=[]} {dtype=S32} 7 == 7

  assertAll "broadcast scalar to rank 1" $ broadcast {to=[1]} {dtype=S32} 7 == fromLiteral [7]

  assertAll "broadcast scalar to rank 2" $
    broadcast {to=[2, 3]} 7 == fromLiteral [[7, 7, 7], [7, 7, 7]]

  assertAll "broadcast scalar to rank 3" $
    broadcast {to=[1, 1, 1]} {dtype=S32} 7 == fromLiteral [[[7]]]

  assertAll "broadcast rank 1 to empty" $ broadcast {to=[0]} 7 == fromLiteral []

  let x = fromLiteral {dtype=S32} [7]
  assertAll "broadcast rank 1 to itself" $ broadcast {to=[1]} x == fromLiteral [7]

  let x = fromLiteral {dtype=S32} [7]
  assertAll "broadcast rank 1 to larger rank 1" $ broadcast {to=[3]} x == fromLiteral [7, 7, 7]

  let x = fromLiteral {dtype=S32} [7]
  assertAll "broadcast rank 1 to rank 2" $
    broadcast {to=[2, 3]} x == fromLiteral [[7, 7, 7], [7, 7, 7]]

  let x = fromLiteral {dtype=S32} [5, 7]
  assertAll "broadcast rank 1 to empty" $ broadcast {to=[2, 0]} x == fromLiteral [[], []]

  let x = fromLiteral {dtype=S32} [5, 7]
  assertAll "broadcast rank 1 to rank 2" $
    broadcast {to=[3, 2]} x == fromLiteral [[5, 7], [5, 7], [5, 7]]

  let x = fromLiteral {dtype=S32} [[2, 3, 5], [7, 11, 13]]
  assertAll "broadcast rank 2 to itself" $
    broadcast {to=[2, 3]} x == fromLiteral [[2, 3, 5], [7, 11, 13]]

  let x = fromLiteral {dtype=S32} [[2, 3, 5], [7, 11, 13]]
  assertAll "broadcast rank 2 to rank 2 empty" $ broadcast {to=[2, 0]} x == fromLiteral [[], []]

  let x = fromLiteral {dtype=S32} [[2, 3, 5], [7, 11, 13]]
  assertAll "broadcast rank 2 to empty" $ broadcast {to=[0, 3]} x == fromLiteral []

  let x = fromLiteral {dtype=S32} [[2, 3, 5], [7, 11, 13]]
      expected = fromLiteral [[[2, 3, 5], [7, 11, 13]], [[2, 3, 5], [7, 11, 13]]]
  assertAll "broadcast rank 2 to rank 3" $ broadcast {to=[2, 2, 3]} x == expected

  let x = fromLiteral {dtype=S32} [[[2, 3, 5]], [[7, 11, 13]]]
      expected = fromLiteral [
        [
          [[2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5]],
          [[7, 11, 13], [7, 11, 13], [7, 11, 13], [7, 11, 13], [7, 11, 13]]
        ],
        [
          [[2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5]],
          [[7, 11, 13], [7, 11, 13], [7, 11, 13], [7, 11, 13], [7, 11, 13]]
        ]
      ]
  assertAll "broadcast rank 3 to rank 4" $ broadcast {to=[2, 2, 5, 3]} x == expected

test_dimbroadcastable : List (a ** b ** DimBroadcastable a b)
test_dimbroadcastable = [
  (0 ** 0 ** Same),
  (1 ** 1 ** Same),
  (3 ** 3 ** Same),
  (1 ** 0 ** Stack),
  (1 ** 1 ** Stack),
  (1 ** 3 ** Stack),
  (0 ** 0 ** Zero),
  (1 ** 0 ** Zero),
  (3 ** 0 ** Zero)
]

test_broadcastable : List (from : Shape ** to : Shape ** Broadcastable from to)
test_broadcastable = [
  ([] ** [] ** Same),
  ([3, 2, 5] ** [3, 2, 5] ** Same),
  ([] ** [3, 2, 5] ** Nest $ Nest $ Nest Same),
  ([3, 1, 5] ** [3, 7, 5] ** Match $ Match Same),
  ([3, 2, 5] ** [1, 3, 2, 5] ** Nest Same),
  ([3, 2, 5] ** [7, 3, 2, 5] ** Nest Same)
]

test_broadcastable_cannot_reduce_rank0 : Broadcastable [5] [] -> Void
test_broadcastable_cannot_reduce_rank0 _ impossible

test_broadcastable_cannot_reduce_rank1 : Broadcastable [3, 2, 5] [] -> Void
test_broadcastable_cannot_reduce_rank1 _ impossible

test_broadcastable_cannot_stack_dimension_gt_one : Broadcastable [3, 2] [3, 7] -> Void
test_broadcastable_cannot_stack_dimension_gt_one (Match Same) impossible
test_broadcastable_cannot_stack_dimension_gt_one (Nest Same) impossible

test_squeeze : IO ()
test_squeeze = do
  let x = fromLiteral {dtype=S32} [[3]]
      squeezed = 3
  assertAll "squeeze can flatten only ones" $ squeeze x == squeezed

  let x = fromLiteral {dtype=S32} [[[3, 4, 5]], [[6, 7, 8]]]
  assertAll "squeeze can no-op" $ squeeze x == x

  let squeezed = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]
  assertAll "squeeze can remove dim from array" $ squeeze x == squeezed

  let x = fill {shape=[1, 3, 1, 1, 2, 5, 1]} {dtype=S32} 0
  assertAll "squeeze can remove many dims from array" $
    squeeze x == fill {shape=[3, 2, 5]} {dtype=S32} 0

test_squeezable_cannot_remove_non_ones : Squeezable [1, 2] [] -> Void
test_squeezable_cannot_remove_non_ones (Nest _) impossible

test_T : IO ()
test_T = do
  assertAll "(.T) for empty array" $ (fromLiteral {dtype=S32} []).T == fromLiteral []
  assertAll "(.T) for single element" $ (fromLiteral {dtype=S32} [[3]]).T == fromLiteral [[3]]

  let x = fromLiteral {dtype=S32} [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
      expected = fromLiteral [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
  assertAll "(.T)" $ x.T == expected

test_map : IO ()
test_map = do
  assertAll "map with function with reused arguments" $ map {a=S32} (\x => x + x) 1 == 2

  assertAll "map with function with unused arguments" $ map {a=S32} (\_ => 2) 1 == 2

  assertAll "map with map in function" $ map {a=S32} (map (+ 1)) 1 == 2

  let x = fromLiteral {dtype=S32} [[1, 15, 5], [-1, 7, 6]]
  assertAll "map for S32 array" $ map abs x == abs x

  let x = fromLiteral {dtype=F64} [[1.0, 2.5, 0.0], [-0.8, -0.1, 5.0]]
  assertAll "map for F64 array" $
    map (1.0 /) x == fromLiteral [[1.0, 0.4, inf], [-1.25, -10.0, 0.2]]

  sequence_ $ do
    x <- ints
    let x = fromLiteral {dtype=S32} x
    pure $ assertAll "map for S32 scalar" $ map (+ 1) x == x + 1

  sequence_ $ do
    x <- doubles
    let x = fromLiteral {dtype=F64} x
    pure $ assertAll "map for F64 scalar" $ sufficientlyEq (map (+ 1.2) x) (x + 1.2)

test_map2 : IO ()
test_map2 = do
  assertAll "map2 with function with reused arguments" $ map2 (\x, y => x + x + y + y) 1 2 == 6

  let l = fromLiteral {dtype=S32} [[1, 2, 3], [-1, -2, -3]]
      r = fromLiteral {dtype=S32} [[1, 4, 2], [-2, -1, -3]]
  assertAll "map2 for S32 array" $ map2 (+) l r == (l + r)

  let l = fromLiteral {dtype=F64} [[1.1, 2.2, 3.3], [-1.1, -2.2, -3.3]]
      r = fromLiteral {dtype=F64} [[1.1, 4.4, 2.2], [-2.2, -1.1, -3.3]]
  assertAll "map2 for F64 matrix" $ sufficientlyEq (map2 (+) l r) (l + r)

  sequence_ $ do
    l <- doubles
    r <- doubles
    let l' = fromLiteral {dtype=F64} l
        r' = fromLiteral {dtype=F64} r
    pure $ assertAll "map2 for F64 scalars" $ sufficientlyEq (map2 (+) l' r') (l' + r')

  sequence_ $ do
    l <- doubles
    let l' = fromLiteral {dtype=F64} l
    pure $ assertAll "map2 for F64 scalars with repeated argument" $
      sufficientlyEq (map2 (+) l' l') (l' + l')

test_reduce : IO ()
test_reduce = do
  let x = fromLiteral {dtype=F64} [[1.1, 2.2, 3.3], [-1.1, -2.2, -3.3]]
  assertAll "reduce for F64 array" $ sufficientlyEq (reduce @{Sum} 1 x) (fromLiteral [6.6, -6.6])

  let x = fromLiteral {dtype=F64} [[1.1, 2.2, 3.3], [-1.1, -2.2, -3.3]]
  assertAll "reduce for F64 array" $
    sufficientlyEq (reduce @{Sum} 0 x) (fromLiteral [0.0, 0.0, 0.0])

  let x = fromLiteral {dtype=PRED} [[True, False, True], [True, False, False]]
  assertAll "reduce for PRED array" $ reduce @{All} 1 x == fromLiteral [False, False]

covering
test_equality : Property
test_equality = property $ do
  shape <- forAll shapes

  let doubles = literal shape doubles
  [x, y] <- forAll (np [doubles, doubles])
  let x' = fromLiteral {dtype=F64} x
      y' = fromLiteral {dtype=F64} y
  [| x == y |] === toLiteral (x' == y')

  let ints = literal shape ints
  [x, y] <- forAll (np [ints, ints])
  let x' = fromLiteral {dtype=S32} x
      y' = fromLiteral {dtype=S32} y
  [| x == y |] === toLiteral (x' == y')

  let bools = literal shape bool
  [x, y] <- forAll (np [bools, bools])
  let x' = fromLiteral {dtype=PRED} x
      y' = fromLiteral {dtype=PRED} y
  [| x == y |] === toLiteral (x' == y')

covering
test_inequality : Property
test_inequality = property $ do
  shape <- forAll shapes

  let doubles = literal shape doubles
  [x, y] <- forAll (np [doubles, doubles])
  let x' = fromLiteral {dtype=F64} x
      y' = fromLiteral {dtype=F64} y
  [| x /= y |] === toLiteral (x' /= y')

  let ints = literal shape ints
  [x, y] <- forAll (np [ints, ints])
  let x' = fromLiteral {dtype=S32} x
      y' = fromLiteral {dtype=S32} y
  [| x /= y |] === toLiteral (x' /= y')

  let bools = literal shape bool
  [x, y] <- forAll (np [bools, bools])
  let x' = fromLiteral {dtype=PRED} x
      y' = fromLiteral {dtype=PRED} y
  [| x /= y |] === toLiteral (x' /= y')

covering
test_less_than : Property
test_less_than = property $ do
  shape <- forAll shapes

  let doubles = literal shape doubles
  [x, y] <- forAll (np [doubles, doubles])
  let x' = fromLiteral {dtype=F64} x
      y' = fromLiteral {dtype=F64} y
  [| x < y |] === toLiteral (x' < y')

  let ints = literal shape ints
  [x, y] <- forAll (np [ints, ints])
  let x' = fromLiteral {dtype=S32} x
      y' = fromLiteral {dtype=S32} y
  [| x < y |] === toLiteral (x' < y')

covering
test_less_than_equal : Property
test_less_than_equal = property $ do
  shape <- forAll shapes

  let doubles = literal shape doubles
  [x, y] <- forAll (np [doubles, doubles])
  let x' = fromLiteral {dtype=F64} x
      y' = fromLiteral {dtype=F64} y
  [| x <= y |] === toLiteral (x' <= y')

  let ints = literal shape ints
  [x, y] <- forAll (np [ints, ints])
  let x' = fromLiteral {dtype=S32} x
      y' = fromLiteral {dtype=S32} y
  [| x <= y |] === toLiteral (x' <= y')

covering
test_greater_than : Property
test_greater_than = property $ do
  shape <- forAll shapes

  let doubles = literal shape doubles
  [x, y] <- forAll (np [doubles, doubles])
  let x' = fromLiteral {dtype=F64} x
      y' = fromLiteral {dtype=F64} y
  [| x > y |] === toLiteral (x' > y')

  let ints = literal shape ints
  [x, y] <- forAll (np [ints, ints])
  let x' = fromLiteral {dtype=S32} x
      y' = fromLiteral {dtype=S32} y
  [| x > y |] === toLiteral (x' > y')

covering
test_greater_than_equal : Property
test_greater_than_equal = property $ do
  shape <- forAll shapes

  let doubles = literal shape doubles
  [x, y] <- forAll (np [doubles, doubles])
  let x' = fromLiteral {dtype=F64} x
      y' = fromLiteral {dtype=F64} y
  [| x >= y |] === toLiteral (x' >= y')

  let ints = literal shape ints
  [x, y] <- forAll (np [ints, ints])
  let x' = fromLiteral {dtype=S32} x
      y' = fromLiteral {dtype=S32} y
  [| x >= y |] === toLiteral (x' >= y')

namespace Vector
  export
  test_dot : IO ()
  test_dot = do
    let l = fromLiteral {dtype=S32} [-2, 0, 1]
        r = fromLiteral {dtype=S32} [3, 1, 2]
    assertAll "vector dot" $ l @@ r == -4

namespace Matrix
  export
  test_dot : IO ()
  test_dot = do
    let l = fromLiteral {dtype=S32} [[-2, 0, 1], [1, 3, 4]]
        r = fromLiteral {dtype=S32} [3, 3, -1]
    assertAll "matrix dot vector" $ l @@ r == fromLiteral [-7, 8]

    let l = fromLiteral {dtype=S32} [[-2, 0, 1], [1, 3, 4]]
        r = fromLiteral {dtype=S32} [[3, -1], [3, 2], [-1, -4]]
    assertAll "matrix dot matrix" $ l @@ r == fromLiteral [[ -7,  -2], [  8, -11]]

covering
test_addition : Property
test_addition = property $ do
  shape <- forAll shapes

  let doubles = literal shape doubles
  [x, y] <- forAll (np [doubles, doubles])
  let x' = fromLiteral {dtype=F64} x
      y' = fromLiteral {dtype=F64} y
  [| x + y |] === toLiteral (x' + y')

  let ints = literal shape ints
  [x, y] <- forAll (np [ints, ints])
  let x' = fromLiteral {dtype=S32} x
      y' = fromLiteral {dtype=S32} y
  [| x + y |] === toLiteral (x' + y')

covering
test_subtraction : Property
test_subtraction = property $ do
  shape <- forAll shapes

  let doubles = literal shape doubles
  [x, y] <- forAll (np [doubles, doubles])
  let x' = fromLiteral {dtype=F64} x
      y' = fromLiteral {dtype=F64} y
  [| x - y |] === toLiteral (x' - y')

  let ints = literal shape ints
  [x, y] <- forAll (np [ints, ints])
  let x' = fromLiteral {dtype=S32} x
      y' = fromLiteral {dtype=S32} y
  [| x - y |] === toLiteral (x' - y')

covering
test_multiplication : Property
test_multiplication = property $ do
  shape <- forAll shapes

  let doubles = literal shape doubles
  [x, y] <- forAll (np [doubles, doubles])
  let x' = fromLiteral {dtype=F64} x
      y' = fromLiteral {dtype=F64} y
  [| x * y |] === toLiteral (x' * y')

  let ints = literal shape ints
  [x, y] <- forAll (np [ints, ints])
  let x' = fromLiteral {dtype=S32} x
      y' = fromLiteral {dtype=S32} y
  [| x * y |] === toLiteral (x' * y')

covering
test_division : Property
test_division = property $ do
  shape <- forAll shapes
  let doubles = literal shape doubles
  [x, y] <- forAll (np [doubles, doubles])
  let x' = fromLiteral {dtype=F64} x
      y' = fromLiteral {dtype=F64} y
  [| x / y |] === toLiteral (x' / y')

covering
test_Sum : Property
test_Sum = property $ do
  shape <- forAll shapes

  x <- forAll (literal shape doubles)
  let x' = fromLiteral {dtype=F64} x
      right = (<+>) @{Sum} x' (neutral @{Sum})
      left = (<+>) @{Sum} (neutral @{Sum}) x'
  toLiteral right === x
  toLiteral left === x

  x <- forAll (literal shape ints)
  let x' = fromLiteral {dtype=S32} x
      right = (<+>) @{Sum} x' (neutral @{Sum})
      left = (<+>) @{Sum} (neutral @{Sum}) x'
  toLiteral right === x
  toLiteral left === x

covering
test_scalar_multiplication : Property
test_scalar_multiplication = property $ do
  shape <- forAll shapes
  case shape of
    [] => success
    (d :: ds) => do
      [lit, scalar] <- forAll (np [literal (d :: ds) doubles, doubles])
      let lit' = fromLiteral {dtype=F64} lit
          scalar' = fromLiteral {dtype=F64} (Scalar scalar)
      map (scalar *) lit === toLiteral (scalar' * lit')

covering
test_Prod : Property
test_Prod = property $ do
  shape <- forAll shapes

  x <- forAll (literal shape doubles)
  let x' = fromLiteral {dtype=F64} x
      right = (<+>) @{Prod} x' (neutral @{Prod})
      left = (<+>) @{Prod} (neutral @{Prod}) x'
  toLiteral right === x
  toLiteral left === x

  x <- forAll (literal shape ints)
  let x' = fromLiteral {dtype=S32} x
      right = (<+>) @{Prod} x' (neutral @{Prod})
      left = (<+>) @{Prod} (neutral @{Prod}) x'
  toLiteral right === x
  toLiteral left === x

covering
test_and : Property
test_and = property $ do
  shape <- forAll shapes
  let bools = literal shape bool
  [x, y] <- forAll (np [bools, bools])
  let x' = fromLiteral {dtype=PRED} x
      y' = fromLiteral {dtype=PRED} y
  [| x `and` y |] === toLiteral (x' && y')

  where

  and : Bool -> Bool -> Bool
  and x y = x && y

test_All : IO ()
test_All = do
  let x = fromLiteral [[True, True], [False, False]]
  assertAll "All neutral is neutral right" $ (<+>) @{All} x (neutral @{All}) == x
  assertAll "All neutral is neutral left" $ (<+>) @{All} (neutral @{All}) x == x

covering
test_or : Property
test_or = property $ do
  shape <- forAll shapes
  let bools = literal shape bool
  [x, y] <- forAll (np [bools, bools])
  let x' = fromLiteral {dtype=PRED} x
      y' = fromLiteral {dtype=PRED} y
  [| x `or` y |] === toLiteral (x' || y')

  where

  or : Bool -> Bool -> Bool
  or x y = x || y

test_Any : IO ()
test_Any = do
  let x = fromLiteral [[True, True], [False, False]]
  assertAll "Any neutral is neutral right" $ (<+>) @{Any} x (neutral @{Any}) == x
  assertAll "Any neutral is neutral left" $ (<+>) @{Any} (neutral @{Any}) x == x

covering
test_not : Property
test_not = property $ do
  shape <- forAll shapes
  x <- forAll (literal shape bool)
  let x' = fromLiteral {dtype=PRED} x
  [| not x |] === toLiteral (not x')

test_select : IO ()
test_select = do
  let onTrue = fromLiteral {dtype=S32} 1
      onFalse = fromLiteral 0
  assertAll "select for scalar True" $ select (fromLiteral True) onTrue onFalse == onTrue
  assertAll "select for scalar False" $ select (fromLiteral False) onTrue onFalse == onFalse

  let pred = fromLiteral [[False, True, True], [True, False, False]]
      onTrue = fromLiteral {dtype=S32} [[0, 1, 2], [3, 4, 5]]
      onFalse = fromLiteral [[6, 7, 8], [9, 10, 11]]
      expected = fromLiteral [[6, 1, 2], [3, 10, 11]]
  assertAll "select for array" $ select pred onTrue onFalse == expected

test_cond : IO ()
test_cond = do
  let x = fromLiteral {dtype=S32} 1
      y = fromLiteral {dtype=S32} 3
  assertAll "cond with function with reused arguments (truthy)" $
    cond (fromLiteral True) (\z => z + z) x (\z => z * z) y == 2
  assertAll "cond with function with reused arguments (falsy)" $
    cond (fromLiteral False) (\z => z + z) x (\z => z * z) y == 9

  let x = fromLiteral {dtype=S32} 0
  assertAll "cond for trivial truthy" $
    cond (fromLiteral True) (+ 1) x (\x => x - 1) x == 1

  let x = fromLiteral {dtype=S32} 0
  assertAll "cond for trivial falsy" $
    cond (fromLiteral False) (+ 1) x (\x => x - 1) x == -1

  let x = fromLiteral {dtype=S32} [2, 3]
      y = fromLiteral [[6, 7], [8, 9]]
  assertAll "cond for non-trivial truthy" $
    cond (fromLiteral True) (fromLiteral 5 *) x diag y == fromLiteral [10, 15]

  let x = fromLiteral {dtype=S32} [2, 3]
      y = fromLiteral [[6, 7], [8, 9]]
  assertAll "cond for non-trivial falsy" $
    cond (fromLiteral False) (fromLiteral 5 *) x diag y == fromLiteral [6, 9]

covering
test_scalar_division : Property
test_scalar_division = property $ do
  shape <- forAll shapes
  case shape of
    [] => success
    (d :: ds) => do
      [lit, scalar] <- forAll (np [literal (d :: ds) doubles, doubles])
      let lit' = fromLiteral {dtype=F64} lit
          scalar' = fromLiteral {dtype=F64} (Scalar scalar)
      map (/ scalar) lit === toLiteral (lit' / scalar')

export covering
test_pow : Property
test_pow = property $ do
  shape <- forAll shapes
  let doubles = literal shape doubles
  [x, y] <- forAll (np [doubles, doubles])
  let x' = fromLiteral {dtype=F64} x
      y' = fromLiteral {dtype=F64} y
  [| pow x y |] === toLiteral (x' ^ y')

namespace S32
  export covering
  testElementwiseUnary :
    (Int -> Int) ->
    (forall shape . Tensor shape S32 -> Tensor shape S32) ->
    Property
  testElementwiseUnary fInt fTensor = property $ do
    shape <- forAll shapes
    x <- forAll (literal shape ints)
    let x' = fromLiteral x
    [| fInt x |] === toLiteral (fTensor x')

namespace F64
  export covering
  testElementwiseUnary :
    (Double -> Double) ->
    (forall shape . Tensor shape F64 -> Tensor shape F64) ->
    Property
  testElementwiseUnary fDouble fTensor = property $ do
    shape <- forAll shapes
    x <- forAll (literal shape doubles)
    let x' = fromLiteral x
    [| fDouble x |] === toLiteral (fTensor x')

tanh' : Double -> Double
tanh' x =
  if x == -inf then -1.0
  else if x == inf then 1.0
  else tanh x

covering
testElementwiseUnaryCases : List (PropertyName, Property)
testElementwiseUnaryCases = [
    ("negate", S32.testElementwiseUnary negate negate),
    ("negate", F64.testElementwiseUnary negate negate),
    ("abs", S32.testElementwiseUnary abs abs),
    ("abs", F64.testElementwiseUnary abs abs),
    ("exp", F64.testElementwiseUnary exp exp),
    ("ceil", F64.testElementwiseUnary ceiling ceil),
    ("floor", F64.testElementwiseUnary floor floor),
    ("log", F64.testElementwiseUnary log log),
    ("logistic", F64.testElementwiseUnary (\x => 1 / (1 + exp (-x))) logistic),
    ("sin", F64.testElementwiseUnary sin sin),
    ("cos", F64.testElementwiseUnary cos cos),
    ("tanh", F64.testElementwiseUnary tanh' tanh),
    ("sqrt", F64.testElementwiseUnary sqrt sqrt)
  ]

test_erf : IO ()
test_erf = do
  let x = fromLiteral [-1.5, -0.5, 0.5, 1.5]
      expected = fromLiteral [-0.96610516, -0.5204998, 0.5204998, 0.9661051]
  assertAll "erf agrees with tfp Normal" $ sufficientlyEq {tol=0.000001} (erf x) expected

min' : Double -> Double -> Double
min' x y = if (x /= x) then x else if (y /= y) then y else min x y

covering
test_min : Property
test_min = property $ do
  shape <- forAll shapes

  let doubles = literal shape doubles
  [x, y] <- forAll (np [doubles, doubles])
  let x' = fromLiteral {dtype=F64} x
      y' = fromLiteral {dtype=F64} y
  [| min' x y |] === toLiteral (min x' y')

  let ints = literal shape ints
  [x, y] <- forAll (np [ints, ints])
  let x' = fromLiteral {dtype=S32} x
      y' = fromLiteral {dtype=S32} y
  [| min x y |] === toLiteral (min x' y')

test_Min : IO ()
test_Min = do
  let x = fromLiteral {dtype=F64} [[1.1, 2.1, -2.0], [-1.3, -1.0, 1.0]]
  assertAll "Min neutral is neutral right" $ (<+>) @{Min} x (neutral @{Min}) == x
  assertAll "Min neutral is neutral left" $ (<+>) @{Min} (neutral @{Min}) x == x

max' : Double -> Double -> Double
max' x y = if (x /= x) then x else if (y /= y) then y else max x y

covering
test_max : Property
test_max = property $ do
  shape <- forAll shapes

  let doubles = literal shape doubles
  [x, y] <- forAll (np [doubles, doubles])
  let x' = fromLiteral {dtype=F64} x
      y' = fromLiteral {dtype=F64} y
  [| max' x y |] === toLiteral (max x' y')

  let ints = literal shape ints
  [x, y] <- forAll (np [ints, ints])
  let x' = fromLiteral {dtype=S32} x
      y' = fromLiteral {dtype=S32} y
  [| max x y |] === toLiteral (max x' y')

test_Max : IO ()
test_Max = do
  let x = fromLiteral {dtype=F64} [[1.1, 2.1, -2.0], [-1.3, -1.0, 1.0]]
  assertAll "Max neutral is neutral right" $ (<+>) @{Max} x (neutral @{Max}) == x
  assertAll "Max neutral is neutral left" $ (<+>) @{Max} (neutral @{Max}) x == x

test_cholesky : IO ()
test_cholesky = do
  let x = fromLiteral [[1.0, 0.0], [2.0, 0.0]]
      expected = fromLiteral [[nan, 0], [nan, nan]]
  assertAll "cholesky zero determinant" $ sufficientlyEq (cholesky x) expected

  -- example generated with tensorflow
  let x = fromLiteral [
              [ 2.236123  ,  0.70387983,  2.8447943 ],
              [ 0.7059226 ,  2.661426  , -0.8714733 ],
              [ 1.3730898 ,  1.4064665 ,  2.7474475 ]
            ]
      expected = fromLiteral [
              [1.4953672 , 0.0       , 0.0       ],
              [0.47207308, 1.5615932 , 0.0       ],
              [0.9182292 , 0.6230785 , 1.2312902 ]
            ]
  assertAll "cholesky" $ sufficientlyEq {tol=0.000001} (cholesky x) expected

test_triangularsolve : IO ()
test_triangularsolve = do
  let a = fromLiteral [
              [0.8578532 , 0.0       , 0.0       ],
              [0.2481904 , 0.9885198 , 0.0       ],
              [0.59390426, 0.14998078, 0.19468737]
            ]
      b = fromLiteral [
              [0.45312142, 0.37276268],
              [0.9210588 , 0.00647926],
              [0.7890165 , 0.77121615]
            ]
      actual = a |\ b
      expected = fromLiteral [
                    [ 0.52820396,  0.43452972],
                    [ 0.79913783, -0.10254406],
                    [ 1.8257918 ,  2.7147462 ]
                  ]
  assertAll "(|\) result" $ sufficientlyEq {tol=0.000001} actual expected
  assertAll "(|\) is invertible with (@@)" $ sufficientlyEq {tol=0.000001} (a @@ actual) b

  let actual = a.T \| b
      expected = fromLiteral [
                    [-2.3692384 , -2.135952  ],
                    [ 0.31686386, -0.594465  ],
                    [ 4.0527363 ,  3.9613056 ]
                  ]
  assertAll "(\|) result" $ sufficientlyEq {tol=0.000001} actual expected
  assertAll "(\|) is invertible with (@@)" $ sufficientlyEq {tol=0.000001} (a.T @@ actual) b

  let a = fromLiteral [[1.0, 2.0], [3.0, 4.0]]
      a_lt = fromLiteral [[1.0, 0.0], [3.0, 4.0]]
      b = fromLiteral [5.0, 6.0]
  assertAll "(|\) upper triangular elements are ignored" $ sufficientlyEq (a |\ b) (a_lt |\ b)

  let a_ut = fromLiteral [[1.0, 2.0], [0.0, 4.0]]
  assertAll "(\|) lower triangular elements are ignored" $ sufficientlyEq (a \| b) (a_ut \| b)

test_trace : IO ()
test_trace = do
  assertAll "trace" $ trace (fromLiteral {dtype=S32} [[-1, 5], [1, 4]]) == 3

export covering
test : IO Bool
test = checkGroup $ MkGroup "Tensor" $ [
    ("test_fromLiteral_toLiteral", test_fromLiteral_toLiteral),
    ("test_equality", test_equality),
    ("test_inequality", test_inequality),
    ("test_less_than", test_less_than),
    ("test_less_than_equal", test_less_than_equal),
    ("test_greater_than", test_greater_than),
    ("test_greater_than_equal", test_greater_than_equal)
  ] ++ testElementwiseUnaryCases ++ [
    ("test_addition", test_addition),
    ("test_not", test_not),
    ("test_and", test_and),
    ("test_or", test_or),
    ("test_subtraction", test_subtraction),
    ("test_multiplication", test_multiplication),
    ("test_scalar_multiplication", test_scalar_multiplication),
    ("test_division", test_division),
    ("test_scalar_division", test_scalar_division),
    ("test_min", test_min),
    ("test_max", test_max),
    ("test_pow", test_pow),
    ("Sum", test_Sum),
    ("Prod", test_Prod)
  ]

export
test' : IO ()
test' = do
  -- test_show_graph
  -- test_show_graphxla
  test_reshape
  test_slice
  test_index
  test_split
  test_concat
  test_diag
  test_triangle
  test_identity
  test_expand
  test_broadcast
  test_squeeze
  test_T
  -- test_map
  -- test_map2
  -- test_reduce
  Vector.test_dot
  Matrix.test_dot
  -- test_All
  -- test_Any
  test_select
  test_cond
  test_erf
  -- test_Min
  -- test_Max
  test_cholesky
  test_triangularsolve
  test_trace
