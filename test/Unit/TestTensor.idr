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

import Utils.Example
import Utils.Property

covering
test_fromLiteral_toLiteral : Property
test_fromLiteral_toLiteral = property $ do
  shape <- forAll shapes

  x <- forAll (literal shape doubles)
  x ==~ toLiteral (fromLiteral {dtype=F64} x)

  x <- forAll (literal shape ints)
  x === toLiteral (fromLiteral {dtype=S32} x)

  x <- forAll (literal shape bool)
  x === toLiteral (fromLiteral {dtype=PRED} x)

covering
test_show_graph : Property
test_show_graph = property $ do
  shape <- forAll shapes

  x <- forAll (literal shape ints)
  let x = fromLiteral {dtype=S32} x
  show @{Graph} x === "S32\{show shape} fromLiteral"

  x <- forAll (literal shape doubles)
  let x = fromLiteral {dtype=F64} x
  show @{Graph} x === "F64\{show shape} fromLiteral"

  let ints = literal shape ints
  [x, y] <- forAll (np [ints, ints])
  let x = fromLiteral {dtype=S32} x
      y = fromLiteral {dtype=S32} y
  show @{Graph {dtype=S32}} (x + y) ===
    """
    S32\{show shape} (+)
      S32\{show shape} fromLiteral
      S32\{show shape} fromLiteral
    """

test_show_graph' : Property
test_show_graph' = withTests 1 $ property $ do
  let x = fromLiteral {dtype=S32} [[0, 0, 0], [0, 0, 0]]
      y = fromLiteral [[0], [0], [0]]
  show @{Graph} (x @@ y) ===
    """
    S32[2, 1] (@@)
      S32[2, 3] fromLiteral
      S32[3, 1] fromLiteral
    """

  let x = fromLiteral {dtype=S32} [0, 0]
      y = fromLiteral [[0, 0], [0, 0]]
  show @{Graph} (cond (fromLiteral True) (fromLiteral [0, 0] *) x diag y) ===
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

test_show_xla : Property
test_show_xla = withTests 1 $ property $ do
  show @{XLA {dtype=S32}} 1 === "constant, shape=[], metadata={:0}"

  show @{XLA {dtype=S32}} (1 + 2) ===
    """
    add, shape=[], metadata={:0}
      constant, shape=[], metadata={:0}
      constant, shape=[], metadata={:0}
    """

  let x = fromLiteral {dtype=F64} [1.3, 2.0, -0.4]
  show @{XLA} x === "constant, shape=[3], metadata={:0}"

test_reshape : Property
test_reshape = withTests 1 $ property $ do
  reshape 3 ===? fromLiteral {dtype=S32} [3]

  let x = fromLiteral {dtype=S32} [3, 4, 5]
      flipped = fromLiteral [[3], [4], [5]]
  reshape x ===? flipped

  let x = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]
      flipped = fromLiteral [[3, 4], [5, 6], [7, 8]]
  reshape x ===? flipped

  let with_extra_dim = fromLiteral {dtype=S32} [[[3, 4, 5]], [[6, 7, 8]]]
  reshape x ===? with_extra_dim

  let flattened = fromLiteral {dtype=S32} [3, 4, 5, 6, 7, 8]
  reshape x ===? flattened

test_slice : Property
test_slice = withTests 1 $ property $ do
  let x = fromLiteral {dtype=S32} [3, 4, 5]
  slice 0 0 0 x ===? fromLiteral []
  slice 0 0 1 x ===? fromLiteral [3]
  slice 0 0 2 x ===? fromLiteral [3, 4]
  slice 0 0 3 x ===? fromLiteral [3, 4, 5]
  slice 0 1 1 x ===? fromLiteral []
  slice 0 1 2 x ===? fromLiteral [4]
  slice 0 1 3 x ===? fromLiteral [4, 5]
  slice 0 2 2 x ===? fromLiteral []
  slice 0 2 3 x ===? fromLiteral [5]

  let x = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]
  slice 0 0 1 x ===? fromLiteral [[3, 4, 5]]
  slice 0 1 1 x ===? fromLiteral []
  slice 1 2 2 x ===? fromLiteral [[], []]
  slice 1 1 3 x ===? fromLiteral [[4, 5], [7, 8]]

test_index : Property
test_index = withTests 1 $ property $ do
  let x = fromLiteral {dtype=S32} [3, 4, 5]
  index 0 0 x ===? fromLiteral 3
  index 0 1 x ===? fromLiteral 4
  index 0 2 x ===? fromLiteral 5

  let x = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]
  index 0 0 x ===? fromLiteral [3, 4, 5]
  index 0 1 x ===? fromLiteral [6, 7, 8]
  index 1 0 x ===? fromLiteral [3, 6]
  index 1 1 x ===? fromLiteral [4, 7]
  index 1 2 x ===? fromLiteral [5, 8]

test_split : Property
test_split = withTests 1 $ property $ do
  let vector = fromLiteral {dtype=S32} [3, 4, 5]

  let (l, r) = split 0 0 vector
  l ===? fromLiteral []
  r ===? fromLiteral [3, 4, 5]

  let (l, r) = split 0 1 vector
  l ===? fromLiteral [3]
  r ===? fromLiteral [4, 5]

  let (l, r) = split 0 2 vector
  l ===? fromLiteral [3, 4]
  r ===? fromLiteral [5]

  let (l, r) = split 0 3 vector
  l ===? fromLiteral [3, 4, 5]
  r ===? fromLiteral []

  let arr = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]

  let (l, r) = split 0 0 arr
  l ===? fromLiteral []
  r ===? fromLiteral [[3, 4, 5], [6, 7, 8]]

  let (l, r) = split 0 1 arr
  l ===? fromLiteral [[3, 4, 5]]
  r ===? fromLiteral [[6, 7, 8]]

  let (l, r) = split 0 2 arr
  l ===? fromLiteral [[3, 4, 5], [6, 7, 8]]
  r ===? fromLiteral []

  let (l, r) = split 1 0 arr
  l ===? fromLiteral [[], []]
  r ===? fromLiteral [[3, 4, 5], [6, 7, 8]]

  let (l, r) = split 1 1 arr
  l ===? fromLiteral [[3], [6]]
  r ===? fromLiteral [[4, 5], [7, 8]]

  let (l, r) = split 1 2 arr
  l ===? fromLiteral [[3, 4], [6, 7]]
  r ===? fromLiteral [[5], [8]]

  let (l, r) = split 1 3 arr
  l ===? fromLiteral [[3, 4, 5], [6, 7, 8]]
  r ===? fromLiteral [[], []]

test_concat : Property
test_concat = withTests 1 $ property $ do
  let vector = fromLiteral {dtype=S32} [3, 4, 5]

  let l = fromLiteral {shape=[0]} []
      r = fromLiteral [3, 4, 5]
  concat 0 l r ===? vector

  let l = fromLiteral [3]
      r = fromLiteral [4, 5]
  concat 0 l r ===? vector

  let l = fromLiteral [3, 4]
      r = fromLiteral [5]
  concat 0 l r ===? vector

  let l = fromLiteral [3, 4, 5]
      r = fromLiteral {shape=[0]} []
  concat 0 l r ===? vector

  let arr = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]

  let l = fromLiteral {shape=[0, 3]} []
      r = fromLiteral [[3, 4, 5], [6, 7, 8]]
  concat 0 l r ===? arr

  let l = fromLiteral [[3, 4, 5]]
      r = fromLiteral [[6, 7, 8]]
  concat 0 l r ===? arr

  let l = fromLiteral [[3, 4, 5], [6, 7, 8]]
      r = fromLiteral {shape=[0, 3]} []
  concat 0 l r ===? arr

  let l = fromLiteral {shape=[2, 0]} [[], []]
      r = fromLiteral [[3, 4, 5], [6, 7, 8]]
  concat 1 l r ===? arr

  let l = fromLiteral [[3], [6]]
      r = fromLiteral [[4, 5], [7, 8]]
  concat 1 l r ===? arr

  let l = fromLiteral [[3, 4], [6, 7]]
      r = fromLiteral [[5], [8]]
  concat 1 l r ===? arr

  let l = fromLiteral [[3, 4, 5], [6, 7, 8]]
      r = fromLiteral {shape=[2, 0]} [[], []]
  concat 1 l r ===? arr

test_diag : Property
test_diag = withTests 1 $ property $ do
  let x = fromLiteral {dtype=S32} []
  diag x ===? fromLiteral []

  let x = fromLiteral {dtype=S32} [[3]]
  diag x ===? fromLiteral [3]

  let x = fromLiteral {dtype=S32} [[1, 2], [3, 4]]
  diag x ===? fromLiteral [1, 4]

test_triangle : Property
test_triangle = withTests 1 $ property $ do
  let x = fromLiteral {dtype=S32} []
  triangle Upper x ===? fromLiteral []
  triangle Lower x ===? fromLiteral []

  let x = fromLiteral {dtype=S32} [[3]]
  triangle Upper x ===? fromLiteral [[3]]
  triangle Lower x ===? fromLiteral [[3]]

  let x = fromLiteral {dtype=S32} [[1, 2], [3, 4]]
  triangle Upper x ===? fromLiteral [[1, 2], [0, 4]]
  triangle Lower x ===? fromLiteral [[1, 0], [3, 4]]

  let x = fromLiteral {dtype=S32} [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  triangle Upper x ===? fromLiteral [[1, 2, 3], [0, 5, 6], [0, 0, 9]]
  triangle Lower x ===? fromLiteral [[1, 0, 0], [4, 5, 0], [7, 8, 9]]

test_identity : Property
test_identity = withTests 1 $ property $ do
  identity ===? fromLiteral {dtype=S32} []
  identity ===? fromLiteral {dtype=S32} [[1]]
  identity ===? fromLiteral {dtype=S32} [[1, 0], [0, 1]]
  identity ===? fromLiteral {dtype=S32} [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

  identity ===? fromLiteral {dtype=F64} []
  identity ===? fromLiteral {dtype=F64} [[1.0]]
  identity ===? fromLiteral {dtype=F64} [[1.0, 0.0], [0.0, 1.0]]
  identity ===? fromLiteral {dtype=F64} [
      [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]
    ]

test_expand : Property
test_expand = withTests 1 $ property $ do
  expand 0 3 ===? fromLiteral {dtype=S32} [3]

  let x = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]
      with_extra_dim = fromLiteral [[[3, 4, 5]], [[6, 7, 8]]]
  expand 1 x ===? with_extra_dim

test_broadcast : Property
test_broadcast = withTests 1 $ property $ do
  broadcast {to=[]} {dtype=S32} 7 ===? 7
  broadcast {to=[1]} {dtype=S32} 7 ===? fromLiteral [7]
  broadcast {to=[2, 3]} 7 ===? fromLiteral [[7, 7, 7], [7, 7, 7]]
  broadcast {to=[1, 1, 1]} {dtype=S32} 7 ===? fromLiteral [[[7]]]
  broadcast {to=[0]} 7 ===? fromLiteral []

  let x = fromLiteral {dtype=S32} [7]
  broadcast {to=[1]} x ===? fromLiteral [7]

  let x = fromLiteral {dtype=S32} [7]
  broadcast {to=[3]} x ===? fromLiteral [7, 7, 7]

  let x = fromLiteral {dtype=S32} [7]
  broadcast {to=[2, 3]} x ===? fromLiteral [[7, 7, 7], [7, 7, 7]]

  let x = fromLiteral {dtype=S32} [5, 7]
  broadcast {to=[2, 0]} x ===? fromLiteral [[], []]

  let x = fromLiteral {dtype=S32} [5, 7]
  broadcast {to=[3, 2]} x ===? fromLiteral [[5, 7], [5, 7], [5, 7]]

  let x = fromLiteral {dtype=S32} [[2, 3, 5], [7, 11, 13]]
  broadcast {to=[2, 3]} x ===? fromLiteral [[2, 3, 5], [7, 11, 13]]

  let x = fromLiteral {dtype=S32} [[2, 3, 5], [7, 11, 13]]
  broadcast {to=[2, 0]} x ===? fromLiteral [[], []]

  let x = fromLiteral {dtype=S32} [[2, 3, 5], [7, 11, 13]]
  broadcast {to=[0, 3]} x ===? fromLiteral []

  let x = fromLiteral {dtype=S32} [[2, 3, 5], [7, 11, 13]]
      expected = fromLiteral [[[2, 3, 5], [7, 11, 13]], [[2, 3, 5], [7, 11, 13]]]
  broadcast {to=[2, 2, 3]} x ===? expected

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
  broadcast {to=[2, 2, 5, 3]} x ===? expected

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

test_squeeze : Property
test_squeeze = withTests 1 $ property $ do
  let x = fromLiteral {dtype=S32} [[3]]
  squeeze x ===? 3

  let x = fromLiteral {dtype=S32} [[[3, 4, 5]], [[6, 7, 8]]]
  squeeze x ===? x

  let squeezed = fromLiteral {dtype=S32} [[3, 4, 5], [6, 7, 8]]
  squeeze x ===? squeezed

  let x = fill {shape=[1, 3, 1, 1, 2, 5, 1]} {dtype=S32} 0
  squeeze x ===? fill {shape=[3, 2, 5]} {dtype=S32} 0

test_squeezable_cannot_remove_non_ones : Squeezable [1, 2] [] -> Void
test_squeezable_cannot_remove_non_ones (Nest _) impossible

test_T : Property
test_T = withTests 1 $ property $ do
  (fromLiteral {dtype=S32} []).T ===? fromLiteral []
  (fromLiteral {dtype=S32} [[3]]).T ===? fromLiteral [[3]]

  let x = fromLiteral {dtype=S32} [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
      expected = fromLiteral [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
  x.T ===? expected

mapResult : Property
mapResult = withTests 1 $ property $ do
  let x = fromLiteral {dtype=S32} [[1, 15, 5], [-1, 7, 6]]
  map abs x ===? abs x

  let x = fromLiteral {dtype=F64} [[1.0, 2.5, 0.0], [-0.8, -0.1, 5.0]]
  map (1.0 /) x ===? fromLiteral [[1.0, 0.4, inf], [-1.25, -10.0, 0.2]]

  -- sequence_ $ do
  --   x <- ints
  --   let x = fromLiteral {dtype=S32} x
  --   pure $ map (+ 1) x ===? x + 1

  -- sequence_ $ do
  --   x <- doubles
  --   let x = fromLiteral {dtype=F64} x
  --   pure $ map (+ 1.2) x ===? x + 1.2

mapNonTrivial : Property
mapNonTrivial = withTests 1 $ property $ do
  map {a=S32} (\x => x + x) 1 ===? 2
  map {a=S32} (\_ => 2) 1 ===? 2
  map {a=S32} (map (+ 1)) 1 ===? 2

map2Result : Property
map2Result = withTests 1 $ property $ do
  let l = fromLiteral {dtype=S32} [[1, 2, 3], [-1, -2, -3]]
      r = fromLiteral {dtype=S32} [[1, 4, 2], [-2, -1, -3]]
  map2 (+) l r ===? (l + r)

  let l = fromLiteral {dtype=F64} [[1.1, 2.2, 3.3], [-1.1, -2.2, -3.3]]
      r = fromLiteral {dtype=F64} [[1.1, 4.4, 2.2], [-2.2, -1.1, -3.3]]
  map2 (+) l r ===? l + r

  -- sequence_ $ do
  --   l <- doubles
  --   r <- doubles
  --   let l' = fromLiteral {dtype=F64} l
  --       r' = fromLiteral {dtype=F64} r
  --   pure $ map2 (+) l' r' ===? l' + r'

  -- sequence_ $ do
  --   l <- doubles
  --   let l' = fromLiteral {dtype=F64} l
  --   pure $ map2 (+) l' l' ===? l' + l'

map2ResultWithReusedFnArgs : Property
map2ResultWithReusedFnArgs = withTests 1 $ property $ do
  map2 (\x, y => x + x + y + y) 1 2 ===? 6

test_reduce : Property
test_reduce = withTests 1 $ property $ do
  let x = fromLiteral {dtype=F64} [[1.1, 2.2, 3.3], [-1.1, -2.2, -3.3]]
  reduce @{Sum} 1 x ===? fromLiteral [6.6, -6.6]

  let x = fromLiteral {dtype=F64} [[1.1, 2.2, 3.3], [-1.1, -2.2, -3.3]]
  reduce @{Sum} 0 x ===? fromLiteral [0.0, 0.0, 0.0]

  let x = fromLiteral {dtype=PRED} [[True, False, True], [True, False, False]]
  reduce @{All} 1 x ===? fromLiteral [False, False]

namespace Vector
  export
  test_dot : Property
  test_dot = withTests 1 $ property $ do
    let l = fromLiteral {dtype=S32} [-2, 0, 1]
        r = fromLiteral {dtype=S32} [3, 1, 2]
    l @@ r ===? -4

namespace Matrix
  export
  test_dot : Property
  test_dot = withTests 1 $ property $ do
    let l = fromLiteral {dtype=S32} [[-2, 0, 1], [1, 3, 4]]
        r = fromLiteral {dtype=S32} [3, 3, -1]
    l @@ r ===? fromLiteral [-7, 8]

    let l = fromLiteral {dtype=S32} [[-2, 0, 1], [1, 3, 4]]
        r = fromLiteral {dtype=S32} [[3, -1], [3, 2], [-1, -4]]
    l @@ r ===? fromLiteral [[ -7,  -2], [  8, -11]]

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
    [| fDouble x |] ==~ toLiteral (fTensor x')

namespace PRED
  export covering
  testElementwiseUnary :
    (Bool -> Bool) ->
    (forall shape . Tensor shape PRED -> Tensor shape PRED) ->
    Property
  testElementwiseUnary fBool fTensor = property $ do
    shape <- forAll shapes
    x <- forAll (literal shape bool)
    let x' = fromLiteral x
    [| fBool x |] === toLiteral (fTensor x')

covering
testElementwiseUnaryCases : List (PropertyName, Property)
testElementwiseUnaryCases = [
    ("negate S32", S32.testElementwiseUnary negate negate),
    ("negate F64", F64.testElementwiseUnary negate negate),
    ("abs S32", S32.testElementwiseUnary abs abs),
    ("abs F64", F64.testElementwiseUnary abs abs),
    ("exp", F64.testElementwiseUnary exp exp),
    ("ceil", F64.testElementwiseUnary ceiling ceil),
    ("floor", F64.testElementwiseUnary floor floor),
    ("log", F64.testElementwiseUnary log log),
    ("logistic", F64.testElementwiseUnary (\x => 1 / (1 + exp (-x))) logistic),
    ("sin", F64.testElementwiseUnary sin sin),
    ("cos", F64.testElementwiseUnary cos cos),
    ("tanh", F64.testElementwiseUnary tanh' tanh),
    ("sqrt", F64.testElementwiseUnary sqrt sqrt),
    ("not", PRED.testElementwiseUnary not not)
  ]

  where
  tanh' : Double -> Double
  tanh' x = let idrisResult = tanh x in
    if isNan idrisResult then
    if isNan x then idrisResult else
    if x < 0 then -1 else 1 else idrisResult

namespace S32
  export covering
  testElementwiseBinary :
    (Int -> Int -> Int) ->
    (forall shape . Tensor shape S32 -> Tensor shape S32 -> Tensor shape S32) ->
    Property
  testElementwiseBinary fInt fTensor = property $ do
    shape <- forAll shapes
    let ints = literal shape ints
    [x, y] <- forAll (np [ints, ints])
    let x' = fromLiteral {dtype=S32} x
        y' = fromLiteral {dtype=S32} y
    [| fInt x y |] === toLiteral (fTensor x' y')

namespace F64
  export covering
  testElementwiseBinary :
    (Double -> Double -> Double) ->
    (forall shape . Tensor shape F64 -> Tensor shape F64 -> Tensor shape F64) ->
    Property
  testElementwiseBinary fDouble fTensor = property $ do
    shape <- forAll shapes
    let doubles = literal shape doubles
    [x, y] <- forAll (np [doubles, doubles])
    let x' = fromLiteral {dtype=F64} x
        y' = fromLiteral {dtype=F64} y
    [| fDouble x y |] ==~ toLiteral (fTensor x' y')

namespace PRED
  export covering
  testElementwiseBinary :
    (Bool -> Bool -> Bool) ->
    (forall shape . Tensor shape PRED -> Tensor shape PRED -> Tensor shape PRED) ->
    Property
  testElementwiseBinary fBool fTensor = property $ do
    shape <- forAll shapes
    let bools = literal shape bool
    [x, y] <- forAll (np [bools, bools])
    let x' = fromLiteral {dtype=PRED} x
        y' = fromLiteral {dtype=PRED} y
    [| fBool x y |] === toLiteral (fTensor x' y')

covering
testElementwiseBinaryCases : List (PropertyName, Property)
testElementwiseBinaryCases = [
    ("(+) F64", F64.testElementwiseBinary (+) (+)),
    ("(+) S32", S32.testElementwiseBinary (+) (+)),
    ("(-) F64", F64.testElementwiseBinary (-) (-)),
    ("(-) S32", S32.testElementwiseBinary (-) (-)),
    ("(*) F64", F64.testElementwiseBinary (*) (*)),
    ("(*) S32", S32.testElementwiseBinary (*) (*)),
    ("(/)", F64.testElementwiseBinary (/) (/)),
    -- ("pow", F64.testElementwiseBinary pow (^)),  bug in idris 0.5.1 for pow
    ("min S32", S32.testElementwiseBinary min min),
    ("max S32", S32.testElementwiseBinary max max),
    ("(&&)", PRED.testElementwiseBinary and (&&)),
    ("(||)", PRED.testElementwiseBinary or (||))
  ]

  where
  and : Bool -> Bool -> Bool
  and x y = x && y

  or : Bool -> Bool -> Bool
  or x y = x || y

covering
test_minF64 : Property
test_minF64 = property $ do
  shape <- forAll shapes
  -- XLA has a bug for nan values
  let doubles = literal shape doublesWithoutNan
  [x, y] <- forAll (np [doubles, doubles])
  let x' = fromLiteral {dtype=F64} x
      y' = fromLiteral {dtype=F64} y
  [| min x y |] ==~ toLiteral (min x' y')

covering
test_maxF64 : Property
test_maxF64 = property $ do
  shape <- forAll shapes
  -- XLA has a bug for nan values
  let doubles = literal shape doublesWithoutNan
  [x, y] <- forAll (np [doubles, doubles])
  let x' = fromLiteral {dtype=F64} x
      y' = fromLiteral {dtype=F64} y
  [| max x y |] ==~ toLiteral (max x' y')

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
      map (scalar *) lit ==~ toLiteral (scalar' * lit')

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
      map (/ scalar) lit ==~ toLiteral (lit' / scalar')

covering
test_Sum : Property
test_Sum = property $ do
  shape <- forAll shapes

  x <- forAll (literal shape doubles)
  let x' = fromLiteral {dtype=F64} x
      right = (<+>) @{Sum} x' (neutral @{Sum})
      left = (<+>) @{Sum} (neutral @{Sum}) x'
  toLiteral right ==~ x
  toLiteral left ==~ x

  x <- forAll (literal shape ints)
  let x' = fromLiteral {dtype=S32} x
      right = (<+>) @{Sum} x' (neutral @{Sum})
      left = (<+>) @{Sum} (neutral @{Sum}) x'
  toLiteral right === x
  toLiteral left === x

covering
test_Prod : Property
test_Prod = property $ do
  shape <- forAll shapes

  x <- forAll (literal shape doubles)
  let x' = fromLiteral {dtype=F64} x
      right = (<+>) @{Prod} x' (neutral @{Prod})
      left = (<+>) @{Prod} (neutral @{Prod}) x'
  toLiteral right ==~ x
  toLiteral left ==~ x

  x <- forAll (literal shape ints)
  let x' = fromLiteral {dtype=S32} x
      right = (<+>) @{Prod} x' (neutral @{Prod})
      left = (<+>) @{Prod} (neutral @{Prod}) x'
  toLiteral right === x
  toLiteral left === x

covering
test_Any : Property
test_Any = property $ do
  shape <- forAll shapes
  x <- forAll (literal shape bool)
  let x' = fromLiteral {dtype=PRED} x
      right = (<+>) @{Any} x' (neutral @{Any})
      left = (<+>) @{Any} (neutral @{Any}) x'
  toLiteral right === x
  toLiteral left === x

covering
test_All : Property
test_All = property $ do
  shape <- forAll shapes
  x <- forAll (literal shape bool)
  let x' = fromLiteral {dtype=PRED} x
      right = (<+>) @{All} x' (neutral @{All})
      left = (<+>) @{All} (neutral @{All}) x'
  toLiteral right === x
  toLiteral left === x

covering
test_Min : Property
test_Min = property $ do
  shape <- forAll shapes
  x <- forAll (literal shape doublesWithoutNan)
  let x' = fromLiteral {dtype=F64} x
      right = (<+>) @{Min} x' (neutral @{Min})
      left = (<+>) @{Min} (neutral @{Min}) x'
  toLiteral right ==~ x
  toLiteral left ==~ x

covering
test_Max : Property
test_Max = property $ do
  shape <- forAll shapes
  x <- forAll (literal shape doublesWithoutNan)
  let x' = fromLiteral {dtype=F64} x
      right = (<+>) @{Max} x' (neutral @{Max})
      left = (<+>) @{Max} (neutral @{Max}) x'
  toLiteral right ==~ x
  toLiteral left ==~ x

namespace S32
  export covering
  testElementwiseComparator :
    (Int -> Int -> Bool) ->
    (forall shape . Tensor shape S32 -> Tensor shape S32 -> Tensor shape PRED) ->
    Property
  testElementwiseComparator fInt fTensor = property $ do
    shape <- forAll shapes
    let ints = literal shape ints
    [x, y] <- forAll (np [ints, ints])
    let x' = fromLiteral {dtype=S32} x
        y' = fromLiteral {dtype=S32} y
    [| fInt x y |] === toLiteral (fTensor x' y')

namespace F64
  export covering
  testElementwiseComparator :
    (Double -> Double -> Bool) ->
    (forall shape . Tensor shape F64 -> Tensor shape F64 -> Tensor shape PRED) ->
    Property
  testElementwiseComparator fDouble fTensor = property $ do
    shape <- forAll shapes
    let doubles = literal shape doubles
    [x, y] <- forAll (np [doubles, doubles])
    let x' = fromLiteral {dtype=F64} x
        y' = fromLiteral {dtype=F64} y
    [| fDouble x y |] === toLiteral (fTensor x' y')

namespace PRED
  export covering
  testElementwiseComparator :
    (Bool -> Bool -> Bool) ->
    (forall shape . Tensor shape PRED -> Tensor shape PRED -> Tensor shape PRED) ->
    Property
  testElementwiseComparator = testElementwiseBinary

covering
testElementwiseComparatorCases : List (PropertyName, Property)
testElementwiseComparatorCases = [
    ("(==) F64", F64.testElementwiseComparator (==) (==)),
    ("(==) S32", S32.testElementwiseComparator (==) (==)),
    ("(==) PRED", PRED.testElementwiseComparator (==) (==)),
    ("(/=) F64", F64.testElementwiseComparator (/=) (/=)),
    ("(/=) S32", S32.testElementwiseComparator (/=) (/=)),
    ("(/=) PRED", PRED.testElementwiseComparator (/=) (/=)),
    ("(<) F64", F64.testElementwiseComparator (<) (<)),
    ("(<) S32", S32.testElementwiseComparator (<) (<)),
    ("(>) F64", F64.testElementwiseComparator (>) (>)),
    ("(>) S32", S32.testElementwiseComparator (>) (>)),
    ("(<=) F64", F64.testElementwiseComparator (<=) (<=)),
    ("(<=) S32", S32.testElementwiseComparator (<=) (<=)),
    ("(>=) F64", F64.testElementwiseComparator (>=) (>=)),
    ("(>=) S32", S32.testElementwiseComparator (>=) (>=))
  ]

test_select : Property
test_select = withTests 1 $ property $ do
  let onTrue = fromLiteral {dtype=S32} 1
      onFalse = fromLiteral 0
  select (fromLiteral True) onTrue onFalse ===? onTrue
  select (fromLiteral False) onTrue onFalse ===? onFalse

  let pred = fromLiteral [[False, True, True], [True, False, False]]
      onTrue = fromLiteral {dtype=S32} [[0, 1, 2], [3, 4, 5]]
      onFalse = fromLiteral [[6, 7, 8], [9, 10, 11]]
      expected = fromLiteral [[6, 1, 2], [3, 10, 11]]
  select pred onTrue onFalse ===? expected

condResultTrivialUsage : Property
condResultTrivialUsage = withTests 1 $ property $ do
  let x = fromLiteral {dtype=S32} 0
  cond (fromLiteral True) (+ 1) x (\x => x - 1) x ===? 1

  let x = fromLiteral {dtype=S32} 0
  cond (fromLiteral False) (+ 1) x (\x => x - 1) x ===? -1

  let x = fromLiteral {dtype=S32} [2, 3]
      y = fromLiteral [[6, 7], [8, 9]]
  cond (fromLiteral True) (fromLiteral 5 *) x diag y ===? fromLiteral [10, 15]

  let x = fromLiteral {dtype=S32} [2, 3]
      y = fromLiteral [[6, 7], [8, 9]]
  cond (fromLiteral False) (fromLiteral 5 *) x diag y ===? fromLiteral [6, 9]

condResultWithReusedArgs : Property
condResultWithReusedArgs = withTests 1 $ property $ do
  let x = fromLiteral {dtype=S32} 1
      y = fromLiteral {dtype=S32} 3
  cond (fromLiteral True) (\z => z + z) x (\z => z * z) y ===? 2
  cond (fromLiteral False) (\z => z + z) x (\z => z * z) y ===? 9

test_erf : Property
test_erf = withTests 1 $ property $ do
  let x = fromLiteral [-1.5, -0.5, 0.5, 1.5]
      expected = fromLiteral [-0.96610516, -0.5204998, 0.5204998, 0.9661051]
  fpTensorEq {tol=0.000001} (erf x) expected

test_cholesky : Property
test_cholesky = withTests 1 $ property $ do
  let x = fromLiteral [[1.0, 0.0], [2.0, 0.0]]
      expected = fromLiteral [[nan, 0], [nan, nan]]
  cholesky x ===? expected

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
  fpTensorEq {tol=0.000001} (cholesky x) expected

triangularSolveResultAndInverse : Property
triangularSolveResultAndInverse = withTests 1 $ property $ do
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
  fpTensorEq {tol=0.000001} actual expected
  fpTensorEq {tol=0.000001} (a @@ actual) b

  let actual = a.T \| b
      expected = fromLiteral [
                    [-2.3692384 , -2.135952  ],
                    [ 0.31686386, -0.594465  ],
                    [ 4.0527363 ,  3.9613056 ]
                  ]
  fpTensorEq {tol=0.000001} actual expected
  fpTensorEq {tol=0.000001} (a.T @@ actual) b

triangularSolveIgnoresOppositeElems : Property
triangularSolveIgnoresOppositeElems = withTests 1 $ property $ do
  let a = fromLiteral [[1.0, 2.0], [3.0, 4.0]]
      a_lt = fromLiteral [[1.0, 0.0], [3.0, 4.0]]
      b = fromLiteral [5.0, 6.0]
  a |\ b ===? a_lt |\ b

  let a_ut = fromLiteral [[1.0, 2.0], [0.0, 4.0]]
  a \| b ===? a_ut \| b

test_trace : Property
test_trace = withTests 1 $ property $ do
  let x = fromLiteral {dtype=S32} [[-1, 5], [1, 4]]
  trace x ===? 3

export covering
group : Group
group = MkGroup "Tensor" $ [
      ("toLiteral . fromLiteral", test_fromLiteral_toLiteral)
    , ("show @{Graph}", test_show_graph)
    , ("show @{Graph} 2", test_show_graph')
    , ("show @{XLA}", test_show_xla)
    , ("reshape", test_reshape)
    , ("slice", test_slice)
    , ("index", test_index)
    , ("split", test_split)
    , ("concat", test_concat)
    , ("diag", test_diag)
    , ("triangle", test_triangle)
    , ("identity", test_identity)
    , ("expand", test_expand)
    , ("broadcast", test_broadcast)
    , ("squeeze", test_squeeze)
    , ("(.T)", test_T)
    , ("map", mapResult)
    , ("map with non-trivial function", mapNonTrivial)
    , ("map2", map2Result)
    , ("map2 with re-used function arguments", map2ResultWithReusedFnArgs)
    , ("reduce", test_reduce)
    , ("Vector.(@@)", Vector.test_dot)
    , ("Matrix.(@@)", Matrix.test_dot)
  ]
  ++ testElementwiseComparatorCases
  ++ testElementwiseUnaryCases
  ++ testElementwiseBinaryCases
  ++ [
      ("Scalarwise.(*)", test_scalar_multiplication)
    , ("Scalarwise.(/)", test_scalar_division)
    , ("Sum", test_Sum)
    , ("Prod", test_Prod)
    , ("min F64", test_minF64)
    , ("max F64", test_maxF64)
    , ("Min", test_Min)
    , ("Max", test_Max)
    , ("Any", test_Any)
    , ("All", test_All)
    , ("select", test_select)
    , ("cond for trivial usage", condResultTrivialUsage)
    , ("cond for re-used arguments", condResultWithReusedArgs)
    , ("erf", test_erf)
    , ("cholesky", test_cholesky)
    , (#"(|\) and (/|) result and inverse"#, triangularSolveResultAndInverse)
    , (#"(|\) and (/|) ignore opposite elements"#, triangularSolveIgnoresOppositeElems)
    , ("trace", test_trace)
  ]
