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
import System

import Tensor

import Utils

test_const_toArray : IO ()
test_const_toArray = do
  let x = [[True, False, False], [False, True, False]]
      x' = toArray $ const {shape=[_, _]} {dtype=PRED} x
  assert "const toArray returns original Bool" (x' == x)

  let x =  [[1, 15, 5], [-1, 7, 6]]
      x' = toArray $ const {shape=[_, _]} {dtype=S32} x
  assert "const toArray returns original Int" (x' == x)

  let name = "const toArray returns original Double"
      x = toArray $ const {shape=[_, _]} {dtype=F64} [[-1.5], [1.3], [4.3]]
  assert name $ sufficientlyEq (index 0 (index 0 x)) (-1.5)
  assert name $ sufficientlyEq (index 0 (index 1 x)) 1.3
  assert name $ sufficientlyEq (index 0 (index 2 x)) 4.3

  let name = "const toArray returns original scalar"
  traverse_ (\x =>
      let x' = toArray {shape=[]} {dtype=PRED} (const x)
       in assert name (x == x')
    ) bools
  traverse_ (\x => let x' = toArray {shape=[]} {dtype=S32} (const x) in assert name (x == x')) ints
  traverse_ (\x =>
      let x' = toArray {shape=[]} {dtype=F64} (const x)
       in assert name (sufficientlyEq x x')
    ) doubles

test_show_graph : IO ()
test_show_graph = do
  let x = const {shape=[]} {dtype=S32} 1
  assert "show @{Graph} for scalar Int" $ show @{Graph} x == "S32[] const"

  let x = const {shape=[]} {dtype=S32} 1
      y = const {shape=[]} {dtype=S32} 2
  assert "show @{Graph} for scalar addition" $ show @{Graph} (Tensor.(+) x y) ==
    "S32[] (+): [S32[] const, S32[] const]"

  let x = const {shape=[_]} {dtype=F64} [1.3, 2.0, -0.4]
  assert "show @{Graph} for vector F64" $ show @{Graph} x == "F64[3] const"

  let x = const {shape=[_, _]} {dtype=S32} [[0, 0, 0], [0, 0, 0]]
      y = const {shape=[_, _]} [[0], [0], [0]]
  assert "show @{Graph} for differing argument shapes" $ show @{Graph} (x @@ y) ==
    "S32[2, 1] (@@): [S32[2, 3] const, S32[3, 1] const]"

  let x = const {shape=[_]} {dtype=S32} [0, 0]
      y = const {shape=[_, _]} {dtype=S32} [[0, 0], [0, 0]]
  assert "show @{Graph} for non-trivial cond" $
    show @{Graph} (cond (const True) (const [0, 0] *) x diag y) ==
      "S32[2] cond: [PRED[] const, S32[2] (*): [S32[2] const, S32[2] parameter], " ++
        "S32[2] const, S32[2] diag: [S32[2, 2] parameter], S32[2, 2] const]"

test_show_graphxla : IO ()
test_show_graphxla = do
  let x = const {shape=[]} {dtype=S32} 1
  assert "show @{XLA} for scalar Int" $ show @{XLA} x == "constant, shape=[], metadata={:0}"

  let x = const {shape=[]} {dtype=S32} 1
      y = const {shape=[]} {dtype=S32} 2
  assert "show @{XLA} for scalar addition" $ show @{XLA} (Tensor.(+) x y) ==
    """
    add, shape=[], metadata={:0}
      constant, shape=[], metadata={:0}
      constant, shape=[], metadata={:0}
    """

  let x = const {shape=[_]} {dtype=F64} [1.3, 2.0, -0.4]
  assert "show @{XLA} for vector F64" $ show @{XLA} x == "constant, shape=[3], metadata={:0}"

test_reshape : IO ()
test_reshape = do
  let x = const {shape=[]} {dtype=S32} 3
      expected = const {shape=[1]} {dtype=S32} [3]
  assertAll "reshape add dims scalar" $ reshape x == expected

  let x = const {shape=[3]} {dtype=S32} [3, 4, 5]
      flipped = const {shape=[3, 1]} {dtype=S32} [[3], [4], [5]]
  assertAll "reshape flip dims vector" $ reshape x == flipped

  let x = const {shape=[2, 3]} {dtype=S32} [[3, 4, 5], [6, 7, 8]]
      flipped = const {shape=[3, 2]} {dtype=S32} [[3, 4], [5, 6], [7, 8]]
  assertAll "reshape flip dims array" $ reshape x == flipped

  let with_extra_dim = const {shape=[2, 1, 3]} {dtype=S32} [[[3, 4, 5]], [[6, 7, 8]]]
  assertAll "reshape add dimension array" $ reshape x == with_extra_dim

  let flattened = const {shape=[6]} {dtype=S32} [3, 4, 5, 6, 7, 8]
  assertAll "reshape as flatten array" $ reshape x == flattened

test_slice : IO ()
test_slice = do
  let x = const {shape=[3]} {dtype=S32} [3, 4, 5]
  assertAll "slice vector 0 0" $ slice 0 0 0 x == const []
  assertAll "slice vector 0 1" $ slice 0 0 1 x == const [3]
  assertAll "slice vector 0 2" $ slice 0 0 2 x == const [3, 4]
  assertAll "slice vector 0 3" $ slice 0 0 3 x == const [3, 4, 5]
  assertAll "slice vector 1 1" $ slice 0 1 1 x == const []
  assertAll "slice vector 1 2" $ slice 0 1 2 x == const [4]
  assertAll "slice vector 1 3" $ slice 0 1 3 x == const [4, 5]
  assertAll "slice vector 2 2" $ slice 0 2 2 x == const []
  assertAll "slice vector 2 2" $ slice 0 2 3 x == const [5]

  let x = const {shape=[2, 3]} {dtype=S32} [[3, 4, 5], [6, 7, 8]]
  assertAll "slice array 0 0 1" $ slice 0 0 1 x == const [[3, 4, 5]]
  assertAll "slice array 0 1 1" $ slice 0 1 1 x == const []
  assertAll "slice array 1 2 2" $ slice 1 2 2 x == const [[], []]
  assertAll "slice array 1 1 3" $ slice 1 1 3 x == const [[4, 5], [7, 8]]

test_index : IO ()
test_index = do
  let x = const {shape=[3]} {dtype=S32} [3, 4, 5]
  assertAll "index vector 0" $ index 0 0 x == const 3
  assertAll "index vector 1" $ index 0 1 x == const 4
  assertAll "index vector 2" $ index 0 2 x == const 5

  let x = const {shape=[2, 3]} {dtype=S32} [[3, 4, 5], [6, 7, 8]]
  assertAll "index array 0 0" $ index 0 0 x == const [3, 4, 5]
  assertAll "index array 0 1" $ index 0 1 x == const [6, 7, 8]
  assertAll "index array 1 0" $ index 1 0 x == const [3, 6]
  assertAll "index array 1 1" $ index 1 1 x == const [4, 7]
  assertAll "index array 1 2" $ index 1 2 x == const [5, 8]

test_split : IO ()
test_split = do
  let vector = const {shape=[3]} {dtype=S32} [3, 4, 5]

  let (l, r) = split 0 0 vector
  assertAll "split vector 0 left" $ l == const []
  assertAll "split vector 0 right" $ r == const [3, 4, 5]

  let (l, r) = split 0 1 vector
  assertAll "split vector 1 left" $ l == const [3]
  assertAll "split vector 1 right" $ r == const [4, 5]

  let (l, r) = split 0 2 vector
  assertAll "split vector 2 left" $ l == const [3, 4]
  assertAll "split vector 2 right" $ r == const [5]

  let (l, r) = split 0 3 vector
  assertAll "split vector 3 left" $ l == const [3, 4, 5]
  assertAll "split vector 3 right" $ r == const []

  let arr = const {shape=[2, 3]} {dtype=S32} [[3, 4, 5], [6, 7, 8]]

  let (l, r) = split 0 0 arr
  assertAll "split array 0 0 left" $ l == const []
  assertAll "split array 0 0 right" $ r == const [[3, 4, 5], [6, 7, 8]]

  let (l, r) = split 0 1 arr
  assertAll "split array 0 1 left" $ l == const [[3, 4, 5]]
  assertAll "split array 0 1 right" $ r == const [[6, 7, 8]]

  let (l, r) = split 0 2 arr
  assertAll "split array 0 2 left" $ l == const [[3, 4, 5], [6, 7, 8]]
  assertAll "split array 0 2 right" $ r == const []

  let (l, r) = split 1 0 arr
  assertAll "split array 1 0 left" $ l == const [[], []]
  assertAll "split array 1 0 right" $ r == const [[3, 4, 5], [6, 7, 8]]

  let (l, r) = split 1 1 arr
  assertAll "split array 1 1 left" $ l == const [[3], [6]]
  assertAll "split array 1 1 right" $ r == const [[4, 5], [7, 8]]

  let (l, r) = split 1 2 arr
  assertAll "split array 1 2 left" $ l == const [[3, 4], [6, 7]]
  assertAll "split array 1 2 right" $ r == const [[5], [8]]

  let (l, r) = split 1 3 arr
  assertAll "split array 1 3 left" $ l == const [[3, 4, 5], [6, 7, 8]]
  assertAll "split array 1 3 right" $ r == const [[], []]

test_concat : IO ()
test_concat = do
  let vector = const {shape=[3]} {dtype=S32} [3, 4, 5]

  let l = const {shape=[0]} []
      r = const {shape=[3]} [3, 4, 5]
  assertAll "concat vector" $ concat 0 l r == vector

  let l = const {shape=[1]} [3]
      r = const {shape=[2]} [4, 5]
  assertAll "concat vector" $ concat 0 l r == vector

  let l = const {shape=[2]} [3, 4]
      r = const {shape=[1]} [5]
  assertAll "concat vector" $ concat 0 l r == vector

  let l = const {shape=[3]} [3, 4, 5]
      r = const {shape=[0]} []
  assertAll "concat vector" $ concat 0 l r == vector

  let arr = const {shape=[2, 3]} {dtype=S32} [[3, 4, 5], [6, 7, 8]]

  let l = const {shape=[0, 3]} []
      r = const {shape=[2, 3]} [[3, 4, 5], [6, 7, 8]]
  assertAll "concat array 0" $ concat 0 l r == arr

  let l = const {shape=[1, 3]} [[3, 4, 5]]
      r = const {shape=[1, 3]} [[6, 7, 8]]
  assertAll "concat array 0" $ concat 0 l r == arr

  let l = const {shape=[2, 3]} [[3, 4, 5], [6, 7, 8]]
      r = const {shape=[0, 3]} []
  assertAll "concat array 0" $ concat 0 l r == arr

  let l = const {shape=[2, 0]} [[], []]
      r = const {shape=[2, 3]} [[3, 4, 5], [6, 7, 8]]
  assertAll "concat array 1" $ concat 1 l r == arr

  let l = const {shape=[2, 1]} [[3], [6]]
      r = const {shape=[2, 2]} [[4, 5], [7, 8]]
  assertAll "concat array 1" $ concat 1 l r == arr

  let l = const {shape=[2, 2]} [[3, 4], [6, 7]]
      r = const {shape=[2, 1]} [[5], [8]]
  assertAll "concat array 1" $ concat 1 l r == arr

  let l = const {shape=[2, 3]} [[3, 4, 5], [6, 7, 8]]
      r = const {shape=[2, 0]} [[], []]
  assertAll "concat array 1" $ concat 1 l r == arr

test_diag : IO ()
test_diag = do
  let x = const {dtype=S32} []
  assertAll "diag empty" $ diag x == const []

  let x = const {dtype=S32} [[3]]
  assertAll "diag 1" $ diag x == const [3]

  let x = const {dtype=S32} [[1, 2], [3, 4]]
  assertAll "diag 2" $ diag x == const [1, 4]

test_triangle : IO ()
test_triangle = do
  let x = const {dtype=S32} []
  assertAll "triangle upper empty" $ triangle Upper x == const []
  assertAll "triangle lower empty" $ triangle Lower x == const []

  let x = const {dtype=S32} [[3]]
  assertAll "triangle upper 1" $ triangle Upper x == const [[3]]
  assertAll "triangle lower 1" $ triangle Lower x == const [[3]]

  let x = const {dtype=S32} [[1, 2], [3, 4]]
  assertAll "triangle upper 2" $ triangle Upper x == const [[1, 2], [0, 4]]
  assertAll "triangle lower 2" $ triangle Lower x == const [[1, 0], [3, 4]]

  let x = const {dtype=S32} [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  assertAll "triangle upper 3" $ triangle Upper x == const [[1, 2, 3], [0, 5, 6], [0, 0, 9]]
  assertAll "triangle lower 3" $ triangle Lower x == const [[1, 0, 0], [4, 5, 0], [7, 8, 9]]

test_identity : IO ()
test_identity = do
  assertAll "identity 0 S32" $ identity == const {dtype=S32} []
  assertAll "identity 1 S32" $ identity == const {dtype=S32} [[1]]
  assertAll "identity 2 S32" $ identity == const {dtype=S32} [[1, 0], [0, 1]]
  assertAll "identity 4 S32" $
    identity == const {dtype=S32} [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

  assertAll "identity 0 F64" $ identity == const {dtype=F64} []
  assertAll "identity 1 F64" $ identity == const {dtype=F64} [[1]]
  assertAll "identity 2 F64" $ identity == const {dtype=F64} [[1, 0], [0, 1]]
  assertAll "identity 4 F64" $
    identity == const {dtype=F64} [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

test_expand : IO ()
test_expand = do
  let x = const {shape=[]} {dtype=S32} 3
      expected = const {shape=[1]} {dtype=S32} [3]
  assertAll "expand add dims scalar" $ expand 0 x == expected

  let x = const {shape=[2, 3]} {dtype=S32} [[3, 4, 5], [6, 7, 8]]
      with_extra_dim = const {shape=[2, 1, 3]} {dtype=S32} [[[3, 4, 5]], [[6, 7, 8]]]
  assertAll "expand add dimension array" $ expand 1 x == with_extra_dim

test_broadcast : IO ()
test_broadcast = do
  let x = const {shape=[]} {dtype=S32} 7
  assertAll "broadcast scalar to itself" $ broadcast {to=[]} x == const 7

  let x = const {shape=[]} {dtype=S32} 7
  assertAll "broadcast scalar to rank 1" $ broadcast {to=[1]} x == const [7]

  let x = const {shape=[]} {dtype=S32} 7
  assertAll "broadcast scalar to rank 2" $
    broadcast {to=[2, 3]} x == const [[7, 7, 7], [7, 7, 7]]

  let x = const {shape=[]} {dtype=S32} 7
  assertAll "broadcast scalar to rank 3" $ broadcast {to=[1, 1, 1]} x == const [[[7]]]

  let x = const {shape=[1]} {dtype=S32} [7]
  assertAll "broadcast rank 1 to empty" $ broadcast {to=[0]} x == const []

  let x = const {shape=[1]} {dtype=S32} [7]
  assertAll "broadcast rank 1 to itself" $ broadcast {to=[1]} x == const [7]

  let x = const {shape=[1]} {dtype=S32} [7]
  assertAll "broadcast rank 1 to larger rank 1" $ broadcast {to=[3]} x == const [7, 7, 7]

  let x = const {shape=[1]} {dtype=S32} [7]
  assertAll "broadcast rank 1 to rank 2" $
    broadcast {to=[2, 3]} x == const [[7, 7, 7], [7, 7, 7]]

  let x = const {shape=[2]} {dtype=S32} [5, 7]
  assertAll "broadcast rank 1 to empty" $ broadcast {to=[2, 0]} x == const [[], []]

  let x = const {shape=[2]} {dtype=S32} [5, 7]
  assertAll "broadcast rank 1 to rank 2" $
    broadcast {to=[3, 2]} x == const [[5, 7], [5, 7], [5, 7]]

  let x = const {shape=[2, 3]} {dtype=S32} [[2, 3, 5], [7, 11, 13]]
  assertAll "broadcast rank 2 to itself" $
    broadcast {to=[2, 3]} x == const [[2, 3, 5], [7, 11, 13]]

  let x = const {shape=[2, 3]} {dtype=S32} [[2, 3, 5], [7, 11, 13]]
  assertAll "broadcast rank 2 to rank 2 empty" $ broadcast {to=[2, 0]} x == const [[], []]

  let x = const {shape=[2, 3]} {dtype=S32} [[2, 3, 5], [7, 11, 13]]
  assertAll "broadcast rank 2 to empty" $ broadcast {to=[0, 3]} x == const []

  let x = const {shape=[2, 3]} {dtype=S32} [[2, 3, 5], [7, 11, 13]]
      expected = const [[[2, 3, 5], [7, 11, 13]], [[2, 3, 5], [7, 11, 13]]]
  assertAll "broadcast rank 2 to rank 3" $ broadcast {to=[2, 2, 3]} x == expected

  let x = const {shape=[2, 1, 3]} {dtype=S32} [[[2, 3, 5]], [[7, 11, 13]]]
      expected = const [
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
  let x = const {shape=[1, 1]} {dtype=S32} [[3]]
      squeezed = const {shape=[]} {dtype=S32} 3
  assertAll "squeeze can flatten only ones" $ squeeze x == squeezed

  let x = const {shape=[2, 1, 3]} {dtype=S32} [[[3, 4, 5]], [[6, 7, 8]]]
  assertAll "squeeze can no-op" $ squeeze x == x

  let squeezed = const {shape=[2, 3]} {dtype=S32} [[3, 4, 5], [6, 7, 8]]
  assertAll "squeeze can remove dim from array" $ squeeze x == squeezed

  let x = fill {shape=[1, 3, 1, 1, 2, 5, 1]} {dtype=S32} 0
  assertAll "squeeze can remove many dims from array" $
    squeeze x == fill {shape=[3, 2, 5]} {dtype=S32} 0

test_squeezable_cannot_remove_non_ones : Squeezable [1, 2] [] -> Void
test_squeezable_cannot_remove_non_ones (Nest _) impossible

test_T : IO ()
test_T = do
  assertAll "(.T) for empty array" $ (const {dtype=S32} []).T == const []
  assertAll "(.T) for single element" $ (const {dtype=S32} [[3]]).T == const [[3]]

  let x = const {shape=[_, _]} {dtype=S32} [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
      expected = const [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
  assertAll "(.T)" $ x.T == expected

test_map : IO ()
test_map = do
  let x = const {shape=[_, _]} {dtype=S32} [[1, 15, 5], [-1, 7, 6]]
  assertAll "map for S32 array" $ map abs x == abs x

  let x = const {shape=[_, _]} {dtype=F64} [[1.0, 2.5, 0.0], [-0.8, -0.1, 5.0]]
  assertAll "map for F64 array" $
    map (const 1 /) x == const [[1.0, 0.4, inf], [-1.25, -10, 0.2]]

  sequence_ $ do
    x <- ints
    let x = const {shape=[]} {dtype=S32} x
    pure $ assertAll "map for S32 scalar" $ map (+ const 1) x == x + const 1

  sequence_ $ do
    x <- doubles
    let x = const {shape=[]} {dtype=F64} x
    pure $ assertAll "map for F64 scalar" $ sufficientlyEq (map (+ const 1.2) x) (x + const 1.2)

test_map2 : IO ()
test_map2 = do
  let l = const {shape=[_, _]} {dtype=S32} [[1, 2, 3], [-1, -2, -3]]
      r = const {shape=[_, _]} {dtype=S32} [[1, 4, 2], [-2, -1, -3]]
  assertAll "map2 for S32 array" $ map2 (+) l r == (l + r)

  let l = const {shape=[_, _]} {dtype=F64} [[1.1, 2.2, 3.3], [-1.1, -2.2, -3.3]]
      r = const {shape=[_, _]} {dtype=F64} [[1.1, 4.4, 2.2], [-2.2, -1.1, -3.3]]
  assertAll "map2 for F64 matrix" $ sufficientlyEq (map2 (+) l r) (l + r)

  sequence_ $ do
    l <- doubles
    r <- doubles
    let l' = const {shape=[]} {dtype=F64} l
        r' = const {shape=[]} {dtype=F64} r
    pure $ assertAll "map2 for F64 scalars" $ sufficientlyEq (map2 (+) l' r') (l' + r')

  sequence_ $ do
    l <- doubles
    let l' = const {shape=[]} {dtype=F64} l
    pure $ assertAll "map2 for F64 scalars with repeated argument" $
      sufficientlyEq (map2 (+) l' l') (l' + l')

test_reduce : IO ()
test_reduce = do
  let x = const {shape=[_, _]} {dtype=F64} [[1.1, 2.2, 3.3], [-1.1, -2.2, -3.3]]
  assertAll "reduce for F64 array" $ sufficientlyEq (reduce @{Sum} 1 x) (const [6.6, -6.6])

  let x = const {shape=[_, _]} {dtype=F64} [[1.1, 2.2, 3.3], [-1.1, -2.2, -3.3]]
  assertAll "reduce for F64 array" $ sufficientlyEq (reduce @{Sum} 0 x) (const [0, 0, 0])

  let x = const {shape=[_, _]} {dtype=PRED} [[True, False, True], [True, False, False]]
  assertAll "reduce for PRED array" $ reduce @{All} 1 x == const [False, False]

test_elementwise_equality : IO ()
test_elementwise_equality = do
  let x = const {shape=[_]} {dtype=PRED} [True, True, False]
      y = const {shape=[_]} {dtype=PRED} [False, True, False]
      eq = toArray {shape=[_]} (y == x)
  assert "== for boolean vector" $ eq == [False, True, True]

  let x = const {shape=[_, _]} {dtype=S32} [[1, 15, 5], [-1, 7, 6]]
      y = const {shape=[_, _]} {dtype=S32} [[2, 15, 3], [2, 7, 6]]
      eq = toArray (y == x)
  assert "== for integer matrix" $ eq == [[False, True, False], [False, True, True]]

  let x = const {shape=[_, _]} {dtype=F64} [[1.1, 15.3, 5.2], [-1.6, 7.1, 6.0]]
      y = const {shape=[_, _]} {dtype=F64} [[2.2, 15.3, 3.4], [2.6, 7.1, 6.0]]
      eq = toArray (y == x)
  assert "== for double matrix" $ eq == [[False, True, False], [False, True, True]]

  sequence_ [compareScalars {dtype=PRED} x y | x <- bools, y <- bools]
  sequence_ [compareScalars {dtype=S32} x y | x <- ints, y <- ints]
  sequence_ [compareScalars {dtype=F64} x y | x <- doubles, y <- doubles]

  where
    compareScalars : Hashable ty => Primitive dtype => Prelude.Eq ty => PrimitiveRW dtype ty
                     => Primitive.Eq dtype => ty -> ty -> IO ()
    compareScalars l r =
      let actual = toArray {shape=[]} ((const {dtype} l) == (const {dtype} r))
       in assert "== for scalars" (actual == (l == r))

test_elementwise_inequality : IO ()
test_elementwise_inequality = do
  let x = const {shape=[_]} {dtype=PRED} [True, True, False]
      y = const {shape=[_]} {dtype=PRED} [False, True, False]
  assertAll "== for boolean vector" $ (y /= x) == const {shape=[_]} [True, False, False]

  let x = const {shape=[_, _]} {dtype=S32} [[1, 15, 5], [-1, 7, 6]]
      y = const {shape=[_, _]} {dtype=S32} [[2, 15, 3], [2, 7, 6]]
  assertAll "== for integer matrix" $
    (x /= y) == const [[True, False, True], [True, False, False]]

  let x = const {shape=[_, _]} {dtype=F64} [[1.1, 15.3, 5.2], [-1.6, 7.1, 6.0]]
      y = const {shape=[_, _]} {dtype=F64} [[2.2, 15.3, 3.4], [2.6, 7.1, 6.0]]
  assertAll "== for double matrix" $
    (x /= y) == const [[True, False, True], [True, False, False]]

  sequence_ [compareScalars {dtype=PRED} l r | l <- bools, r <- bools]
  sequence_ [compareScalars {dtype=S32} l r | l <- ints, r <- ints]
  sequence_ [compareScalars {dtype=F64} l r | l <- doubles, r <- doubles]

  where
    compareScalars : Hashable ty => Primitive dtype => Primitive.Eq dtype => Prelude.Eq ty
                     => PrimitiveRW dtype ty => ty -> ty -> IO ()
    compareScalars l r =
      assertAll "/= for scalars" $ (const {dtype} l /= const r) == const {shape=[]} (l /= r)

test_comparison : IO ()
test_comparison = do
  let x = const {shape=[_, _]} {dtype=S32} [[1, 2, 3], [-1, -2, -3]]
      y = const {shape=[_, _]} {dtype=S32} [[1, 4, 2], [-2, -1, -3]]
  assertAll "> for S32 matrix" $ (y > x) == const [[False, True, False], [False, True, False]]
  assertAll "< for S32 matrix" $ (y < x) == const [[False, False, True], [True, False, False]]
  assertAll ">= for S32 matrix" $ (y >= x) == const [[True, True, False], [False, True, True]]
  assertAll "<= for S32 matrix" $ (y <= x) == const [[True, False, True], [True, False, True]]

  let x = const {shape=[_, _]} {dtype=F64} [[1.1, 2.2, 3.3], [-1.1, -2.2, -3.3]]
      y = const {shape=[_, _]} {dtype=F64} [[1.1, 4.4, 2.2], [-2.2, -1.1, -3.3]]
  assertAll "> for F64 matrix" $ (y > x) == const [[False, True, False], [False, True, False]]
  assertAll "< for F64 matrix" $ (y < x) == const [[False, False, True], [True, False, False]]
  assertAll ">= for F64 matrix" $ (y >= x) == const [[True, True, False], [False, True, True]]
  assertAll "<= for F64 matrix" $ (y <= x) == const [[True, False, True], [True, False, True]]

  sequence_ [compareScalars {dtype=S32} l r | l <- ints, r <- ints]
  sequence_ [compareScalars {dtype=F64} l r | l <- doubles, r <- doubles]

  where
    compareScalars : Hashable ty => Primitive.Ord dtype => Prelude.Ord ty => PrimitiveRW dtype ty
                     => Primitive dtype => ty -> ty -> IO ()
    compareScalars l r = do
      let l' = const {dtype} l
          r' = const {dtype} r
      assertAll "> for scalars" $ (l' > r') == const {shape=[]} {dtype=PRED} (l > r)
      assertAll "< for scalars" $ (l' < r') == const {shape=[]} {dtype=PRED} (l < r)
      assertAll ">= for scalars" $ (l' >= r') == const {shape=[]} {dtype=PRED} (l >= r)
      assertAll "<= for scalars" $ (l' <= r') == const {shape=[]} {dtype=PRED} (l <= r)

namespace S32
  export
  testElementwiseBinary : String -> (Int -> Int -> Int)
      -> (forall shape . Tensor shape S32 -> Tensor shape S32 -> Tensor shape S32) -> IO ()
  testElementwiseBinary name f_native f_tensor = do
    let x = [[1, 15, 5], [-1, 7, 6]]
        y = [[11, 5, 7], [-3, -4, 0]]
        expected = const {shape=[2, 3]} {dtype=S32} $ [
          [f_native 1 11, f_native 15 5, f_native 5 7],
          [f_native (-1) (-3), f_native 7 (-4), f_native 6 0]
        ]
    assertAll (name ++ " for S32 array") $ f_tensor (const x) (const y) == expected

    sequence_ $ do
      l <- ints
      r <- ints
      pure $ assertAll (name ++ " for S32 scalar " ++ show l ++ " " ++ show r) $
        f_tensor (const l) (const r) == const {shape=[]} {dtype=S32} (f_native l r)

namespace F64
  export
  testElementwiseBinary : String -> (Double -> Double -> Double)
    -> (forall shape . Tensor shape F64 -> Tensor shape F64 -> Tensor shape F64) -> IO ()
  testElementwiseBinary name f_native f_tensor = do
    let x = [[3, 4, -5], [0, 0.3, 0]]
        y = [[1, -2.3, 0.2], [0.1, 0, 0]]
        expected = const {shape=[2, 3]} {dtype=F64} $ [
          [f_native 3 1, f_native 4 (-2.3), f_native (-5) 0.2],
          [f_native 0 0.1, f_native 0.3 0, f_native 0 0]
        ]
    assertAll (name ++ " for F64 array") $
      sufficientlyEq (f_tensor (const x) (const y)) expected

    sequence_ $ do
      l <- doubles
      r <- doubles
      pure $ assertAll (name ++ " for F64 scalar " ++ show l ++ " " ++ show r) $
        sufficientlyEq (f_tensor (const l) (const r)) $
          const {shape=[]} {dtype=F64} (f_native l r)

namespace Vector
  export
  test_dot : IO ()
  test_dot = do
    let l = const {shape=[3]} {dtype=S32} [-2, 0, 1]
        r = const {shape=[3]} {dtype=S32} [3, 1, 2]
    assertAll "vector dot" $ l @@ r == const (-4)

namespace Matrix
  export
  test_dot : IO ()
  test_dot = do
    let l = const {shape=[2, 3]} {dtype=S32} [[-2, 0, 1], [1, 3, 4]]
        r = const {shape=[3]} {dtype=S32} [3, 3, -1]
    assertAll "matrix dot vector" $ l @@ r == const [-7, 8]

    let l = const {shape=[2, 3]} {dtype=S32} [[-2, 0, 1], [1, 3, 4]]
        r = const {shape=[3, 2]} {dtype=S32} [[3, -1], [3, 2], [-1, -4]]
    assertAll "matrix dot matrix" $ l @@ r == const [[ -7,  -2], [  8, -11]]

test_add : IO ()
test_add = do
  S32.testElementwiseBinary "(+)" (+) (+)
  F64.testElementwiseBinary "(+)" (+) (+)

test_Sum : IO ()
test_Sum = do
  let x = const {shape=[_, _]} {dtype=F64} [[1.1, 2.1, -2.0], [-1.3, -1.0, 1.0]]
  assertAll "Sum neutral is neutral right" $ (<+>) @{Sum} x (neutral @{Sum}) == x
  assertAll "Sum neutral is neutral left" $ (<+>) @{Sum} (neutral @{Sum}) x == x

test_subtract : IO ()
test_subtract = do
  let l = const [[1, 15, 5], [-1, 7, 6]]
      r = const [[11, 5, 7], [-3, -4, 0]]
  assertAll "- for S32 matrix" $
    (l - r) == const {shape=[_, _]} {dtype=S32} [[-10, 10, -2], [2, 11, 6]]

  let l = const [1.8, 1.3, 4.0]
      r = const [-3.3, 0.0, 0.3]
      diff = toArray {shape=[3]} {dtype=F64} (l - r)
  sequence_ (zipWith ((assert "- for F64 matrix") .: sufficientlyEq) diff [5.1, 1.3, 3.7])

  sequence_ $ do
    l <- ints
    r <- ints
    pure $ assertAll "- for S32 scalar" $
      (const l - const r) == const {shape=[]} {dtype=S32} (l - r)

  sequence_ $ do
    l <- doubles
    r <- doubles
    pure $
      let diff = toArray {shape=[]} {dtype=F64} (const l - const r)
       in assert "- for F64 scalar" (sufficientlyEq diff (l - r))

test_elementwise_multiplication : IO ()
test_elementwise_multiplication = do
  S32.testElementwiseBinary "(*)" (*) (*)
  F64.testElementwiseBinary "(*)" (*) (*)

test_scalar_multiplication : IO ()
test_scalar_multiplication = do
  let r = const {shape=[_, _]} {dtype=S32} [[11, 5, 7], [-3, -4, 0]]
  sequence_ $ do
    l <- ints
    pure $ assertAll "* for int array" $
      (const l) * r == const [[11 * l, 5 * l, 7 * l], [-3 * l, -4 * l, 0]]

  let r = const {shape=[_, _]} {dtype=F64} [[-3.3], [0.0], [0.3]]
  sequence_ $ do
    l <- doubles
    pure $ assertAll "* for double array" $
      sufficientlyEq ((const l) * r) (const [[-3.3 * l], [0.0 * l], [0.3 * l]])

  sequence_ $ do
    l <- ints
    r <- ints
    pure $ assertAll "* for int scalar" $
      (const l * const r) == const {shape=[]} {dtype=S32} (l * r)

  sequence_ $ do
    l <- doubles
    r <- doubles
    pure $ assertAll "* for double array" $
      sufficientlyEq (const l * const r) (const {shape=[]} (l * r))

test_Prod : IO ()
test_Prod = do
  let x = const {shape=[_, _]} {dtype=F64} [[1.1, 2.1, -2.0], [-1.3, -1.0, 1.0]]
  assertAll "Prod neutral is neutral right" $ (<+>) @{Prod} x (neutral @{Prod}) == x
  assertAll "Prod neutral is neutral left" $ (<+>) @{Prod} (neutral @{Prod}) x == x

assertBooleanOpArray : String -> (Tensor [2, 2] PRED -> Tensor [2, 2] PRED -> Tensor [2, 2] PRED)
                       -> Array [2, 2] Bool -> IO ()
assertBooleanOpArray name op expected = do
  let l = const [[True, True], [False, False]]
      r = const [[True, False], [True, False]]
  assertAll name $ op l r == const expected

assertBooleanOpScalar : String -> (Tensor [] PRED -> Tensor [] PRED -> Tensor [] PRED)
                        -> (Bool -> Lazy Bool -> Bool) -> IO ()
assertBooleanOpScalar name tensor_op bool_op =
  sequence_ $ do
    l <- bools
    r <- bools
    pure $ assertAll name $ tensor_op (const l) (const r) == const (bool_op l r)

test_elementwise_and : IO ()
test_elementwise_and = do
  assertBooleanOpArray "&& for array" (&&) [[True, False], [False, False]]
  assertBooleanOpScalar "&& for scalar" (&&) (&&)

test_All : IO ()
test_All = do
  let x = const {shape=[_, _]} {dtype=PRED} [[True, True], [False, False]]
  assertAll "All neutral is neutral right" $ (<+>) @{All} x (neutral @{All}) == x
  assertAll "All neutral is neutral left" $ (<+>) @{All} (neutral @{All}) x == x

test_elementwise_or : IO ()
test_elementwise_or = do
  assertBooleanOpArray "|| for array" (||) [[True, True], [True, False]]
  assertBooleanOpScalar "|| for scalar" (||) (||)

test_Any : IO ()
test_Any = do
  let x = const {shape=[_, _]} {dtype=PRED} [[True, True], [False, False]]
  assertAll "Any neutral is neutral right" $ (<+>) @{Any} x (neutral @{Any}) == x
  assertAll "Any neutral is neutral left" $ (<+>) @{Any} (neutral @{Any}) x == x

test_elementwise_not : IO ()
test_elementwise_not = do
  assertAll "not for array" $
    not (const [True, False]) == const {shape=[_]} [False, True]
  sequence_ [assertAll "not for scalar" $
             not (const x) == const {shape=[]} (not x) | x <- bools]

test_select : IO ()
test_select = do
  let onTrue = const {shape=[]} {dtype=S32} 1
      onFalse = const 0
  assertAll "select for scalar True" $ select (const True) onTrue onFalse == onTrue
  assertAll "select for scalar False" $ select (const False) onTrue onFalse == onFalse

  let pred = const [[False, True, True], [True, False, False]]
      onTrue = const [[0, 1, 2], [3, 4, 5]]
      onFalse = const [[6, 7, 8], [9, 10, 11]]
      expected = const {shape=[_, _]} {dtype=S32} [[6, 1, 2], [3, 10, 11]]
  assertAll "select for array" $ select pred onTrue onFalse == expected

test_cond : IO ()
test_cond = do
  let x = const {shape=[]} {dtype=S32} 0
  assertAll "cond for trivial truthy" $
    cond (const True) (+ const 1) x (\x => x - const 1) x == const 1

  let x = const {shape=[]} {dtype=S32} 0
  assertAll "cond for trivial falsy" $
    cond (const False) (+ const 1) x (\x => x - const 1) x == const (-1)

  let x = const {shape=[_]} {dtype=S32} [2, 3]
      y = const {shape=[_, _]} {dtype=S32} [[6, 7], [8, 9]]
  assertAll "cond for non-trivial truthy" $ cond (const True) (const 5 *) x diag y == const [10, 15]

  let x = const {shape=[_]} {dtype=S32} [2, 3]
      y = const {shape=[_, _]} {dtype=S32} [[6, 7], [8, 9]]
  assertAll "cond for non-trivial falsy" $ cond (const False) (const 5 *) x diag y == const [6, 9]

test_elementwise_division : IO ()
test_elementwise_division = do
  F64.testElementwiseBinary "(/)" (/) (/)

test_scalar_division : IO ()
test_scalar_division = do
  let l = const {shape=[_, _]} {dtype=F64} [[-3.3], [0.0], [0.3]]
  sequence_ $ do
    r <- doubles
    pure $ assertAll "/ for array" $
      sufficientlyEq (l / const r) (const [[-3.3 / r], [0.0 / r], [0.3 / r]])

  sequence_ $ do
    l <- doubles
    r <- doubles
    pure $ assertAll "/ for scalar" $
      sufficientlyEq (const l / const r) (const {shape=[]} (l / r))

test_pow : IO ()
test_pow = do
  let x = [[3, 4, -5], [0, 0.3, 0]]
      y = [[1, -2.3, 0.2], [0.1, 0, 2]]
      expected = const {shape=[2, 3]} {dtype=F64} $ [
        [pow 3 1, pow 4 (-2.3), pow (-5) 0.2],
        [pow 0 0.1, pow 0.3 0, pow 0 2]
      ]
  assertAll ("^ for F64 array") $ sufficientlyEq ((const x) ^ (const y)) expected

  sequence_ $ do
    l <- the (List Double) [-3.4, -1.1, -0.1, 0.0, 0.1, 1.1, 3.4]
    r <- the (List Double) [-3.4, -1.1, -0.1, 0.1, 1.1, 3.4]
    pure $ assertAll ("^ for F64 scalar " ++ show l ++ " " ++ show r) $
      sufficientlyEq ((const l) ^ (const r)) $ const {shape=[]} {dtype=F64} (pow l r)

test_abs : IO ()
test_abs = do
  let x = const {shape=[_]} {dtype=S32} [1, 0, -5]
  assertAll "abs for int array" $ abs x == const [1, 0, 5]

  let x = const {shape=[3]} {dtype=F64} [1.8, -1.3, 0.0]
      actual = toArray (abs x)
  sequence_ (zipWith ((assert "abs for double array") .: sufficientlyEq) actual [1.8, 1.3, 0.0])

  sequence_ $ do
    x <- ints
    pure $ assertAll "abs for int scalar" $
      abs (const {shape=[]} {dtype=S32} x) == const (abs x)

  traverse_ (\x =>
      let actual = toArray (abs $ const {shape=[]} {dtype=F64} x)
       in assert "abs for double scalar" (sufficientlyEq actual (abs x))
    ) doubles

namespace S32
  export
  testElementwiseUnary : String -> (Int -> Int)
                         -> (forall shape . Tensor shape S32 -> Tensor shape S32) -> IO ()
  testElementwiseUnary name f_native f_tensor = do
    let x = [[1, 15, -5], [-1, 7, 0]]
        expected = const {shape=[_, _]} {dtype=S32} (map (map f_native) x)
    assertAll (name ++ " for S32 array") $ f_tensor (const x) == expected

    sequence_ [
        assertAll (name ++ " for S32 scalar " ++ show x) $ 
        (f_tensor $ const x) == (const {shape=[]} (f_native x)) | x <- ints
      ]

namespace F64
  export
  testElementwiseUnary : String -> (Double -> Double)
                         -> (forall shape . Tensor shape F64 -> Tensor shape F64) -> IO ()
  testElementwiseUnary name f_native f_tensor = do
    let x = [[1.3, 1.5, -5.2], [-1.1, 7.0, 0.0]]
        expected = const {shape=[_, _]} {dtype=F64} (map (map f_native) x)
    assertAll (name ++ " for F64 array") $ sufficientlyEq (f_tensor (const x)) expected

    sequence_ [
        assertAll (name ++ " for F64 scalar " ++ show x) $ sufficientlyEq
        (f_tensor $ const x) (const {shape=[]} (f_native x)) | x <- doubles
      ]

test_negate : IO ()
test_negate = do
  S32.testElementwiseUnary "negate" negate negate
  F64.testElementwiseUnary "negate" negate negate

tanh' : Double -> Double
tanh' x =
  if x == -inf then -1.0
  else if x == inf then 1.0
  else tanh x

testElementwiseUnaryDoubleCases : IO ()
testElementwiseUnaryDoubleCases = do
  F64.testElementwiseUnary "exp" exp exp
  F64.testElementwiseUnary "floor" floor floor
  F64.testElementwiseUnary "ceil" ceiling ceil
  F64.testElementwiseUnary "log" log log
  F64.testElementwiseUnary "logistic" (\x => 1 / (1 + exp (-x))) logistic
  F64.testElementwiseUnary "sin" sin sin
  F64.testElementwiseUnary "cos" cos cos
  F64.testElementwiseUnary "tanh" tanh' tanh
  F64.testElementwiseUnary "sqrt" sqrt sqrt

test_erf : IO ()
test_erf = do
  let x = const {shape=[_]} [-1.5, -0.5, 0.5, 1.5]
      expected = const [-0.96610516, -0.5204998, 0.5204998, 0.9661051]
  assertAll "erf agrees with tfp Normal" $ sufficientlyEq {tol=0.000001} (erf x) expected

min' : Double -> Double -> Double
min' x y = if (x /= x) then x else if (y /= y) then y else min x y

test_min : IO ()
test_min = do
  S32.testElementwiseBinary "min" min min
  F64.testElementwiseBinary "min" min' min

test_Min : IO ()
test_Min = do
  let x = const {shape=[_, _]} {dtype=F64} [[1.1, 2.1, -2.0], [-1.3, -1.0, 1.0]]
  assertAll "Min neutral is neutral right" $ (<+>) @{Min} x (neutral @{Min}) == x
  assertAll "Min neutral is neutral left" $ (<+>) @{Min} (neutral @{Min}) x == x

max' : Double -> Double -> Double
max' x y = if (x /= x) then x else if (y /= y) then y else max x y

test_max : IO ()
test_max = do
  S32.testElementwiseBinary "max" max max
  F64.testElementwiseBinary "max" max' max

test_Max : IO ()
test_Max = do
  let x = const {shape=[_, _]} {dtype=F64} [[1.1, 2.1, -2.0], [-1.3, -1.0, 1.0]]
  assertAll "Max neutral is neutral right" $ (<+>) @{Max} x (neutral @{Max}) == x
  assertAll "Max neutral is neutral left" $ (<+>) @{Max} (neutral @{Max}) x == x

test_cholesky : IO ()
test_cholesky = do
  let x = const {shape=[_, _]} {dtype=F64} [[1, 0], [2, 0]]
      expected = const [[nan, nan], [nan, nan]]
  assertAll "cholesky zero determinant" $ sufficientlyEq (cholesky x) expected

  -- example generated with tensorflow
  let x = const {shape=[_, _]} {dtype=F64} [
              [ 2.236123  ,  0.70387983,  2.8447943 ],
              [ 0.7059226 ,  2.661426  , -0.8714733 ],
              [ 1.3730898 ,  1.4064665 ,  2.7474475 ]
            ]
      expected = const [
              [1.4953672 , 0.0       , 0.0       ],
              [0.47207308, 1.5615932 , 0.0       ],
              [0.9182292 , 0.6230785 , 1.2312902 ]
            ]
  assertAll "cholesky" $ sufficientlyEq {tol=0.000001} (cholesky x) expected

test_triangularsolve : IO ()
test_triangularsolve = do
  let a = const {shape=[_, _]} [
              [0.8578532 , 0.0       , 0.0       ],
              [0.2481904 , 0.9885198 , 0.0       ],
              [0.59390426, 0.14998078, 0.19468737]
            ]
      b = const {shape=[_, _]} [
              [0.45312142, 0.37276268],
              [0.9210588 , 0.00647926],
              [0.7890165 , 0.77121615]
            ]
      actual = a |\ b
      expected = const {shape=[_, _]} [
                    [ 0.52820396,  0.43452972],
                    [ 0.79913783, -0.10254406],
                    [ 1.8257918 ,  2.7147462 ]
                  ]
  assertAll "(|\) result" $ sufficientlyEq {tol=0.000001} actual expected
  assertAll "(|\) is invertible with (@@)" $ sufficientlyEq {tol=0.000001} (a @@ actual) b

  let actual = a.T \| b
      expected = const {shape=[_, _]} [
                    [-2.3692384 , -2.135952  ],
                    [ 0.31686386, -0.594465  ],
                    [ 4.0527363 ,  3.9613056 ]
                  ]
  assertAll "(\|) result" $ sufficientlyEq {tol=0.000001} actual expected
  assertAll "(\|) is invertible with (@@)" $ sufficientlyEq {tol=0.000001} (a.T @@ actual) b

  let a = const {shape=[_, _]} [[1, 2], [3, 4]]
      a_lt = const {shape=[_, _]} [[1, 0], [3, 4]]
      b = const {shape=[_]} [5, 6]
  assertAll "(|\) upper triangular elements are ignored" $ sufficientlyEq (a |\ b) (a_lt |\ b)

  let a_ut = const {shape=[_, _]} [[1, 2], [0, 4]]
  assertAll "(\|) lower triangular elements are ignored" $ sufficientlyEq (a \| b) (a_ut \| b)

test_trace : IO ()
test_trace = do
  assertAll "trace" $ trace (const {dtype=S32} [[-1, 5], [1, 4]]) == const 3

export
test : IO ()
test = do
  test_const_toArray
  test_show_graph
  test_show_graphxla
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
  test_map
  test_map2
  test_reduce
  test_elementwise_equality
  test_elementwise_inequality
  test_comparison
  Vector.test_dot
  Matrix.test_dot
  test_add
  test_Sum
  test_subtract
  test_elementwise_multiplication
  test_scalar_multiplication
  test_Prod
  test_elementwise_division
  test_scalar_division
  test_pow
  test_elementwise_and
  test_All
  test_elementwise_or
  test_Any
  test_elementwise_not
  test_select
  test_cond
  test_abs
  test_negate
  testElementwiseUnaryDoubleCases
  test_erf
  test_min
  test_Min
  test_max
  test_Max
  test_cholesky
  test_triangularsolve
  test_trace
