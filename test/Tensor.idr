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
import Data.Nat
import Tensor

test_can_construct_scalar : Tensor [] Double
test_can_construct_scalar = const 0.0

test_can_construct_vector : Tensor [3] Double
test_can_construct_vector = const [0.0, 1.0, -2.0]

test_can_construct_matrix : Tensor [2, 3] Double
test_can_construct_matrix = const [[0.0, -1.0, -2.0], [3.0, 4.0, 5.0]]

test_can_construct_int_matrix : Tensor [2, 3] Int
test_can_construct_int_matrix = const [[0, -1, -2], [3, 4, 5]]

test_T : Tensor [2, 3] Double -> Tensor [3, 2] Double
test_T x = x.T

test_T_with_leading : Tensor [2, 3, 5] Double -> Tensor [2, 5, 3] Double
test_T_with_leading x = x.T

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

test_broadcastable : List (
    fr ** tr ** from : Shape {rank=fr} ** to : Shape {rank=tr} ** Broadcastable from to
)
test_broadcastable = [
    (_ ** _ ** [] ** [] ** Same),
    (_ ** _ ** [3, 2, 5] ** [3, 2, 5] ** Same),
    (_ ** _ ** [] ** [3, 2, 5] ** Nest $ Nest $ Nest Same),
    (_ ** _ ** [3, 1, 5] ** [3, 7, 5] ** Match $ Match Same),
    (_ ** _ ** [3, 2, 5] ** [1, 3, 2, 5] ** Nest Same),
    (_ ** _ ** [3, 2, 5] ** [7, 3, 2, 5] ** Nest Same)
]

test_broadcastable_cannot_reduce_rank0 : Broadcastable [5] [] -> Void
test_broadcastable_cannot_reduce_rank0 _ impossible

test_broadcastable_cannot_reduce_rank1 : Broadcastable [3, 2, 5] [] -> Void
test_broadcastable_cannot_reduce_rank1 _ impossible

test_broadcastable_cannot_stack_dimension_gt_one : Broadcastable [3, 2] [3, 7] -> Void
test_broadcastable_cannot_stack_dimension_gt_one (Match Same) impossible
test_broadcastable_cannot_stack_dimension_gt_one (Nest Same) impossible

test_squeezable_can_noop : Squeezable [3, 2, 5] [3, 2, 5]
test_squeezable_can_noop = Same

test_squeezable_can_remove_ones : Squeezable [1, 3, 1, 1, 2, 5, 1] [3, 2, 5]
test_squeezable_can_remove_ones = Nest (Match (Nest (Nest (Match (Match (Nest Same))))))

test_squeezable_can_flatten_only_ones : Squeezable [1, 1] []
test_squeezable_can_flatten_only_ones = Nest (Nest Same)

test_squeezable_cannot_remove_non_ones : Squeezable [1, 2] [] -> Void
test_squeezable_cannot_remove_non_ones (Nest _) impossible

test_tensor_contraction11 : Tensor [4] Double -> Tensor [4] Double -> Tensor [] Double
test_tensor_contraction11 x y = x @@ y

test_tensor_contraction12 : Tensor [4] Double -> Tensor [4, 5] Double -> Tensor [5] Double
test_tensor_contraction12 x y = x @@ y

test_tensor_contraction21 : Tensor [3, 4] Double -> Tensor [4] Double -> Tensor [3] Double
test_tensor_contraction21 x y = x @@ y

test_tensor_contraction22 : Tensor [3, 4] Double -> Tensor [4, 5] Double -> Tensor [3, 5] Double
test_tensor_contraction22 x y = x @@ y

test_det : Tensor [3, 3] Double -> Tensor [] Double
test_det x = det x

test_det_with_leading : Tensor [2, 3, 3] Double -> Tensor [2] Double
test_det_with_leading x = det x

assert : Bool -> IO ()
assert x = putStrLn $ if x then "PASS" else "FAIL"

test_const_eval : IO ()
test_const_eval = do
    let x = [[True, False, False], [False, True, False]]
    x' <- eval $ const {shape=[_, _]} {dtype=Bool} x
    assert $ x' == x

    let x =  [[1, 15, 5], [-1, 7, 6]]
    x' <- eval $ const {shape=[_, _]} {dtype=Int} x
    assert $ x' == x

    x <- eval $ const {shape=[_, _]} {dtype=Double} [[-1.5], [1.3], [4.3]]
    assert $ abs (index 0 (index 0 x) - (-1.5)) < 0.000001
    assert $ abs (index 0 (index 1 x) - 1.3) < 0.000001
    assert $ abs (index 0 (index 2 x) - 4.3) < 0.000001

    x <- eval $ const {shape=[]} True
    assert x

    x <- eval $ const {shape=[]} {dtype=Int} 3
    assert $ x == 3

    x <- eval $ const {shape=[]} {dtype=Double} 3.4
    assert $ abs (x - 3.4) < 0.000001

test_toString : IO ()
test_toString = do
    str <- toString $ const {shape=[]} {dtype=Int} 1
    assert $ str == "constant, shape=[], metadata={:0}"

    let x = const {shape=[]} {dtype=Int} 1
        y = const {shape=[]} {dtype=Int} 2
    str <- toString (x + y)
    assert $ str ==
        """
        add, shape=[], metadata={:0}
          constant, shape=[], metadata={:0}
          constant, shape=[], metadata={:0}
        """

    str <- toString $ const {shape=[_]} {dtype=Double} [1.3, 2.0, -0.4]
    assert $ str == "constant, shape=[3], metadata={:0}"

test_broadcast : IO ()
test_broadcast = do
    let x = const {shape=[]} {dtype=Int} 7
    broadcasted <- eval $ broadcast {to=[]} x
    assert $ broadcasted == 7

    let x = const {shape=[]} {dtype=Int} 7
    broadcasted <- eval $ broadcast {to=[1]} x
    assert $ broadcasted == [7]

    let x = const {shape=[]} {dtype=Int} 7
    broadcasted <- eval $ broadcast {to=[2, 3]} x
    assert $ broadcasted == [[7, 7, 7], [7, 7, 7]]

    let x = const {shape=[]} {dtype=Int} 7
    broadcasted <- eval $ broadcast {to=[1, 1, 1]} x
    assert $ broadcasted == [[[7]]]

    let x = const {shape=[1]} {dtype=Int} [7]
    broadcasted <- eval $ broadcast {to=[0]} x
    assert $ broadcasted == []

    let x = const {shape=[1]} {dtype=Int} [7]
    broadcasted <- eval $ broadcast {to=[1]} x
    assert $ broadcasted == [7]

    let x = const {shape=[1]} {dtype=Int} [7]
    broadcasted <- eval $ broadcast {to=[3]} x
    assert $ broadcasted == [7, 7, 7]

    let x = const {shape=[1]} {dtype=Int} [7]
    broadcasted <- eval $ broadcast {to=[2, 3]} x
    assert $ broadcasted == [[7, 7, 7], [7, 7, 7]]

    let x = const {shape=[2]} {dtype=Int} [5, 7]
    broadcasted <- eval $ broadcast {to=[2, 0]} x
    assert $ broadcasted == [[], []]

    let x = const {shape=[2]} {dtype=Int} [5, 7]
    broadcasted <- eval $ broadcast {to=[3, 2]} x
    assert $ broadcasted == [[5, 7], [5, 7], [5, 7]]

    let x = const {shape=[2, 3]} {dtype=Int} [[2, 3, 5], [7, 11, 13]]
    broadcasted <- eval $ broadcast {to=[2, 3]} x
    assert $ broadcasted == [[2, 3, 5], [7, 11, 13]]

    let x = const {shape=[2, 3]} {dtype=Int} [[2, 3, 5], [7, 11, 13]]
    broadcasted <- eval $ broadcast {to=[2, 0]} x
    assert $ broadcasted == [[], []]

    let x = const {shape=[2, 3]} {dtype=Int} [[2, 3, 5], [7, 11, 13]]
    broadcasted <- eval $ broadcast {to=[0, 3]} x
    assert $ broadcasted == []

    let x = const {shape=[2, 3]} {dtype=Int} [[2, 3, 5], [7, 11, 13]]
    broadcasted <- eval $ broadcast {to=[2, 2, 3]} x
    assert $ broadcasted == [[[2, 3, 5], [7, 11, 13]], [[2, 3, 5], [7, 11, 13]]]

    let x = const {shape=[2, 1, 3]} {dtype=Int} [[[2, 3, 5]], [[7, 11, 13]]]
    broadcasted <- eval $ broadcast {to=[2, 2, 5, 3]} x
    assert $ broadcasted == [
        [
            [[2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5]],
            [[7, 11, 13], [7, 11, 13], [7, 11, 13], [7, 11, 13], [7, 11, 13]]
        ],
        [
            [[2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5]],
            [[7, 11, 13], [7, 11, 13], [7, 11, 13], [7, 11, 13], [7, 11, 13]]
        ]
    ]

test_add : IO ()
test_add = do
    let x = const {shape=[_, _]} {dtype=Int} [[1, 15, 5], [-1, 7, 6]]
        y = const {shape=[_, _]} {dtype=Int} [[11, 5, 7], [-3, -4, 0]]
    sum <- eval (x + y)
    assert $ sum == [[12, 20, 12], [-4, 3, 6]]

    let x = const {shape=[_, _]} {dtype=Double} [[1.8], [1.3], [4.0]]
        y = const {shape=[_, _]} {dtype=Double} [[-3.3], [0.0], [0.3]]
    sum <- eval (x + y)
    assert $ abs (index 0 (index 0 sum) - (-1.5)) < 0.000001
    assert $ abs (index 0 (index 1 sum) - 1.3) < 0.000001
    assert $ abs (index 0 (index 2 sum) - 4.3) < 0.000001

    let x = const {shape=[]} {dtype=Int} 3
        y = const {shape=[]} {dtype=Int} (-7)
    sum <- eval (x + y)
    assert $ sum == -4

    let x = const {shape=[]} {dtype=Double} 3.4
        y = const {shape=[]} {dtype=Double} (-7.1)
    sum <- eval (x + y)
    assert $ abs (sum - (-3.7)) < 0.000001

    let x = const {shape=[1]} {dtype=Int} [3]
        y = const {shape=[]} {dtype=Int} 5
    sum <- eval (x + y)
    assert $ sum == [8]

    let x = const {shape=[2]} {dtype=Int} [3, 5]
        y = const {shape=[]} {dtype=Int} 7
    sum <- eval (x + y)
    assert $ sum == [10, 12]

    let x = const {shape=[2, 3]} {dtype=Int} [[2, 3, 5], [7, 11, 13]]
        y = const {shape=[3]} {dtype=Int} [17, 19, 23]
    sum <- eval (x + y)
    assert $ sum == [[19, 22, 28], [24, 30, 36]]

    let x = const {shape=[2, 0]} {dtype=Int} [[], []]
        y = const {shape=[2, 3]} {dtype=Int} [[2, 3, 5], [7, 11, 13]]
    sum <- eval (x + y)
    assert $ sum == [[], []]

test : IO ()
test = do
    test_const_eval
    test_toString
    test_broadcast
    test_add
