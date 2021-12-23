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

test_add : IO ()
test_add = do let x = const {shape=[2, 3]} {dtype=Int} [[1, 15, 5], [-1, 7, 6]]
                  y = const {shape=[2, 3]} {dtype=Int} [[11, 5, 7], [-3, -4, 0]]
              sum <- eval {shape=[2, 3]} (x + y)
              assert $ sum == [[12, 20, 12], [-4, 3, 6]]

              let x = const {shape=[3, 1]} {dtype=Double} [[1.8], [1.3], [4.0]]
                  y = const {shape=[3, 1]} {dtype=Double} [[-3.3], [0.0], [0.3]]
              sum <- eval {shape=[3, 1]} {dtype=Double} (x + y)
              assert $ abs (index 0 (index 0 sum) - (-1.5)) < 0.000001
              assert $ abs (index 0 (index 1 sum) - 1.3) < 0.000001
              assert $ abs (index 0 (index 2 sum) - 4.3) < 0.000001

              let x = const {shape=[]} {dtype=Int} 3
                  y = const {shape=[]} {dtype=Int} (-7)
              sum <- eval {shape=[]} {dtype=Int} (x + y)
              assert $ sum == -4

              let x = const {shape=[]} {dtype=Double} 3.4
                  y = const {shape=[]} {dtype=Double} (-7.1)
              sum <- eval {shape=[]} {dtype=Double} (x + y)
              assert $ abs (sum - (-3.7)) < 0.000001

test_toString : IO ()
test_toString = do let x = const {shape=[]} {dtype=Int} 1
                   str <- toString x
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

                   let x = const {shape=[3]} {dtype=Double} [1.3, 2.0, -0.4]
                   str <- toString x
                   assert $ str == "constant, shape=[3], metadata={:0}"

test : IO ()
test = do test_add
          test_toString
