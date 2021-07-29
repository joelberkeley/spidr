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

test_can_construct_integer_matrix : Tensor [2, 3] Integer
test_can_construct_integer_matrix = const [[0, -1, -2], [3, 4, 5]]

test_T : Tensor [2, 3] Double -> Tensor [3, 2] Double
test_T x = x.T

test_T_with_leading : Tensor [2, 3, 5] Double -> Tensor [2, 5, 3] Double
test_T_with_leading x = x.T

test_dimbroadcastable : List ((a ** b ** DimBroadcastable a b))
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

test_broadcastable : List (from ** to ** Broadcastable from to)

test_broadcastable_identity0 : Broadcastable [] []
test_broadcastable_identity0 = Same

test_broadcastable_identity1 : Broadcastable [3, 2, 5] [3, 2, 5]
test_broadcastable_identity1 = Same

test_broadcastable_scalar_to_any : Broadcastable [] [3, 2, 5]
test_broadcastable_scalar_to_any = Nest (Nest (Nest Same))

test_broadcastable_cannot_reduce_rank0 : Broadcastable [5] [] -> Void
test_broadcastable_cannot_reduce_rank0 _ impossible

test_broadcastable_cannot_reduce_rank1 : Broadcastable [3, 2, 5] [] -> Void
test_broadcastable_cannot_reduce_rank1 _ impossible

test_broadcastable_can_stack_inner_one : Broadcastable [3, 1, 5] [3, 7, 5]
test_broadcastable_can_stack_inner_one = Match (Match Same)

test_broadcastable_cannot_stack_dimension_gt_one : Broadcastable [3, 2] [3, 7] -> Void
test_broadcastable_cannot_stack_dimension_gt_one (Match Same) impossible
test_broadcastable_cannot_stack_dimension_gt_one (Nest Same) impossible

test_broadcastable_can_nest0 : Broadcastable [3, 2, 5] [1, 3, 2, 5]
test_broadcastable_can_nest0 = Nest Same

test_broadcastable_can_nest1 : Broadcastable [3, 2, 5] [7, 3, 2, 5]
test_broadcastable_can_nest1 = Nest Same

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
