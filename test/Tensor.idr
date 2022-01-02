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
import System

import Tensor

floatingPointTolerance : Double
floatingPointTolerance = 0.00000001

doubleSufficientlyEq : Double -> Double -> Bool
doubleSufficientlyEq x y = abs (x - y) < floatingPointTolerance

assert : Bool -> IO ()
assert x = if x then pure () else do
    putStrLn "Test failed"
    exitFailure

||| Assert an element-wise relation holds betweens two tensors.
assertRelation : {shape : _} ->
    (Tensor shape dtype -> Tensor shape dtype -> Tensor shape Bool)
    -> Tensor shape dtype -> Tensor shape dtype -> IO ()
assertRelation prop x y = do
    satisfies_prop <- eval (prop x y)
    assert (arrayAll satisfies_prop) where
        arrayAll : {shape : _} -> Array shape {dtype=Bool} -> Bool
        arrayAll {shape = []} x = x
        arrayAll {shape = (0 :: _)} [] = True
        arrayAll {shape = ((S d) :: ds)} (x :: xs) = arrayAll x && arrayAll {shape=(d :: ds)} xs

-- WARNING: This uses (-) (==#) (<#) and absE, and thus assumes they work, so
-- we shouldn't use it to test those functions.
fpEq : {shape : _} -> Tensor shape Double -> Tensor shape Double -> Tensor shape Bool
fpEq x y = absE (x - y) <# fill floatingPointTolerance

bools : List Bool
bools = [True, False]

ints : List Int
ints = [-3, -1, 0, 1, 3]

doubles : List Double
doubles = [-3.4, -1.1, -0.1, 0.0, 0.1, 1.1, 3.4]

test_const_eval : IO ()
test_const_eval = do
    let x = [[True, False, False], [False, True, False]]
    x' <- eval $ const {shape=[_, _]} {dtype=Bool} x
    assert $ x' == x

    let x =  [[1, 15, 5], [-1, 7, 6]]
    x' <- eval $ const {shape=[_, _]} {dtype=Int} x
    assert $ x' == x

    x <- eval $ const {shape=[_, _]} {dtype=Double} [[-1.5], [1.3], [4.3]]
    assert $ doubleSufficientlyEq (index 0 (index 0 x)) (-1.5)
    assert $ doubleSufficientlyEq (index 0 (index 1 x)) 1.3
    assert $ doubleSufficientlyEq (index 0 (index 2 x)) 4.3

    traverse_ (\x => do x' <- eval {shape=[]} (const x); assert (x == x')) bools
    traverse_ (\x => do x' <- eval {shape=[]} (const x); assert (x == x')) ints
    traverse_ (\x => do
            x' <- eval {shape=[]} (const x)
            assert (doubleSufficientlyEq x x')
        ) doubles

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
    assertRelation (==#) (broadcast {to=[]} x) (const 7)

    let x = const {shape=[]} {dtype=Int} 7
    assertRelation (==#) (broadcast {to=[1]} x) (const [7])

    let x = const {shape=[]} {dtype=Int} 7
    assertRelation (==#) (broadcast {to=[2, 3]} x) (const [[7, 7, 7], [7, 7, 7]])

    let x = const {shape=[]} {dtype=Int} 7
    assertRelation (==#) (broadcast {to=[1, 1, 1]} x) (const [[[7]]])

    let x = const {shape=[1]} {dtype=Int} [7]
    assertRelation (==#) (broadcast {to=[0]} x) (const [])

    let x = const {shape=[1]} {dtype=Int} [7]
    assertRelation (==#) (broadcast {to=[1]} x) (const [7])

    let x = const {shape=[1]} {dtype=Int} [7]
    assertRelation (==#) (broadcast {to=[3]} x) (const [7, 7, 7])

    let x = const {shape=[1]} {dtype=Int} [7]
    assertRelation (==#) (broadcast {to=[2, 3]} x) (const [[7, 7, 7], [7, 7, 7]])

    let x = const {shape=[2]} {dtype=Int} [5, 7]
    assertRelation (==#) (broadcast {to=[2, 0]} x) (const [[], []])

    let x = const {shape=[2]} {dtype=Int} [5, 7]
    assertRelation (==#) (broadcast {to=[3, 2]} x) (const [[5, 7], [5, 7], [5, 7]])

    let x = const {shape=[2, 3]} {dtype=Int} [[2, 3, 5], [7, 11, 13]]
    assertRelation (==#) (broadcast {to=[2, 3]} x) (const [[2, 3, 5], [7, 11, 13]])

    let x = const {shape=[2, 3]} {dtype=Int} [[2, 3, 5], [7, 11, 13]]
    assertRelation (==#) (broadcast {to=[2, 0]} x) (const [[], []])

    let x = const {shape=[2, 3]} {dtype=Int} [[2, 3, 5], [7, 11, 13]]
    assertRelation (==#) (broadcast {to=[0, 3]} x) (const [])

    let x = const {shape=[2, 3]} {dtype=Int} [[2, 3, 5], [7, 11, 13]]
        expected = const [[[2, 3, 5], [7, 11, 13]], [[2, 3, 5], [7, 11, 13]]]
    assertRelation (==#) (broadcast {to=[2, 2, 3]} x) expected

    let x = const {shape=[2, 1, 3]} {dtype=Int} [[[2, 3, 5]], [[7, 11, 13]]]
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
    assertRelation (==#) (broadcast {to=[2, 2, 5, 3]} x) expected

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

test_T : Tensor [2, 3] Double -> Tensor [3, 2] Double
test_T x = x.T

test_T_with_leading : Tensor [2, 3, 5] Double -> Tensor [2, 5, 3] Double
test_T_with_leading x = x.T

test_elementwise_equality : IO ()
test_elementwise_equality = do
    let x = const {shape=[_]} {dtype=Bool} [True, True, False]
        y = const {shape=[_]} {dtype=Bool} [False, True, False]
    eq <- eval (y ==# x)
    assert $ eq == [False, True, True]

    let x = const {shape=[_, _]} {dtype=Int} [[1, 15, 5], [-1, 7, 6]]
        y = const {shape=[_, _]} {dtype=Int} [[2, 15, 3], [2, 7, 6]]
    eq <- eval (y ==# x)
    assert $ eq == [[False, True, False], [False, True, True]]

    let x = const {shape=[_, _]} {dtype=Double} [[1.1, 15.3, 5.2], [-1.6, 7.1, 6.0]]
        y = const {shape=[_, _]} {dtype=Double} [[2.2, 15.3, 3.4], [2.6, 7.1, 6.0]]
    eq <- eval (y ==# x)
    assert $ eq == [[False, True, False], [False, True, True]]

    sequence_ [compareScalars x y | x <- bools, y <- bools]
    sequence_ [compareScalars x y | x <- ints, y <- ints]
    sequence_ [compareScalars x y | x <- doubles, y <- doubles]

    where
        compareScalars : (Primitive dtype, Eq dtype) => dtype -> dtype -> IO ()
        compareScalars l r = do
            actual <- eval {shape=[]} ((const l) ==# (const r))
            assert (actual == (l == r))

test_elementwise_inequality : IO ()
test_elementwise_inequality = do
    let x = const [True, True, False]
        y = const [False, True, False]
    assertRelation (==#) (y /=# x) (const {shape=[_]} {dtype=Bool} [True, False, False])

    let x = const {shape=[_, _]} {dtype=Int} [[1, 15, 5], [-1, 7, 6]]
        y = const {shape=[_, _]} {dtype=Int} [[2, 15, 3], [2, 7, 6]]
    assertRelation (==#) (x /=# y) (const [[True, False, True], [True, False, False]])

    let x = const {shape=[_, _]} {dtype=Double} [[1.1, 15.3, 5.2], [-1.6, 7.1, 6.0]]
        y = const {shape=[_, _]} {dtype=Double} [[2.2, 15.3, 3.4], [2.6, 7.1, 6.0]]
    assertRelation (==#) (x /=# y) (const [[True, False, True], [True, False, False]])

    sequence_ [compareScalars l r | l <- bools, r <- bools]
    sequence_ [compareScalars l r | l <- ints, r <- ints]
    sequence_ [compareScalars l r | l <- doubles, r <- doubles]

    where
        compareScalars : (Primitive dtype, Eq dtype) => dtype -> dtype -> IO ()
        compareScalars l r =
            assertRelation (==#) ((const l) /=# (const r)) (const {shape=[]} (l /= r))

test_comparison : IO ()
test_comparison = do
    let x = const {shape=[_, _]} {dtype=Int} [[1, 2, 3], [-1, -2, -3]]
        y = const {shape=[_, _]} {dtype=Int} [[1, 4, 2], [-2, -1, -3]]
    assertRelation (==#) (y ># x) (const [[False, True, False], [False, True, False]])
    assertRelation (==#) (y <# x) (const [[False, False, True], [True, False, False]])
    assertRelation (==#) (y >=# x) (const [[True, True, False], [False, True, True]])
    assertRelation (==#) (y <=# x) (const [[True, False, True], [True, False, True]])

    let x = const {shape=[_, _]} {dtype=Double} [[1.1, 2.2, 3.3], [-1.1, -2.2, -3.3]]
        y = const {shape=[_, _]} {dtype=Double} [[1.1, 4.4, 2.2], [-2.2, -1.1, -3.3]]
    assertRelation (==#) (y ># x) (const [[False, True, False], [False, True, False]])
    assertRelation (==#) (y <# x) (const [[False, False, True], [True, False, False]])
    assertRelation (==#) (y >=# x) (const [[True, True, False], [False, True, True]])
    assertRelation (==#) (y <=# x) (const [[True, False, True], [True, False, True]])

    sequence_ [compareScalars l r | l <- ints, r <- ints]
    sequence_ [compareScalars l r | l <- doubles, r <- doubles]

    where
        compareScalars : (Primitive dtype, Ord dtype) => dtype -> dtype -> IO ()
        compareScalars l r = do
            let l' = const l
                r' = const r
            assertRelation (==#) (l' ># r') (const {shape=[]} (l > r))
            assertRelation (==#) (l' <# r') (const {shape=[]} (l < r))
            assertRelation (==#) (l' >=# r') (const {shape=[]} (l >= r))
            assertRelation (==#) (l' <=# r') (const {shape=[]} (l <= r))

test_tensor_contraction11 : Tensor [4] Double -> Tensor [4] Double -> Tensor [] Double
test_tensor_contraction11 x y = x @@ y

test_tensor_contraction12 : Tensor [4] Double -> Tensor [4, 5] Double -> Tensor [5] Double
test_tensor_contraction12 x y = x @@ y

test_tensor_contraction21 : Tensor [3, 4] Double -> Tensor [4] Double -> Tensor [3] Double
test_tensor_contraction21 x y = x @@ y

test_tensor_contraction22 : Tensor [3, 4] Double -> Tensor [4, 5] Double -> Tensor [3, 5] Double
test_tensor_contraction22 x y = x @@ y

test_add : IO ()
test_add = do
    let x = const [[1, 15, 5], [-1, 7, 6]]
        y = const [[11, 5, 7], [-3, -4, 0]]
    assertRelation (==#) (x + y) (const {shape=[_, _]} {dtype=Int} [[12, 20, 12], [-4, 3, 6]])

    let x = const [[1.8], [1.3], [4.0]]
        y = const [[-3.3], [0.0], [0.3]]
    assertRelation fpEq (x + y) $ const {shape=[_, _]} {dtype=Double} [[-1.5], [1.3], [4.3]]

    -- todo generalise
    let x = const 3
        y = const (-7)
    assertRelation (==#) (x + y) (const {shape=[]} {dtype=Int} (-4))

    -- todo generalise
    assertRelation fpEq (const 3.4 + const (-7.1)) $ const {shape=[]} {dtype=Double} (-3.7)

test_subtract : IO ()
test_subtract = do
    let l = const [[1, 15, 5], [-1, 7, 6]]
        r = const [[11, 5, 7], [-3, -4, 0]]
    assertRelation (==#) (l - r) $ const {shape=[_, _]} {dtype=Int} [[-10, 10, -2], [2, 11, 6]]

    let l = const [1.8, 1.3, 4.0]
        r = const [-3.3, 0.0, 0.3]
    diff <- eval {shape=[3]} {dtype=Double} (l - r)
    sequence_ (zipWith (assert .: doubleSufficientlyEq) diff [5.1, 1.3, 3.7])

    sequence_ [assertRelation (==#) (const l - const r) (const {shape=[]} (l - r))
                | l <- ints, r <- ints]
    sequence_ [do diff <- eval {shape=[]} (const l - const r)
                  assert (doubleSufficientlyEq diff (l - r)) | l <- doubles, r <- doubles]

test_elementwise_multiplication : IO ()
test_elementwise_multiplication = do
    let x = const [[1, 15, 5], [-1, 7, 6]]
        y = const [[11, 5, 7], [-3, -4, 0]]
    assertRelation (==#) (x *# y) (const {shape=[_, _]} {dtype=Int} [[11, 75, 35], [3, -28, 0]])

    let x = const [[1.8], [1.3], [4.0]]
        y = const [[-3.3], [0.0], [0.3]]
    assertRelation fpEq (x *# y) $ const {shape=[_, _]} {dtype=Double} [[-1.8 * 3.3], [0.0], [1.2]]

    -- todo generalise
    let x = const {shape=[]} {dtype=Int} 3
        y = const {shape=[]} {dtype=Int} (-7)
    assertRelation (==#) (x *# y) (const (-21))

    -- todo generalise
    assertRelation fpEq (const 3.4 *# const (-7.1)) $ const {shape=[]} {dtype=Double} (-3.4 * 7.1)

test_constant_multiplication : IO ()
test_constant_multiplication = do
    let x = const {shape=[]} {dtype=Int} 2
        y = const {shape=[_, _]} {dtype=Int} [[11, 5, 7], [-3, -4, 0]]
    assertRelation (==#) (x * y) (const [[22, 10, 14], [-6, -8, 0]])

    -- todo generalise
    let x = const 2.3
        y = const [[-3.3], [0.0], [0.3]]
    assertRelation fpEq (x * y) $ const {shape=[_, _]} {dtype=Double} [[-7.59], [0.0], [0.69]]

    -- todo generalise
    let x = const {shape=[]} {dtype=Int} 3
        y = const {shape=[]} {dtype=Int} (-7)
    assertRelation (==#) (x * y) (const (-21))

    -- todo generalise
    assertRelation fpEq (const 3.4 * const (-7.1)) $ const {shape=[]} {dtype=Double} (-3.4 * 7.1)

test_absE : IO ()
test_absE = do
    let x = const {shape=[_]} {dtype=Int} [1, 0, -5]
    assertRelation (==#) (absE x) (const [1, 0, 5])

    let x = const {shape=[3]} {dtype=Double} [1.8, -1.3, 0.0]
    actual <- eval (absE x)
    assert $ all (== True) (zipWith doubleSufficientlyEq actual [1.8, 1.3, 0.0])

    traverse_ (\x => assertRelation (==#) (absE (const {shape=[]} x)) (const (abs x))) ints
    traverse_ (\x => do
            actual <- eval (absE $ const {shape=[]} x)
            assert (doubleSufficientlyEq actual (abs x))
        ) doubles

test_det : Tensor [3, 3] Double -> Tensor [] Double
test_det x = det x

test_det_with_leading : Tensor [2, 3, 3] Double -> Tensor [2] Double
test_det_with_leading x = det x

main : IO ()
main = do
    test_const_eval
    test_toString
    test_broadcast
    test_elementwise_equality
    test_elementwise_inequality
    test_comparison
    test_add
    test_subtract
    test_elementwise_multiplication
    test_constant_multiplication
    test_absE
    putStrLn "Tests passed"
