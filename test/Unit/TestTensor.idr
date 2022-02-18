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

export
test_const_eval : IO ()
test_const_eval = do
    let x = [[True, False, False], [False, True, False]]
    x' <- eval $ const {shape=[_, _]} {dtype=PRED} x
    assert "const eval returns original Bool" (x' == x)

    let x =  [[1, 15, 5], [-1, 7, 6]]
    x' <- eval $ const {shape=[_, _]} {dtype=S32} x
    assert "const eval returns original Int" (x' == x)

    let name = "const eval returns original Double"
    x <- eval $ const {shape=[_, _]} {dtype=F64} [[-1.5], [1.3], [4.3]]
    assert name $ sufficientlyEq (index 0 (index 0 x)) (-1.5)
    assert name $ sufficientlyEq (index 0 (index 1 x)) 1.3
    assert name $ sufficientlyEq (index 0 (index 2 x)) 4.3

    let name = "const eval returns original scalar"
    traverse_ (\x => do x' <- eval {shape=[]} {dtype=PRED} (const x); assert name (x == x')) bools
    traverse_ (\x => do x' <- eval {shape=[]} {dtype=S32} (const x); assert name (x == x')) ints
    traverse_ (\x => do
            x' <- eval {shape=[]} {dtype=F64} (const x)
            assert name (sufficientlyEq x x')
        ) doubles

export
test_toString : IO ()
test_toString = do
    str <- toString $ const {shape=[]} {dtype=S32} 1
    assert "toString for scalar Int" (str == "constant, shape=[], metadata={:0}")

    let x = const {shape=[]} {dtype=S32} 1
        y = const {shape=[]} {dtype=S32} 2
    str <- toString (x + y)
    assert "toString for scalar addition" $ str ==
        """
        add, shape=[], metadata={:0}
          constant, shape=[], metadata={:0}
          constant, shape=[], metadata={:0}
        """

    str <- toString $ const {shape=[_]} {dtype=F64} [1.3, 2.0, -0.4]
    assert "toString for vector F64" $ str == "constant, shape=[3], metadata={:0}"

export
test_reshape : IO ()
test_reshape = do
    let x = const {shape=[]} {dtype=S32} 3
        expected = const {shape=[1]} {dtype=S32} [3]
    assertAll "reshape add dims scalar" $ reshape x ==# expected

    let x = const {shape=[3]} {dtype=S32} [3, 4, 5]
        flipped = const {shape=[3, 1]} {dtype=S32} [[3], [4], [5]]
    assertAll "reshape flip dims vector" $ reshape x ==# flipped

    let x = const {shape=[2, 3]} {dtype=S32} [[3, 4, 5], [6, 7, 8]]
        flipped = const {shape=[3, 2]} {dtype=S32} [[3, 4], [5, 6], [7, 8]]
    assertAll "reshape flip dims array" $ reshape x ==# flipped

    let with_extra_dim = const {shape=[2, 1, 3]} {dtype=S32} [[[3, 4, 5]], [[6, 7, 8]]]
    assertAll "reshape add dimension array" $ reshape x ==# with_extra_dim

    let flattened = const {shape=[6]} {dtype=S32} [3, 4, 5, 6, 7, 8]
    assertAll "reshape as flatten array" $ reshape x ==# flattened

export
test_slice : IO ()
test_slice = do
    let x = const {shape=[3]} {dtype=S32} [3, 4, 5]
    assertAll "slice vector 0 0" $ slice 0 0 0 x ==# const []
    assertAll "slice vector 0 1" $ slice 0 0 1 x ==# const [3]
    assertAll "slice vector 0 2" $ slice 0 0 2 x ==# const [3, 4]
    assertAll "slice vector 0 3" $ slice 0 0 3 x ==# const [3, 4, 5]
    assertAll "slice vector 1 1" $ slice 0 1 1 x ==# const []
    assertAll "slice vector 1 2" $ slice 0 1 2 x ==# const [4]
    assertAll "slice vector 1 3" $ slice 0 1 3 x ==# const [4, 5]
    assertAll "slice vector 2 2" $ slice 0 2 2 x ==# const []
    assertAll "slice vector 2 2" $ slice 0 2 3 x ==# const [5]

    let x = const {shape=[2, 3]} {dtype=S32} [[3, 4, 5], [6, 7, 8]]
    assertAll "slice array 0 0 1" $ slice 0 0 1 x ==# const [[3, 4, 5]]
    assertAll "slice array 0 1 1" $ slice 0 1 1 x ==# const []
    assertAll "slice array 1 2 2" $ slice 1 2 2 x ==# const [[], []]
    assertAll "slice array 1 1 3" $ slice 1 1 3 x ==# const [[4, 5], [7, 8]]

export
test_index : IO ()
test_index = do
    let x = const {shape=[3]} {dtype=S32} [3, 4, 5]
    assertAll "index vector 0" $ index 0 0 x ==# const 3
    assertAll "index vector 1" $ index 0 1 x ==# const 4
    assertAll "index vector 2" $ index 0 2 x ==# const 5

    let x = const {shape=[2, 3]} {dtype=S32} [[3, 4, 5], [6, 7, 8]]
    assertAll "index array 0 0" $ index 0 0 x ==# const [3, 4, 5]
    assertAll "index array 0 1" $ index 0 1 x ==# const [6, 7, 8]
    assertAll "index array 1 0" $ index 1 0 x ==# const [3, 6]
    assertAll "index array 1 1" $ index 1 1 x ==# const [4, 7]
    assertAll "index array 1 2" $ index 1 2 x ==# const [5, 8]

export
test_expand : IO ()
test_expand = do
    let x = const {shape=[]} {dtype=S32} 3
        expected = const {shape=[1]} {dtype=S32} [3]
    assertAll "expand add dims scalar" $ expand 0 x ==# expected

    let x = const {shape=[2, 3]} {dtype=S32} [[3, 4, 5], [6, 7, 8]]
        with_extra_dim = const {shape=[2, 1, 3]} {dtype=S32} [[[3, 4, 5]], [[6, 7, 8]]]
    assertAll "expand add dimension array" $ expand 1 x ==# with_extra_dim

export
test_broadcast : IO ()
test_broadcast = do
    let x = const {shape=[]} {dtype=S32} 7
    assertAll "broadcast scalar to itself" $ broadcast {to=[]} x ==# const 7

    let x = const {shape=[]} {dtype=S32} 7
    assertAll "broadcast scalar to rank 1" $ broadcast {to=[1]} x ==# const [7]

    let x = const {shape=[]} {dtype=S32} 7
    assertAll "broadcast scalar to rank 2" $
        broadcast {to=[2, 3]} x ==# const [[7, 7, 7], [7, 7, 7]]

    let x = const {shape=[]} {dtype=S32} 7
    assertAll "broadcast scalar to rank 3" $ broadcast {to=[1, 1, 1]} x ==# const [[[7]]]

    let x = const {shape=[1]} {dtype=S32} [7]
    assertAll "broadcast rank 1 to empty" $ broadcast {to=[0]} x ==# const []

    let x = const {shape=[1]} {dtype=S32} [7]
    assertAll "broadcast rank 1 to itself" $ broadcast {to=[1]} x ==# const [7]

    let x = const {shape=[1]} {dtype=S32} [7]
    assertAll "broadcast rank 1 to larger rank 1" $ broadcast {to=[3]} x ==# const [7, 7, 7]

    let x = const {shape=[1]} {dtype=S32} [7]
    assertAll "broadcast rank 1 to rank 2" $
        broadcast {to=[2, 3]} x ==# const [[7, 7, 7], [7, 7, 7]]

    let x = const {shape=[2]} {dtype=S32} [5, 7]
    assertAll "broadcast rank 1 to empty" $ broadcast {to=[2, 0]} x ==# const [[], []]

    let x = const {shape=[2]} {dtype=S32} [5, 7]
    assertAll "broadcast rank 1 to rank 2" $
        broadcast {to=[3, 2]} x ==# const [[5, 7], [5, 7], [5, 7]]

    let x = const {shape=[2, 3]} {dtype=S32} [[2, 3, 5], [7, 11, 13]]
    assertAll "broadcast rank 2 to itself" $
        broadcast {to=[2, 3]} x ==# const [[2, 3, 5], [7, 11, 13]]

    let x = const {shape=[2, 3]} {dtype=S32} [[2, 3, 5], [7, 11, 13]]
    assertAll "broadcast rank 2 to rank 2 empty" $ broadcast {to=[2, 0]} x ==# const [[], []]

    let x = const {shape=[2, 3]} {dtype=S32} [[2, 3, 5], [7, 11, 13]]
    assertAll "broadcast rank 2 to empty" $ broadcast {to=[0, 3]} x ==# const []

    let x = const {shape=[2, 3]} {dtype=S32} [[2, 3, 5], [7, 11, 13]]
        expected = const [[[2, 3, 5], [7, 11, 13]], [[2, 3, 5], [7, 11, 13]]]
    assertAll "broadcast rank 2 to rank 3" $ broadcast {to=[2, 2, 3]} x ==# expected

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
    assertAll "broadcast rank 3 to rank 4" $ broadcast {to=[2, 2, 5, 3]} x ==# expected

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

export
test_squeeze : IO ()
test_squeeze = do
    let x = const {shape=[1, 1]} {dtype=S32} [[3]]
        squeezed = const {shape=[]} {dtype=S32} 3
    assertAll "squeeze can flatten only ones" $ squeeze x ==# squeezed

    let x = const {shape=[2, 1, 3]} {dtype=S32} [[[3, 4, 5]], [[6, 7, 8]]]
    assertAll "squeeze can no-op" $ squeeze x ==# x

    let squeezed = const {shape=[2, 3]} {dtype=S32} [[3, 4, 5], [6, 7, 8]]
    assertAll "squeeze can remove dim from array" $ squeeze x ==# squeezed

    let x = fill {shape=[1, 3, 1, 1, 2, 5, 1]} {dtype=S32} 0
    assertAll "squeeze can remove many dims from array" $
        squeeze x ==# fill {shape=[3, 2, 5]} {dtype=S32} 0

test_squeezable_cannot_remove_non_ones : Squeezable [1, 2] [] -> Void
test_squeezable_cannot_remove_non_ones (Nest _) impossible

test_T : Tensor [2, 3] F64 -> Tensor [3, 2] F64
test_T x = x.T

test_T_with_leading : Tensor [2, 3, 5] F64 -> Tensor [2, 5, 3] F64
test_T_with_leading x = x.T

export
test_map : IO ()
test_map = do
    let x = const {shape=[_, _]} {dtype=S32} [[1, 15, 5], [-1, 7, 6]]
    assertAll "map for S32 array" $ map absEach x ==# absEach x

    let x = const {shape=[_, _]} {dtype=F64} [[1.0, 2.5, 0.0], [-0.8, -0.1, 5.0]]
    assertAll "map for F64 array" $
        map (const 1 /) x ==# const [[1.0, 0.4, inf], [-1.25, -10, 0.2]]

    sequence_ $ do
        x <- ints
        let x = const {shape=[]} {dtype=S32} x
        pure $ assertAll "map for S32 scalar" $ map (+ const 1) x ==# x + const 1

    sequence_ $ do
        x <- doubles
        let x = const {shape=[]} {dtype=F64} x
        pure $ assertAll "map for F64 scalar" $
            sufficientlyEqEach (map (+ const 1.2) x) (x + const 1.2)

export
test_map2 : IO ()
test_map2 = do
    let l = const {shape=[_, _]} {dtype=S32} [[1, 2, 3], [-1, -2, -3]]
        r = const {shape=[_, _]} {dtype=S32} [[1, 4, 2], [-2, -1, -3]]
    assertAll "map2 for S32 array" $ map2 (+) l r ==# (l + r)

    let l = const {shape=[_, _]} {dtype=F64} [[1.1, 2.2, 3.3], [-1.1, -2.2, -3.3]]
        r = const {shape=[_, _]} {dtype=F64} [[1.1, 4.4, 2.2], [-2.2, -1.1, -3.3]]
    assertAll "map2 for F64 matrix" $ sufficientlyEqEach (map2 (+) l r) (l + r)

    sequence_ $ do
        l <- doubles
        r <- doubles
        let l' = const {shape=[]} {dtype=F64} l
            r' = const {shape=[]} {dtype=F64} r
        pure $ assertAll "map2 for F64 scalars" $ sufficientlyEqEach (map2 (+) l' r') (l' + r')

    sequence_ $ do
        l <- doubles
        let l' = const {shape=[]} {dtype=F64} l
        pure $ assertAll "map2 for F64 scalars with repeated argument" $
            sufficientlyEqEach (map2 (+) l' l') (l' + l')

export
test_reduce : IO ()
test_reduce = do
    let x = const {shape=[_, _]} {dtype=F64} [[1.1, 2.2, 3.3], [-1.1, -2.2, -3.3]]
    assertAll "reduce for F64 array" $ sufficientlyEqEach (reduce @{Sum} 1 x) (const [6.6, -6.6])

    let x = const {shape=[_, _]} {dtype=F64} [[1.1, 2.2, 3.3], [-1.1, -2.2, -3.3]]
    assertAll "reduce for F64 array" $ sufficientlyEqEach (reduce @{Sum} 0 x) (const [0, 0, 0])

    let x = const {shape=[_, _]} {dtype=PRED} [[True, False, True], [True, False, False]]
    assertAll "reduce for PRED array" $ reduce @{All} 1 x ==# const [False, False]

export
test_elementwise_equality : IO ()
test_elementwise_equality = do
    let x = const {shape=[_]} {dtype=PRED} [True, True, False]
        y = const {shape=[_]} {dtype=PRED} [False, True, False]
    eq <- eval {shape=[_]} (y ==# x)
    assert "==# for boolean vector" $ eq == [False, True, True]

    let x = const {shape=[_, _]} {dtype=S32} [[1, 15, 5], [-1, 7, 6]]
        y = const {shape=[_, _]} {dtype=S32} [[2, 15, 3], [2, 7, 6]]
    eq <- eval (y ==# x)
    assert "==# for integer matrix" $ eq == [[False, True, False], [False, True, True]]

    let x = const {shape=[_, _]} {dtype=F64} [[1.1, 15.3, 5.2], [-1.6, 7.1, 6.0]]
        y = const {shape=[_, _]} {dtype=F64} [[2.2, 15.3, 3.4], [2.6, 7.1, 6.0]]
    eq <- eval (y ==# x)
    assert "==# for double matrix" $ eq == [[False, True, False], [False, True, True]]

    sequence_ [compareScalars {dtype=PRED} x y | x <- bools, y <- bools]
    sequence_ [compareScalars {dtype=S32} x y | x <- ints, y <- ints]
    sequence_ [compareScalars {dtype=F64} x y | x <- doubles, y <- doubles]

    where
        compareScalars : Primitive dtype => Prelude.Eq ty => PrimitiveRW dtype ty
                         => Primitive.Eq dtype => ty -> ty -> IO ()
        compareScalars l r = do
            actual <- eval {shape=[]} ((const {dtype} l) ==# (const {dtype} r))
            assert "==# for scalars" (actual == (l == r))

export
test_elementwise_inequality : IO ()
test_elementwise_inequality = do
    let x = const {shape=[_]} {dtype=PRED} [True, True, False]
        y = const {shape=[_]} {dtype=PRED} [False, True, False]
    assertAll "==# for boolean vector" $ (y /=# x) ==# const {shape=[_]} [True, False, False]

    let x = const {shape=[_, _]} {dtype=S32} [[1, 15, 5], [-1, 7, 6]]
        y = const {shape=[_, _]} {dtype=S32} [[2, 15, 3], [2, 7, 6]]
    assertAll "==# for integer matrix" $
        (x /=# y) ==# const [[True, False, True], [True, False, False]]

    let x = const {shape=[_, _]} {dtype=F64} [[1.1, 15.3, 5.2], [-1.6, 7.1, 6.0]]
        y = const {shape=[_, _]} {dtype=F64} [[2.2, 15.3, 3.4], [2.6, 7.1, 6.0]]
    assertAll "==# for double matrix" $
        (x /=# y) ==# const [[True, False, True], [True, False, False]]

    sequence_ [compareScalars {dtype=PRED} l r | l <- bools, r <- bools]
    sequence_ [compareScalars {dtype=S32} l r | l <- ints, r <- ints]
    sequence_ [compareScalars {dtype=F64} l r | l <- doubles, r <- doubles]

    where
        compareScalars : Primitive dtype => Primitive.Eq dtype => Prelude.Eq ty
                         => PrimitiveRW dtype ty => ty -> ty -> IO ()
        compareScalars l r =
            assertAll "/=# for scalars" $ (const {dtype} l /=# const r) ==# const {shape=[]} (l /= r)

export
test_comparison : IO ()
test_comparison = do
    let x = const {shape=[_, _]} {dtype=S32} [[1, 2, 3], [-1, -2, -3]]
        y = const {shape=[_, _]} {dtype=S32} [[1, 4, 2], [-2, -1, -3]]
    assertAll "># for S32 matrix" $ (y ># x) ==# const [[False, True, False], [False, True, False]]
    assertAll "<# for S32 matrix" $ (y <# x) ==# const [[False, False, True], [True, False, False]]
    assertAll ">=# for S32 matrix" $ (y >=# x) ==# const [[True, True, False], [False, True, True]]
    assertAll "<=# for S32 matrix" $ (y <=# x) ==# const [[True, False, True], [True, False, True]]

    let x = const {shape=[_, _]} {dtype=F64} [[1.1, 2.2, 3.3], [-1.1, -2.2, -3.3]]
        y = const {shape=[_, _]} {dtype=F64} [[1.1, 4.4, 2.2], [-2.2, -1.1, -3.3]]
    assertAll "># for F64 matrix" $
        (y ># x) ==# const [[False, True, False], [False, True, False]]
    assertAll "<# for F64 matrix" $
        (y <# x) ==# const [[False, False, True], [True, False, False]]
    assertAll ">=# for F64 matrix" $
        (y >=# x) ==# const [[True, True, False], [False, True, True]]
    assertAll "<=# for F64 matrix" $
        (y <=# x) ==# const [[True, False, True], [True, False, True]]

    sequence_ [compareScalars {dtype=S32} l r | l <- ints, r <- ints]
    sequence_ [compareScalars {dtype=F64} l r | l <- doubles, r <- doubles]

    where
        compareScalars : Primitive.Ord dtype => Prelude.Ord ty => PrimitiveRW dtype ty
                         => Primitive dtype => ty -> ty -> IO ()
        compareScalars l r = do
            let l' = const {dtype} l
                r' = const {dtype} r
            assertAll "># for scalars" $ (l' ># r') ==# const {shape=[]} {dtype=PRED} (l > r)
            assertAll "<# for scalars" $ (l' <# r') ==# const {shape=[]} {dtype=PRED} (l < r)
            assertAll ">=# for scalars" $ (l' >=# r') ==# const {shape=[]} {dtype=PRED} (l >= r)
            assertAll "<=# for scalars" $ (l' <=# r') ==# const {shape=[]} {dtype=PRED} (l <= r)

test_tensor_contraction11 : Tensor [4] F64 -> Tensor [4] F64 -> Tensor [] F64
test_tensor_contraction11 x y = x @@ y

test_tensor_contraction12 : Tensor [4] F64 -> Tensor [4, 5] F64 -> Tensor [5] F64
test_tensor_contraction12 x y = x @@ y

test_tensor_contraction21 : Tensor [3, 4] F64 -> Tensor [4] F64 -> Tensor [3] F64
test_tensor_contraction21 x y = x @@ y

test_tensor_contraction22 : Tensor [3, 4] F64 -> Tensor [4, 5] F64 -> Tensor [3, 5] F64
test_tensor_contraction22 x y = x @@ y

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
        assertAll (name ++ " for S32 array") $ f_tensor (const x) (const y) ==# expected

        sequence_ $ do
            l <- ints
            r <- ints
            pure $ assertAll (name ++ " for S32 scalar " ++ show l ++ " " ++ show r) $
                f_tensor (const l) (const r) ==# const {shape=[]} {dtype=S32} (f_native l r)

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
            sufficientlyEqEach (f_tensor (const x) (const y)) expected

        sequence_ $ do
            l <- doubles
            r <- doubles
            pure $ assertAll (name ++ " for F64 scalar " ++ show l ++ " " ++ show r) $
                sufficientlyEqEach (f_tensor (const l) (const r)) $
                    const {shape=[]} {dtype=F64} (f_native l r)

export
test_add : IO ()
test_add = do
    S32.testElementwiseBinary "(+)" (+) (+)
    F64.testElementwiseBinary "(+)" (+) (+)

export
test_Sum : IO ()
test_Sum = do
    let x = const {shape=[_, _]} {dtype=F64} [[1.1, 2.1, -2.0], [-1.3, -1.0, 1.0]]
    assertAll "Sum neutral is neutral right" $ (<+>) @{Sum} x (neutral @{Sum}) ==# x
    assertAll "Sum neutral is neutral left" $ (<+>) @{Sum} (neutral @{Sum}) x ==# x

export
test_subtract : IO ()
test_subtract = do
    let l = const [[1, 15, 5], [-1, 7, 6]]
        r = const [[11, 5, 7], [-3, -4, 0]]
    assertAll "- for S32 matrix" $
        (l - r) ==# const {shape=[_, _]} {dtype=S32} [[-10, 10, -2], [2, 11, 6]]

    let l = const [1.8, 1.3, 4.0]
        r = const [-3.3, 0.0, 0.3]
    diff <- eval {shape=[3]} {dtype=F64} (l - r)
    sequence_ (zipWith ((assert "- for F64 matrix") .: sufficientlyEq) diff [5.1, 1.3, 3.7])

    sequence_ $ do
        l <- ints
        r <- ints
        pure $ assertAll "- for S32 scalar" $ (const l - const r) ==# const {shape=[]} {dtype=S32} (l - r)

    sequence_ $ do
        l <- doubles
        r <- doubles
        pure $ do
            diff <- eval {shape=[]} {dtype=F64} (const l - const r)
            assert "- for F64 scalar" (sufficientlyEq diff (l - r))

export
test_elementwise_multiplication : IO ()
test_elementwise_multiplication = do
    S32.testElementwiseBinary "(*#)" (*) (*#)
    F64.testElementwiseBinary "(*#)" (*) (*#)

export
test_scalar_multiplication : IO ()
test_scalar_multiplication = do
    let r = const {shape=[_, _]} {dtype=S32} [[11, 5, 7], [-3, -4, 0]]
    sequence_ $ do
        l <- ints
        pure $ assertAll "* for int array" $
            (const l) * r ==# const [[11 * l, 5 * l, 7 * l], [-3 * l, -4 * l, 0]]

    let r = const {shape=[_, _]} {dtype=F64} [[-3.3], [0.0], [0.3]]
    sequence_ $ do
        l <- doubles
        pure $ assertAll "* for double array" $
            sufficientlyEqEach ((const l) * r) (const [[-3.3 * l], [0.0 * l], [0.3 * l]])

    sequence_ $ do
        l <- ints
        r <- ints
        pure $ assertAll "* for int scalar" $
            (const l * const r) ==# const {shape=[]} {dtype=S32} (l * r)

    sequence_ $ do
        l <- doubles
        r <- doubles
        pure $ assertAll "* for double array" $
            sufficientlyEqEach (const l * const r) (const {shape=[]} (l * r))

export
test_Prod : IO ()
test_Prod = do
    let x = const {shape=[_, _]} {dtype=F64} [[1.1, 2.1, -2.0], [-1.3, -1.0, 1.0]]
    assertAll "Prod neutral is neutral right" $ (<+>) @{Prod} x (neutral @{Prod}) ==# x
    assertAll "Prod neutral is neutral left" $ (<+>) @{Prod} (neutral @{Prod}) x ==# x

assertBooleanOpArray : String -> (Tensor [2, 2] PRED -> Tensor [2, 2] PRED -> Tensor [2, 2] PRED)
                       -> Array [2, 2] Bool -> IO ()
assertBooleanOpArray name op expected = do
    let l = const [[True, True], [False, False]]
        r = const [[True, False], [True, False]]
    assertAll name $ op l r ==# const expected

assertBooleanOpScalar : String -> (Tensor [] PRED -> Tensor [] PRED -> Tensor [] PRED)
                        -> (Bool -> Lazy Bool -> Bool) -> IO ()
assertBooleanOpScalar name tensor_op bool_op =
    sequence_ $ do
        l <- bools
        r <- bools
        pure $ assertAll name $ tensor_op (const l) (const r) ==# const (bool_op l r)

export
test_elementwise_and : IO ()
test_elementwise_and = do
    assertBooleanOpArray "&&# for array" (&&#) [[True, False], [False, False]]
    assertBooleanOpScalar "&&# for scalar" (&&#) (&&)

export
test_All : IO ()
test_All = do
    let x = const {shape=[_, _]} {dtype=PRED} [[True, True], [False, False]]
    assertAll "All neutral is neutral right" $ (<+>) @{All} x (neutral @{All}) ==# x
    assertAll "All neutral is neutral left" $ (<+>) @{All} (neutral @{All}) x ==# x

export
test_elementwise_or : IO ()
test_elementwise_or = do
    assertBooleanOpArray "||# for array" (||#) [[True, True], [True, False]]
    assertBooleanOpScalar "||# for scalar" (||#) (||)

export
test_Any : IO ()
test_Any = do
    let x = const {shape=[_, _]} {dtype=PRED} [[True, True], [False, False]]
    assertAll "Any neutral is neutral right" $ (<+>) @{Any} x (neutral @{Any}) ==# x
    assertAll "Any neutral is neutral left" $ (<+>) @{Any} (neutral @{Any}) x ==# x

export
test_elementwise_notEach : IO ()
test_elementwise_notEach = do
    assertAll "notEach for array" $
        notEach (const [True, False]) ==# const {shape=[_]} [False, True]
    sequence_ [assertAll "notEach for scalar" $
               notEach (const x) ==# const {shape=[]} (not x) | x <- bools]

export
test_elementwise_division : IO ()
test_elementwise_division = do
    F64.testElementwiseBinary "(/#)" (/) (/#)

export
test_scalar_division : IO ()
test_scalar_division = do
    let l = const {shape=[_, _]} {dtype=F64} [[-3.3], [0.0], [0.3]]
    sequence_ $ do
        r <- doubles
        pure $ assertAll "/ for array" $
            sufficientlyEqEach (l / const r) (const [[-3.3 / r], [0.0 / r], [0.3 / r]])

    sequence_ $ do
        l <- doubles
        r <- doubles
        pure $ assertAll "/ for scalar" $
            sufficientlyEqEach (const l / const r) (const {shape=[]} (l / r))

export
test_absEach : IO ()
test_absEach = do
    let x = const {shape=[_]} {dtype=S32} [1, 0, -5]
    assertAll "absEach for int array" $ absEach x ==# const [1, 0, 5]

    let x = const {shape=[3]} {dtype=F64} [1.8, -1.3, 0.0]
    actual <- eval (absEach x)
    sequence_ (zipWith ((assert "absEach for double array") .: sufficientlyEq)
        actual [1.8, 1.3, 0.0])

    sequence_ $ do
        x <- ints
        pure $ assertAll "absEach for int scalar" $
            absEach (const {shape=[]} {dtype=S32} x) ==# const (abs x)

    traverse_ (\x => do
            actual <- eval (absEach $ const {shape=[]} {dtype=F64} x)
            assert "absEach for double scalar" (sufficientlyEq actual (abs x))
        ) doubles

namespace S32
    export
    testElementwiseUnary : String -> (Int -> Int)
        -> (forall shape . Tensor shape S32 -> Tensor shape S32) -> IO ()
    testElementwiseUnary name f_native f_tensor = do
        let x = [[1, 15, -5], [-1, 7, 0]]
            expected = const {shape=[_, _]} {dtype=S32} (map (map f_native) x)
        assertAll (name ++ " for S32 array") $ f_tensor (const x) ==# expected

        sequence_
            [assertAll (name ++ " for S32 scalar " ++ show x) $ 
                (f_tensor $ const x) ==# (const {shape=[]} (f_native x)) | x <- ints]

namespace F64
    export
    testElementwiseUnary : String -> (Double -> Double)
        -> (forall shape . Tensor shape F64 -> Tensor shape F64) -> IO ()
    testElementwiseUnary name f_native f_tensor = do
        let x = [[1.3, 1.5, -5.2], [-1.1, 7.0, 0.0]]
            expected = const {shape=[_, _]} {dtype=F64} (map (map f_native) x)
        assertAll (name ++ " for F64 array") $
            sufficientlyEqEach (f_tensor (const x)) expected

        sequence_
            [assertAll (name ++ " for F64 scalar " ++ show x) $ sufficientlyEqEach
                (f_tensor $ const x) (const {shape=[]} (f_native x)) | x <- doubles]

export
test_negate : IO ()
test_negate = do
    S32.testElementwiseUnary "negate" negate negate
    F64.testElementwiseUnary "negate" negate negate

tanh' : Double -> Double
tanh' x =
    if x == -inf then -1.0
    else if x == inf then 1.0
    else tanh x

export
testElementwiseUnaryDoubleCases : IO ()
testElementwiseUnaryDoubleCases = do
    F64.testElementwiseUnary "expEach" exp expEach
    F64.testElementwiseUnary "floorEach" floor floorEach
    F64.testElementwiseUnary "ceilEach" ceiling ceilEach
    F64.testElementwiseUnary "logEach" log logEach
    F64.testElementwiseUnary "logisticEach" (\x => 1 / (1 + exp (-x))) logisticEach
    F64.testElementwiseUnary "sinEach" sin sinEach
    F64.testElementwiseUnary "cosEach" cos cosEach
    F64.testElementwiseUnary "tanhEach" tanh' tanhEach
    F64.testElementwiseUnary "sqrtEach" sqrt sqrtEach

min' : Double -> Double -> Double
min' x y = if (x /= x) then x else if (y /= y) then y else min x y

export
test_minEach : IO ()
test_minEach = do
    S32.testElementwiseBinary "minEach" min minEach
    F64.testElementwiseBinary "minEach" min' minEach

export
test_Min : IO ()
test_Min = do
    let x = const {shape=[_, _]} {dtype=F64} [[1.1, 2.1, -2.0], [-1.3, -1.0, 1.0]]
    assertAll "Min neutral is neutral right" $ (<+>) @{Min} x (neutral @{Min}) ==# x
    assertAll "Min neutral is neutral left" $ (<+>) @{Min} (neutral @{Min}) x ==# x

max' : Double -> Double -> Double
max' x y = if (x /= x) then x else if (y /= y) then y else max x y

export
test_maxEach : IO ()
test_maxEach = do
    S32.testElementwiseBinary "maxEach" max maxEach
    F64.testElementwiseBinary "maxEach" max' maxEach

export
test_Max : IO ()
test_Max = do
    let x = const {shape=[_, _]} {dtype=F64} [[1.1, 2.1, -2.0], [-1.3, -1.0, 1.0]]
    assertAll "Max neutral is neutral right" $ (<+>) @{Max} x (neutral @{Max}) ==# x
    assertAll "Max neutral is neutral left" $ (<+>) @{Max} (neutral @{Max}) x ==# x

test_det : Tensor [3, 3] F64 -> Tensor [] F64
test_det x = det x

test_det_with_leading : Tensor [2, 3, 3] F64 -> Tensor [2] F64
test_det_with_leading x = det x
