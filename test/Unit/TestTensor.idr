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

test_T : Tensor [2, 3] F64 -> Tensor [3, 2] F64
test_T x = x.T

test_T_with_leading : Tensor [2, 3, 5] F64 -> Tensor [2, 5, 3] F64
test_T_with_leading x = x.T

export
test_map : IO ()
test_map = do
    let x = const {shape=[_, _]} {dtype=S32} [[1, 15, 5], [-1, 7, 6]]
    assertAll "map for Int array" $ map absEach x ==# absEach x

    let x = const {shape=[_, _]} {dtype=F64} [[1.0, 2.5, 0.0], [-0.8, -0.1, 5.0]]
    assertAll "map for Double array" $
        map (const 1 /) x ==# const [[1.0, 0.4, 1 / 0.0], [-1.25, -10, 0.2]]

    sequence_ $ do
        x <- ints
        let x = const {shape=[]} {dtype=S32} x
        pure $ assertAll "map for int scalar" $ map (+ const 1) x ==# x + const 1

    sequence_ $ do
        x <- doubles
        let x = const {shape=[]} {dtype=F64} x
        pure $ assertAll "map for double scalar" $
            sufficientlyEqEach (map (+ const 1.2) x) (x + const 1.2)

export
test_map2 : IO ()
test_map2 = do
    let l = const {shape=[_, _]} {dtype=S32} [[1, 2, 3], [-1, -2, -3]]
        r = const {shape=[_, _]} {dtype=S32} [[1, 4, 2], [-2, -1, -3]]
    assertAll "map2 for Int array" $ map2 (+) l r ==# (l + r)

    let l = const {shape=[_, _]} {dtype=F64} [[1.1, 2.2, 3.3], [-1.1, -2.2, -3.3]]
        r = const {shape=[_, _]} {dtype=F64} [[1.1, 4.4, 2.2], [-2.2, -1.1, -3.3]]
    assertAll "map2 for Double matrix" $ sufficientlyEqEach (map2 (+) l r) (l + r)

    sequence_ $ do
        l <- doubles
        r <- doubles
        let l' = const {shape=[]} {dtype=F64} l
            r' = const {shape=[]} {dtype=F64} r
        pure $ assertAll "map2 for Double scalars" $ sufficientlyEqEach (map2 (+) l' r') (l' + r')

    sequence_ $ do
        l <- doubles
        let l' = const {shape=[]} {dtype=F64} l
        pure $ assertAll "map2 for Double scalars with repeated argument" $
            sufficientlyEqEach (map2 (+) l' l') (l' + l')

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

export
test_add : IO ()
test_add = do
    let x = const [[1, 15, 5], [-1, 7, 6]]
        y = const [[11, 5, 7], [-3, -4, 0]]
    assertAll "+ for S32 matrix" $
        x + y ==# const {shape=[_, _]} {dtype=S32} [[12, 20, 12], [-4, 3, 6]]

    let x = const [[1.8], [1.3], [4.0]]
        y = const [[-3.3], [0.0], [0.3]]
    assertAll "+ for F64 matrix" $ sufficientlyEqEach (x + y) $
        const {shape=[_, _]} {dtype=F64} [[-1.5], [1.3], [4.3]]

    sequence_ $ do
        l <- ints
        r <- ints
        pure $ assertAll "+ for scalar S32" $ (const l + const r) ==# const {shape=[]} {dtype=S32} (l + r)

    sequence_ $ do
        l <- doubles
        r <- doubles
        pure $ assertAll "+ for scalar F64" $
            sufficientlyEqEach (const l + const r) (const {shape=[]} (l + r))

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
    let x = const [[1, 15, 5], [-1, 7, 6]]
        y = const [[11, 5, 7], [-3, -4, 0]]
    assertAll "*# for int array" $
        x *# y ==# const {shape=[_, _]} {dtype=S32} [[11, 75, 35], [3, -28, 0]]

    let x = const [[1.8], [1.3], [4.0]]
        y = const [[-3.3], [0.0], [0.3]]
    assertAll "*# for double array" $ sufficientlyEqEach (x *# y) $
        const {shape=[_, _]} {dtype=F64} [[-1.8 * 3.3], [0.0], [1.2]]

    sequence_ $ do
        l <- ints
        r <- ints
        pure $ assertAll "*# for int scalar" $
            (const l *# const r) ==# const {shape=[]} {dtype=S32} (l * r)

    sequence_ $ do
        l <- doubles
        r <- doubles
        pure $ assertAll "*# for double scalar" $
            sufficientlyEqEach (const l *# const r) (const {shape=[]} (l * r))

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
test_elementwise_or : IO ()
test_elementwise_or = do
    assertBooleanOpArray "||# for array" (||#) [[True, True], [True, False]]
    assertBooleanOpScalar "||# for scalar" (||#) (||)

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
    let x = const [[3, 4, -5], [0, 0.3, 0]]
        y = const [[1, -2.3, 0.2], [0.1, 0, 0]]
        expected = const {shape=[_, _]} {dtype=F64} [[3, -4 / 2.3, -25], [0, 0.3 / 0, 0 / 0]]
    assertAll "/# for array" $ sufficientlyEqEach (x /# y) expected

    sequence_ $ do
        l <- doubles
        r <- doubles
        pure $ assertAll "/# for scalar" $
            sufficientlyEqEach (const l /# const r) (const {shape=[]} (l / r))

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

export
test_minEach : IO ()
test_minEach = do
    let x = const {shape=[_, _]} {dtype=S32} [[1, 2, -2], [-1, -1, 1]]
        y = const {shape=[_, _]} {dtype=S32} [[2, 1, -1], [-2,  0, 0]]
    assertAll "minEach for S32 array" $ minEach x y ==# const [[1, 1, -2], [-2, -1, 0]]

    let x = const {shape=[_, _]} {dtype=F64} [[1.1, 2.1, -2.0], [-1.3, -1.0, 1.0]]
        y = const {shape=[_, _]} {dtype=F64} [[2.0, 1.2, -1.1], [-2.3,  0.0, 0.0]]
    assertAll "minEach for F64 array" $ minEach x y ==# const [[1.1, 1.2, -2.0], [-2.3, -1.0, 0.0]]

    sequence_ $ do
        l <- ints
        r <- ints
        pure $ assertAll ("minEach for S32 scalars " ++ show (l, r)) $
            minEach (const l) (const r) ==# const {shape=[]} {dtype=S32} (min l r)

    sequence_ $ do
        l <- doubles
        r <- doubles
        pure $ assertAll ("minEach for F64 scalars " ++ show (l, r)) $
            sufficientlyEqEach (minEach (const l) (const r)) (const {shape=[]} (minDouble l r))

        where
        minDouble : Double -> Double -> Double
        minDouble x y = if (x /= x) then x else if (y /= y) then y else min x y

export
test_maxEach : IO ()
test_maxEach = do
    let x = const {shape=[_, _]} {dtype=S32} [[1, 2, -2], [-1, -1, 1]]
        y = const {shape=[_, _]} {dtype=S32} [[2, 1, -1], [-2,  0, 0]]
    assertAll "maxEach for S32 array" $ maxEach x y ==# const [[2, 2, -1], [-1, 0, 1]]

    let x = const {shape=[_, _]} {dtype=F64} [[1.1, 2.1, -2.0], [-1.3, -1.0, 1.0]]
        y = const {shape=[_, _]} {dtype=F64} [[2.0, 1.2, -1.1], [-2.3,  0.0, 0.0]]
    assertAll "maxEach for F64 array" $ maxEach x y ==# const [[2.0, 2.1, -1.1], [-1.3, 0.0, 1.0]]

    sequence_ $ do
        l <- ints
        r <- ints
        pure $ assertAll ("maxEach for S32 scalars " ++ show (l, r)) $
            maxEach (const l) (const r) ==# const {shape=[]} {dtype=S32} (max l r)

    sequence_ $ do
        l <- doubles
        r <- doubles
        pure $ assertAll ("maxEach for F64 scalars " ++ show (l, r)) $
            sufficientlyEqEach (maxEach (const l) (const r)) (const {shape=[]} (maxDouble l r))

        where
        maxDouble : Double -> Double -> Double
        maxDouble x y = if (x /= x) then x else if (y /= y) then y else max x y

export
test_negate : IO ()
test_negate = do
    let x = const [[1, 15, -5], [-1, 7, 0]]
    assertAll "negate for int array" $
        (-x) ==# const {shape=[_, _]} {dtype=S32} [[-1, -15, 5], [1, -7, 0]]

    let x = const [[1.3, 1.5, -5.2], [-1.1, 7.0, 0.0]]
        expected = const {shape=[_, _]} {dtype=F64} [[-1.3, -1.5, 5.2], [1.1, -7.0, 0.0]]
    assertAll "negate for double array" $ sufficientlyEqEach (-x) expected

    sequence_ [assertAll "negate for int scalar" $
               (- const {dtype=S32} x) ==# const {shape=[]} (-x) | x <- ints]
    sequence_ [assertAll "negate for double scalar" $
               sufficientlyEqEach (- const x) (const {shape=[]} (-x)) | x <- doubles]

test_det : Tensor [3, 3] F64 -> Tensor [] F64
test_det x = det x

test_det_with_leading : Tensor [2, 3, 3] F64 -> Tensor [2] F64
test_det_with_leading x = det x
