{--
Copyright 2022 Joel Berkeley

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
module Compiler.Autodiff

import Compiler.Expr
import Primitive
import Literal

call : Expr -> Expr -> Expr
call y@(FromLiteral _) _ = y
call (Parameter _ _ _) x = x
call y@MinFiniteValue _ = y
call y@MaxFiniteValue _ = y
call (ConvertElementType {dtype} y) x = ConvertElementType {dtype} (call y x)
call (Reshape from to y) x = Reshape from to (call y x)
call (Slice start stop stride y) x = Slice start stop stride (call y x)
call (Concat k y z) x = Concat k (call y x) (call z x)
call (Diag y) x = Diag (call y x)
call (Triangle lower y) x = Triangle lower (call y x)
call (Transpose y) x = Transpose (call y x)
call y@(Identity k) _ = y
call (Broadcast {dtype} from to y) x = Broadcast {dtype} from to (call y x)
call (Map y xs ys) x = ?call__14
call (Reduce y z k w) x = ?call__15
call (Sort y k z xs) x = ?call__16
call (Reverse axes y) x = Reverse axes (call y x)
call (Eq y z) x = Eq (call y x) (call z x)
call (Ne y z) x = Ne (call y x) (call z x)
call (Add y z) x = Add (call y x) (call z x)
call (Sub y z) x = Sub (call y x) (call z x)
call (Mul y z) x = Mul (call y x) (call z x)
call (Div y z) x = Div (call y x) (call z x)
call (Pow y z) x = Pow (call y x) (call z x)
call (Lt y z) x = Lt (call y x) (call z x)
call (Gt y z) x = Gt (call y x) (call z x)
call (Le y z) x = Le (call y x) (call z x)
call (Ge y z) x = Ge (call y x) (call z x)
call (And y z) x = And (call y x) (call z x)
call (Or y z) x = Or (call y x) (call z x)
call (Min y z) x = Min (call y x) (call z x)
call (Max y z) x = Max (call y x) (call z x)
call (Not y) x = Not (call y x)
call (Neg y) x = Neg (call y x)
call (Reciprocal y) x = Reciprocal (call y x)
call (Abs y) x = Abs (call y x)
call (Ceil y) x = Ceil (call y x)
call (Floor y) x = Floor (call y x)
call (Log y) x = Log (call y x)
call (Exp y) x = Exp (call y x)
call (Logistic y) x = Logistic (call y x)
call (Erf y) x = Erf (call y x)
call (Square y) x = Square (call y x)
call (Sqrt y) x = Sqrt (call y x)
call (Sin y) x = Sin (call y x)
call (Cos y) x = Cos (call y x)
call (Tan y) x = Tan (call y x)
call (Asin y) x = Asin (call y x)
call (Acos y) x = Acos (call y x)
call (Atan y) x = Atan (call y x)
call (Sinh y) x = Sinh (call y x)
call (Cosh y) x = Cosh (call y x)
call (Tanh y) x = Tanh (call y x)
call (Asinh y) x = Asinh (call y x)
call (Acosh y) x = Acosh (call y x)
call (Atanh y) x = Atanh (call y x)
call (Select p t f) x = Select (call p x) (call t x) (call f x)
call (Cond y z w v s) x = ?call__58
call (Dot y z) x = Dot (call y x) (call z x)
call (Cholesky y) x = Cholesky (call y x)
call (TriangularSolve y z lower) x = TriangularSolve (call y x) (call z x) lower
call (UniformFloatingPointDistributionValue y z w v xs) x = ?call__62
call (UniformFloatingPointDistributionState y z w v xs) x = ?call__63
call (NormalFloatingPointDistributionValue y z xs) x = ?call__64
call (NormalFloatingPointDistributionState y z xs) x = ?call__65

export
grad : Expr -> Expr -> Expr
grad (FromLiteral _) x = FromLiteral {dtype=F64} 0.0
grad (Parameter _ _ _) x = FromLiteral {dtype=F64} 1.0
grad MinFiniteValue x = FromLiteral {dtype=F64} 0.0
grad MaxFiniteValue x = FromLiteral {dtype=F64} 0.0
grad (ConvertElementType y) x = ?res_5
grad (Reshape xs ys y) x = ?res_6
grad (Slice xs ys zs y) x = ?res_7
grad (Concat k y z) x = ?res_8
grad (Diag y) x = ?res_9
grad (Triangle lower y) x = ?res_10
grad (Transpose y) x = ?res_11
grad (Identity k) x = ?res_12
grad (Broadcast xs ys y) x = ?res_13
grad (Map y xs ys) x = ?res_14
grad (Reduce y z k w) x = ?res_15
grad (Sort y k z xs) x = ?res_16
grad (Reverse xs y) x = ?res_17
grad (Eq y z) x = ?res_18
grad (Ne y z) x = ?res_19
grad (Add y z) x = grad y x `Add` grad z x
grad (Sub y z) x = grad y x `Sub` grad z x
grad (Mul y z) x = (grad y x `Mul` call z x) `Add` (call y x `Mul` grad z x)
grad (Div y z) x =
  (grad y x `Div` call z x) `Sub` (call y x `Mul` (grad z x `Mul` (Reciprocal $ Square $ call z x)))
grad (Pow y z) x = ?res_24
grad (Lt y z) x = ?res_25
grad (Gt y z) x = ?res_26
grad (Le y z) x = ?res_27
grad (Ge y z) x = ?res_28
grad (And y z) x = ?res_29
grad (Or y z) x = ?res_30
grad (Min y z) x = ?res_31
grad (Max y z) x = ?res_32
grad (Not y) x = ?res_33
grad (Neg y) x = ?res_34
grad (Reciprocal y) x = ?res_35
grad (Abs y) x = ?res_36
grad (Ceil y) x = ?res_37
grad (Floor y) x = ?res_38
grad (Log y) x = grad y x `Div` call y x
grad (Exp y) x = grad y x `Mul` (Exp (call y x))
grad (Logistic y) x = ?res_41
grad (Erf y) x = ?res_42
grad (Square y) x = (FromLiteral {dtype=F64} 2.0 `Mul` call y x) `Mul` grad y x
grad (Sqrt y) x = grad y x `Div` (FromLiteral {dtype=F64} 2.0 `Mul` (Sqrt (call y x)))
grad (Sin y) x = grad y x `Mul` Cos (call y x)
grad (Cos y) x = grad y x `Mul` (Neg $ Sin (call y x))
grad (Tan y) x = grad y x `Div` (Square (Cos (call y x)))
grad (Asin y) x = grad y x `Div` (Sqrt (FromLiteral {dtype=F64} 1.0 `Sub` Square (call y x)))
grad (Acos y) x = grad y x `Div` (Neg $ Sqrt (FromLiteral {dtype=F64} 1.0 `Sub` Square (call y x)))
grad (Atan y) x = grad y x `Div` (Neg $ (FromLiteral {dtype=F64} 1.0 `Add` Square (call y x)))
grad (Sinh y) x = grad y x `Mul` Cosh (call y x)
grad (Cosh y) x = grad y x `Mul` Sinh (call y x)
grad (Tanh y) x = grad y x `Div` (Square (Cosh (call y x)))
grad (Asinh y) x = grad y x `Div` (Sqrt (FromLiteral {dtype=F64} 1.0 `Add` Square (call y x)))
grad (Acosh y) x = grad y x `Div` (Sqrt (FromLiteral {dtype=F64} 1.0 `Sub` Square (call y x)))
grad (Atanh y) x = grad y x `Div` (Sqrt (Square (call y x) `Sub` FromLiteral {dtype=F64} 1.0))
grad (Select y z w) x = ?res_57
grad (Cond y z w v s) x = ?res_58
grad (Dot y z) x = ?res_59
grad (Cholesky y) x = ?res_60
grad (TriangularSolve y z w) x = ?res_61
grad (UniformFloatingPointDistributionValue y z w v xs) x = ?res_62
grad (UniformFloatingPointDistributionState y z w v xs) x = ?res_63
grad (NormalFloatingPointDistributionValue y z xs) x = ?res_64
grad (NormalFloatingPointDistributionState y z xs) x = ?res_65
