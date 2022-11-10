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
module Compiler.Graph

import Decidable.Equality
import Syntax.PreorderReasoning

import Compiler.LiteralRW
import Compiler.Xla.TensorFlow.Compiler.Xla.XlaData

import Data.Fin
import Literal
import Primitive
import Types
import Util

public export
data FullShape : Type where
  MkFullShape : Shape -> Primitive dtype => FullShape

export
Prelude.Eq FullShape where
  (MkFullShape shape {dtype}) == (MkFullShape shape' {dtype=dtype'}) =
    (shape, typeString {dtype}) == (shape', typeString {dtype=dtype'})

public export
record Graph

public export
data Ref = P Nat Nat | N Nat Nat

export
Show Ref where
  show (P k j) = "P \{show k} \{show j}" 
  show (N k j) = "N \{show k} \{show j}"

Prelude.Eq Ref where
  (P s i) == (P s' i') = (s, i) == (s', i')
  (N s i) == (N s' i') = (s, i) == (s', i')
  _ == _ = False

-- hacky for reindex
(+) : Ref -> Nat -> Ref
(+) (P s p) k = P s (p + k)
(+) (N s i) k = N s (i + k)

public export
data Node : Type where
  FromLiteral : PrimitiveRW dtype ty => {shape : _} -> Literal shape ty -> Node
  Arg : Nat -> Nat -> Node
  Tuple : List Ref -> Node
  GetTupleElement : Nat -> Ref -> Node
  MinValue : Primitive dtype => Node
  MaxValue : Primitive dtype => Node
  MinFiniteValue : Primitive dtype => Node
  MaxFiniteValue : Primitive dtype => Node
  ConvertElementType : Primitive dtype => Ref -> Node
  Reshape : Shape -> Shape -> Ref -> Node
  Slice : List Nat -> List Nat -> List Nat -> Ref -> Node
  DynamicSlice : List Ref -> List Nat -> Ref -> Node
  Concat : Nat -> Ref -> Ref -> Node
  Diag : Ref -> Node
  Triangle : (lower : Bool) -> Ref -> Node
  Transpose : List Nat -> Ref -> Node
  Identity : Primitive dtype => Nat -> Node
  Broadcast : Primitive dtype => Shape -> Shape -> Ref -> Node
  -- Map : Graph -> List Ref -> Shape -> Node
  Reduce : Graph -> Ref -> List Nat -> Ref -> Node
  Sort : Graph -> Nat -> Bool -> List Ref -> Node
  Reverse : List Nat -> Ref -> Node
  Eq : Ref -> Ref -> Node
  Ne : Ref -> Ref -> Node
  Add : Ref -> Ref -> Node
  Sub : Ref -> Ref -> Node
  Mul : Ref -> Ref -> Node
  Div : Ref -> Ref -> Node
  Pow : Ref -> Ref -> Node
  Lt : Ref -> Ref -> Node
  Gt : Ref -> Ref -> Node
  Le : Ref -> Ref -> Node
  Ge : Ref -> Ref -> Node
  And : Ref -> Ref -> Node
  Or : Ref -> Ref -> Node
  Min : Ref -> Ref -> Node
  Max : Ref -> Ref -> Node
  Not : Ref -> Node
  Neg : Ref -> Node
  Reciprocal : Ref -> Node
  Abs : Ref -> Node
  Ceil : Ref -> Node
  Floor : Ref -> Node
  Log : Ref -> Node
  Exp : Ref -> Node
  Logistic : Ref -> Node
  Erf : Ref -> Node
  Square : Ref -> Node
  Sqrt : Ref -> Node
  Sin : Ref -> Node
  Cos : Ref -> Node
  Tan : Ref -> Node
  Asin : Ref -> Node
  Acos : Ref -> Node
  Atan : Ref -> Node
  Sinh : Ref -> Node
  Cosh : Ref -> Node
  Tanh : Ref -> Node
  Asinh : Ref -> Node
  Acosh : Ref -> Node
  Atanh : Ref -> Node
  Argmin : Primitive out => Nat -> Ref -> Node
  Argmax : Primitive out => Nat -> Ref -> Node
  Select : Ref -> Ref -> Ref -> Node
  Cond : Ref -> Graph -> Ref -> Graph -> Ref -> Node
  Dot : Ref -> Ref -> Node
  Cholesky : Ref -> Node
  TriangularSolve : Ref -> Ref -> Bool -> Node
  UniformFloatingPoint : Ref -> Ref -> Ref -> Ref -> Shape -> Node
  NormalFloatingPoint : Ref -> Ref -> Shape -> Node

public export
record Graph where
  constructor MkGraph
--  parents : List Graph
  params : List FullShape
  nodes : List Node

export
Prelude.Eq Node where
  (FromLiteral {ty} {dtype} lit {shape}) == (FromLiteral {dtype=dtype'} lit' {shape=shape'}) =
    case decEq shape shape' of
      Yes eq =>
        (xlaIdentifier {dtype} == xlaIdentifier {dtype=dtype'})
        --
        --
        -- INVALID BELIEVE ME
        --
        --
        && lit == believe_me lit'
      No _ => False
  (Arg scope position) == (Arg scope' position') = (scope, position) == (scope', position')
  (Tuple xs) == (Tuple xs') = xs == xs'
  (GetTupleElement idx tuple) == (GetTupleElement idx' tuple') = idx == idx' && tuple == tuple'
  (MinValue {dtype}) == (MinValue {dtype=dtype'}) =
    typeString {dtype} == typeString {dtype=dtype'}
  (MaxValue {dtype}) == (MaxValue {dtype=dtype'}) =
    typeString {dtype} == typeString {dtype=dtype'}
  (MinFiniteValue {dtype}) == (MinFiniteValue {dtype=dtype'}) =
    typeString {dtype} == typeString {dtype=dtype'}
  (MaxFiniteValue {dtype}) == (MaxFiniteValue {dtype=dtype'}) =
    typeString {dtype} == typeString {dtype=dtype'}
  (ConvertElementType {dtype} operand) == (ConvertElementType {dtype=dtype'} operand') =
    typeString {dtype} == typeString {dtype=dtype'} && operand == operand'
  (Reshape from to x) == (Reshape from' to' x') = (from, to) == (from', to') && x == x'
  (Slice starts stops strides x) == (Slice starts' stops' strides' x') =
    (starts, stops, strides) == (starts', stops', strides') && x == x'
  (DynamicSlice starts sizes x) == (DynamicSlice starts' sizes' x') =
    starts == starts' && sizes == sizes' && x == x'
  (Concat axis x y) == (Concat axis' x' y') = axis == axis' && x == x' && y == y'
  (Diag x) == (Diag x') = x == x'
  (Triangle lower x) == (Triangle lower' x') = lower == lower' && x == x'
  (Transpose ordering x) == (Transpose ordering' x') = ordering == ordering' && x == x'
  (Identity {dtype} n) == (Identity {dtype=dtype'} n') =
    (typeString {dtype}, n) == (typeString {dtype=dtype'}, n')
  (Broadcast from to x) == (Broadcast from' to' x') = (from, to) == (from', to') && x == x'
  -- (Map {a} (MkFn params f) xs dims) == (Map {a=a'} (MkFn params' f') xs' dims') =
  --   case decEq a a' of
  --     Yes eq =>
  --       params == (rewrite eq in params')
  --       && f == f'
  --       && xs == (rewrite eq in xs')
  --       && dims == dims'
  --     No _ => False
  (Reduce (MkGraph _ _) neutral axes x) == (Reduce (MkGraph _ _) neutral' axes' x') = ?reduce_eq
  (Sort (MkGraph _ _) dimension isStable operands) ==
    (Sort (MkGraph _ _) dimension' isStable' operands') = ?sort_eq
  (Reverse axes expr) == (Reverse axes' expr') = axes == axes' && expr == expr'
  (Eq l r) == (Eq l' r') = l == l' && r == r'
  (Ne l r) == (Ne l' r') = l == l' && r == r'
  (Add l r) == (Add l' r') = l == l' && r == r'
  (Sub l r) == (Sub l' r') = l == l' && r == r'
  (Mul l r) == (Mul l' r') = l == l' && r == r'
  (Div l r) == (Div l' r') = l == l' && r == r'
  (Pow l r) == (Pow l' r') = l == l' && r == r'
  (Lt l r) == (Lt l' r') = l == l' && r == r'
  (Gt l r) == (Gt l' r') = l == l' && r == r'
  (Le l r) == (Le l' r') = l == l' && r == r'
  (Ge l r) == (Ge l' r') = l == l' && r == r'
  (And l r) == (And l' r') = l == l' && r == r'
  (Or l r) == (Or l' r') = l == l' && r == r'
  (Min l r) == (Min l' r') = l == l' && r == r'
  (Max l r) == (Max l' r') = l == l' && r == r'
  (Not expr) == (Not expr') = expr == expr'
  (Neg expr) == (Neg expr') = expr == expr'
  (Reciprocal expr) == (Reciprocal expr') = expr == expr'
  (Abs expr) == (Abs expr') = expr == expr'
  (Ceil expr) == (Ceil expr') = expr == expr'
  (Floor expr) == (Floor expr') = expr == expr'
  (Log expr) == (Log expr') = expr == expr'
  (Exp expr) == (Exp expr') = expr == expr'
  (Logistic expr) == (Logistic expr') = expr == expr'
  (Erf expr) == (Erf expr') = expr == expr'
  (Square expr) == (Square expr') = expr == expr'
  (Sqrt expr) == (Sqrt expr') = expr == expr'
  (Sin expr) == (Sin expr') = expr == expr'
  (Cos expr) == (Cos expr') = expr == expr'
  (Tan expr) == (Tan expr') = expr == expr'
  (Asin expr) == (Asin expr') = expr == expr'
  (Acos expr) == (Acos expr') = expr == expr'
  (Atan expr) == (Atan expr') = expr == expr'
  (Sinh expr) == (Sinh expr') = expr == expr'
  (Cosh expr) == (Cosh expr') = expr == expr'
  (Tanh expr) == (Tanh expr') = expr == expr'
  (Asinh expr) == (Asinh expr') = expr == expr'
  (Acosh expr) == (Acosh expr') = expr == expr'
  (Atanh expr) == (Atanh expr') = expr == expr'
  (Argmin {out} axis expr) == (Argmin {out=out'} axis' expr') =
    (typeString {dtype=out}, axis) == (typeString {dtype=out'}, axis) && expr == expr'
  (Argmax {out} axis expr) == (Argmax {out=out'} axis' expr') =
    (typeString {dtype=out}, axis) == (typeString {dtype=out'}, axis) && expr == expr'
  (Select pred f t) == (Select pred' f' t') = pred == pred' && f == f' && t == t'
  (Cond pred (MkGraph _ _) true (MkGraph _ _) false) ==
    (Cond pred' (MkGraph _ _) true' (MkGraph _ _) false') = ?cond_eq
  (Dot x y) == (Dot x' y') = x == x' && y == y'
  (Cholesky x) == (Cholesky x') = x == x'
  (TriangularSolve x y lower) == (TriangularSolve x' y' lower') =
    x == x' && y == y' && lower == lower'
  (UniformFloatingPoint key initialState minval maxval shape) ==
    (UniformFloatingPoint key' initialState' minval' maxval' shape') =
      key == key' && initialState == initialState' && minval == minval' && maxval == maxval'
  (NormalFloatingPoint key initialState shape) == (NormalFloatingPoint key' initialState' shape') =
      key == key' && initialState == initialState'
  _ == _ = False

public export
data CompilerError =
  IndexError String

export
Show CompilerError where
  show (IndexError msg) = "IndexError \{msg}"

export
index : Nat -> List a -> Either CompilerError a
index k [] = Left (IndexError "Cannot index into empty list with index \{show k}")
index 0 (x :: _) = Right x
index (S k) (_ :: xs) = index k xs

