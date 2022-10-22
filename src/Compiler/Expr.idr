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
module Compiler.Expr

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
data Fn : Nat -> Type -> Type where
  MkFn : {arity : _} -> Vect arity FullShape -> a -> Fn arity a

public export
data Expr : Nat -> Type where
  FromLiteral : PrimitiveRW dtype ty => {shape : _} -> Literal shape ty -> Expr n
  Parameter : FullShape -> Nat -> Expr n
  Tuple : List (Fin n) -> Expr n
  GetTupleElement : Nat -> Fin n -> Expr n
  MinValue : Primitive dtype => Expr n
  MaxValue : Primitive dtype => Expr n
  MinFiniteValue : Primitive dtype => Expr n
  MaxFiniteValue : Primitive dtype => Expr n
  ConvertElementType : Primitive dtype => Fin n -> Expr n
  Reshape : Shape -> Shape -> Fin n -> Expr n
  Slice : List Nat -> List Nat -> List Nat -> Fin n -> Expr n
  DynamicSlice : List (Fin n) -> List Nat -> Fin n -> Expr n
  Concat : Nat -> Fin n -> Fin n -> Expr n
  Diag : Fin n -> Expr n
  Triangle : (lower : Bool) -> Fin n -> Expr n
  Transpose : List Nat -> Fin n -> Expr n
  Identity : Primitive dtype => Nat -> Expr n
  Broadcast : Primitive dtype => Shape -> Shape -> Fin n -> Expr n
  -- Map : Fn a (Expr n) -> Vect a (Fin n) -> Shape -> Expr n
  Reduce : Fn 2 (Expr (S (S n))) -> Fin n -> List Nat -> Fin n -> Expr n
  Sort : Fn 2 (Expr (S (S n))) -> Nat -> Bool -> List (Fin n) -> Expr n
  Reverse : List Nat -> Fin n -> Expr n
  Eq : Fin n -> Fin n -> Expr n
  Ne : Fin n -> Fin n -> Expr n
  Add : Fin n -> Fin n -> Expr n
  Sub : Fin n -> Fin n -> Expr n
  Mul : Fin n -> Fin n -> Expr n
  Div : Fin n -> Fin n -> Expr n
  Pow : Fin n -> Fin n -> Expr n
  Lt : Fin n -> Fin n -> Expr n
  Gt : Fin n -> Fin n -> Expr n
  Le : Fin n -> Fin n -> Expr n
  Ge : Fin n -> Fin n -> Expr n
  And : Fin n -> Fin n -> Expr n
  Or : Fin n -> Fin n -> Expr n
  Min : Fin n -> Fin n -> Expr n
  Max : Fin n -> Fin n -> Expr n
  Not : Fin n -> Expr n
  Neg : Fin n -> Expr n
  Reciprocal : Fin n -> Expr n
  Abs : Fin n -> Expr n
  Ceil : Fin n -> Expr n
  Floor : Fin n -> Expr n
  Log : Fin n -> Expr n
  Exp : Fin n -> Expr n
  Logistic : Fin n -> Expr n
  Erf : Fin n -> Expr n
  Square : Fin n -> Expr n
  Sqrt : Fin n -> Expr n
  Sin : Fin n -> Expr n
  Cos : Fin n -> Expr n
  Tan : Fin n -> Expr n
  Asin : Fin n -> Expr n
  Acos : Fin n -> Expr n
  Atan : Fin n -> Expr n
  Sinh : Fin n -> Expr n
  Cosh : Fin n -> Expr n
  Tanh : Fin n -> Expr n
  Asinh : Fin n -> Expr n
  Acosh : Fin n -> Expr n
  Atanh : Fin n -> Expr n
  Argmin : Primitive out => Nat -> Fin n -> Expr n
  Argmax : Primitive out => Nat -> Fin n -> Expr n
  Select : Fin n -> Fin n -> Fin n -> Expr n
  Cond : Fin n -> Fn 1 (Expr (S n)) -> Fin n -> Fn 1 (Expr (S n)) -> Fin n -> Expr n
  Dot : Fin n -> Fin n -> Expr n
  Cholesky : Fin n -> Expr n
  TriangularSolve : Fin n -> Fin n -> Bool -> Expr n
  UniformFloatingPoint : Fin n -> Fin n -> Fin n -> Fin n -> Shape -> Expr n
  NormalFloatingPoint : Fin n -> Fin n -> Shape -> Expr n

export
Prelude.Eq (Expr n) where
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
  (Parameter spec pos) == (Parameter spec' pos') = (spec, pos) == (spec', pos')
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
  (Reduce (MkFn [p0, p1] monoid) neutral axes x) ==
    (Reduce (MkFn [p0', p1'] monoid') neutral' axes' x') =
      p0 == p0' && p1 == p1' && monoid == monoid' && neutral == neutral' && axes == axes' && x == x'
  (Sort (MkFn [p0, p1] comparator) dimension isStable operands) ==
    (Sort (MkFn [p0', p1'] comparator') dimension' isStable' operands') =
      p0 == p0'
      && p1 == p1'
      && comparator == comparator'
      && dimension == dimension'
      && isStable == isStable'
      && operands == operands'
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
  (Cond pred (MkFn [pt] fTrue) true (MkFn [pf] fFalse) false) ==
    (Cond pred' (MkFn [pt'] fTrue') true' (MkFn [pf'] fFalse') false') =
      pred == pred'
      && pt == pt'
      && fTrue == fTrue'
      && true == true'
      && pf == pf'
      && fFalse == fFalse'
      && false == false'
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
data Terms : Nat -> Nat -> Type where
  Nil : Terms n n
  (::) : Expr lower -> Terms (S lower) upper -> Terms lower upper

export
index : (i : Fin upper) -> Terms 0 upper -> Expr (finToNat i)
index i xs = impl 0 i xs where
  impl : (lower : Nat) -> (i : Fin rem) -> Terms lower (lower + rem) -> Expr (lower + finToNat i)
  impl 0 FZ (x :: _) = x 
  impl 0 (FS i) (_ :: xs) = impl 1 i xs
  impl (S lower) FZ (x :: _) = rewrite plusZeroRightNeutral lower in x 
  impl (S lower) (FS {k} i) (_ :: xs) =
    rewrite sym $ plusSuccRightSucc lower (finToNat i) in
            impl (S (S lower)) i (rewrite plusSuccRightSucc lower k in xs)

export
snoc : Expr n -> Terms 0 n -> Terms 0 (S n)
snoc x xs = impl x xs where
  impl : Expr hi -> Terms lo hi -> Terms lo (S hi)
  impl x [] = [x] 
  impl x (y :: ys) = y :: impl x ys

shift' : (n : Nat) -> Nat -> Fin m -> Fin (n + m)
shift' n bound x = if cast x >= bound then shift n x else rewrite sym $ plusCommutative m n in weakenN n x

reindex : (n : Nat) -> Nat -> Terms lo hi -> Terms (lo + n) (hi + n)
reindex n bound [] = []
reindex n bound (x :: xs) = rewrite plusCommutative lo n in impl x :: rewrite sym $ plusCommutative lo n in reindex n bound xs where
  impl : Expr p -> Expr (n + p)
  impl (FromLiteral {dtype} x) = FromLiteral {dtype} x
  impl (Parameter spec k) = Parameter spec k
  impl (Tuple ys) = Tuple [shift' n bound y | y <- ys]
  impl (GetTupleElement k x) = GetTupleElement k (shift' n bound x)
  impl (MinValue {dtype}) = MinValue {dtype} 
  impl (MaxValue {dtype}) = MaxValue {dtype}
  impl (MinFiniteValue {dtype}) = MinFiniteValue {dtype}
  impl (MaxFiniteValue {dtype}) = MaxFiniteValue {dtype}
  impl (ConvertElementType {dtype} x) = ConvertElementType {dtype} (shift' n bound x)
  impl (Reshape ks js x) = Reshape ks js (shift' n bound x)
  impl (Slice ks js is x) = Slice ks js is (shift' n bound x)
  impl (DynamicSlice ys ks x) = DynamicSlice [shift' n bound y | y <- ys] ks (shift' n bound x)
  impl (Concat k x y) = Concat k (shift' n bound x) (shift' n bound y)
  impl (Diag x) = Diag (shift' n bound x)
  impl (Triangle lower x) = Triangle lower (shift' n bound x)
  impl (Transpose ks x) = Transpose ks (shift' n bound x)
  impl (Identity {dtype} k) = Identity {dtype} k
  impl (Broadcast {dtype} ks js x) = Broadcast {dtype} ks js (shift' n bound x)
  impl (Reduce x y ks z) = ?reduce
  impl (Sort x k y ys) = ?sort
  impl (Reverse ks x) = Reverse ks (shift' n bound x)
  impl (Eq x y) = Eq (shift' n bound x) (shift' n bound y)
  impl (Ne x y) = Ne (shift' n bound x) (shift' n bound y)
  impl (Add x y) = Add (shift' n bound x) (shift' n bound y)
  impl (Sub x y) = Sub (shift' n bound x) (shift' n bound y)
  impl (Mul x y) = Mul (shift' n bound x) (shift' n bound y)
  impl (Div x y) = Div (shift' n bound x) (shift' n bound y)
  impl (Pow x y) = Pow (shift' n bound x) (shift' n bound y)
  impl (Lt x y) = Lt (shift' n bound x) (shift' n bound y)
  impl (Gt x y) = Gt (shift' n bound x) (shift' n bound y)
  impl (Le x y) = Le (shift' n bound x) (shift' n bound y)
  impl (Ge x y) = Ge (shift' n bound x) (shift' n bound y)
  impl (And x y) = And (shift' n bound x) (shift' n bound y)
  impl (Or x y) = Or (shift' n bound x) (shift' n bound y)
  impl (Min x y) = Min (shift' n bound x) (shift' n bound y)
  impl (Max x y) = Max (shift' n bound x) (shift' n bound y)
  impl (Not x) = Not (shift' n bound x)
  impl (Neg x) = Neg (shift' n bound x)
  impl (Reciprocal x) = Reciprocal (shift' n bound x)
  impl (Abs x) = Abs (shift' n bound x)
  impl (Ceil x) = Ceil (shift' n bound x)
  impl (Floor x) = Floor (shift' n bound x)
  impl (Log x) = Log (shift' n bound x)
  impl (Exp x) = Exp (shift' n bound x)
  impl (Logistic x) = Logistic (shift' n bound x)
  impl (Erf x) = Erf (shift' n bound x)
  impl (Square x) = Square (shift' n bound x)
  impl (Sqrt x) = Sqrt (shift' n bound x)
  impl (Sin x) = Sin (shift' n bound x)
  impl (Cos x) = Cos (shift' n bound x)
  impl (Tan x) = Tan (shift' n bound x)
  impl (Asin x) = Asin (shift' n bound x)
  impl (Acos x) = Acos (shift' n bound x)
  impl (Atan x) = Atan (shift' n bound x)
  impl (Sinh x) = Sinh (shift' n bound x)
  impl (Cosh x) = Cosh (shift' n bound x)
  impl (Tanh x) = Tanh (shift' n bound x)
  impl (Asinh x) = Asinh (shift' n bound x)
  impl (Acosh x) = Acosh (shift' n bound x)
  impl (Atanh x) = Atanh (shift' n bound x)
  impl (Argmin {out} k x) = Argmin {out} k (shift' n bound x)
  impl (Argmax {out} k x) = Argmax {out} k (shift' n bound x)
  impl (Select x y z) = Select (shift' n bound x) (shift' n bound y) (shift' n bound z)
  impl (Cond x y z w v) = ?cond
  impl (Dot x y) = Dot (shift' n bound x) (shift' n bound y)
  impl (Cholesky x) = Cholesky (shift' n bound x)
  impl (TriangularSolve x y z) = TriangularSolve (shift' n bound x) (shift' n bound y) z
  impl (UniformFloatingPoint x y z w ks) =
    UniformFloatingPoint (shift' n bound x) (shift' n bound y) (shift' n bound z) (shift' n bound w) ks
  impl (NormalFloatingPoint x y ks) = NormalFloatingPoint (shift' n bound x) (shift' n bound y) ks

plusCommutativeLeftParen : (a, b, c : Nat) -> (a + b) + c = (a + c) + b
plusCommutativeLeftParen a b c =
  rewrite sym $ plusAssociative a b c in
    rewrite plusCommutative b c in
      rewrite sym $ plusAssociative a c b
        in Refl

succNested : (a, b, c : Nat) -> S ((a + b) + c) = (a + S b) + c
succNested a b c =
  rewrite plusCommutative (a + b) c in
    rewrite plusSuccRightSucc c (a + b) in
      rewrite plusSuccRightSucc a b in
        rewrite plusCommutative c (a + S b) in
          Refl

ltePlusMiddle : (a, b, c : Nat) -> LTE (a + b) (a + c + b)
ltePlusMiddle a b c = rewrite Calc $
  |~ (a + c + b)
  ~~ (a + (c + b)) ... sym (plusAssociative a c b)
  ~~ (a + (b + c)) ... cong (a +) (plusCommutative c b)
  ~~ ((a + b) + c) ... plusAssociative a b c
  in lteAddRight (a + b)

mergeHelper : {n, m : _} -> Terms 0 n -> Terms 0 m -> ((s ** (Terms 0 (n + s), LTE m (n + s))), Maybe (Fin m))
mergeHelper xs ys = impl xs ys where

  extend : Terms lo p -> Terms p hi -> Terms lo hi
  extend xs [] = xs
  extend [] ys = ys
  extend (x :: xs) ys = x :: extend xs ys

  impl :
    {lo, nx, ny : Nat} ->
    Terms lo (lo + nx) ->
    Terms lo (lo + ny) ->
    ((s ** (Terms lo (lo + nx + s), LTE (lo + ny) (lo + nx + s))), Maybe (Fin (lo + ny)))
  impl {ny = 0} xs _ =
    let lte = rewrite plusZeroRightNeutral lo in rewrite plusZeroRightNeutral (lo + nx) in lteAddRight lo
        terms = rewrite plusZeroRightNeutral (lo + nx) in xs
     in ((0 ** (terms, lte)), Nothing)
  impl {nx = 0} _ ys =
    ((ny ** (rewrite plusZeroRightNeutral lo in ys, rewrite plusZeroRightNeutral lo in reflexive)), Nothing)
  impl {nx = S nx} {ny = S ny} (x :: xs) (y :: ys) =
    if x == y
    then let ys : Terms (S lo) (lo + S ny) = ys
             ((s' ** (terms, lte)), conflict) =
               impl {lo = S lo} {nx, ny} (rewrite plusSuccRightSucc lo nx in xs) (rewrite plusSuccRightSucc lo ny in ys)
             lte = rewrite sym $ succNested lo nx s' in rewrite sym $ plusSuccRightSucc lo ny in lte
          in ((s' ** (rewrite sym $ succNested lo nx s' in x :: terms, lte)), rewrite sym $ plusSuccRightSucc lo ny in conflict)
    else let ys = reindex (S nx) lo (y :: ys)  -- do we always reindex? what if an index points to a value that's not conflicted?
             terms : Terms lo (lo + S nx + S ny) = rewrite sym $ plusCommutativeLeftParen lo (S ny) (S nx) in extend (x :: xs) ys
          in ((S ny ** (terms, ltePlusMiddle lo (S ny) (S nx))), Just (rewrite sym $ plusSuccRightSucc lo ny in weakenN ny $ last {n = lo}))

||| Shift a `Fin p` by adding q and subtracting p.
shiftLTE : {p, q : _} -> Fin p -> LTE p q -> Fin q
shiftLTE FZ (LTESucc _) = last 
shiftLTE (FS x) (LTESucc lte) = FS (shiftLTE x lte)

||| Merge the two lists of terms. The resulting terms start with all terms in the LHS, as they appear in the LHS, then
||| continue with all terms in the RHS, starting at the first term in the RHS that conflicts with the LHS, and
||| continuing until the end of the RHS. Terms x and y conflict if x == y does not hold. For example, in pseudo-syntax:
|||
||| Equal lists
||| merge [a, b, c] [a, b, c] is [a, b, c]
|||
||| One list is a sublist of the other
||| merge [a, b, c] [a, b] is [a, b, c]
||| merge [a, b] [a, b, c] is [a, b, c]
|||
||| There is a conflict
||| merge [a, b, c, d] [a, b, e] is [a, b, c, d, e]
||| merge [a, b, c] [a, b, d, e] is [a, b, c, d, e]
|||
||| The number returned in the dependent pair is how many terms from the RHS have been rebased onto the end of the LHS.
|||
||| In addition to the full set of terms, `merge` returns two functions that can be used to update indices into the
||| original lists of terms into indices into the merged terms. The `Expr` at a specific index in one of the original
||| list of terms is the same `Expr` at the converted index in the merged terms, but only up to the resulting tree of
||| terms, since when we move terms in the list, we must also update indices contained within those terms to preserve
||| the structure of the graph.
export
merge : {n, m : _} -> Terms 0 n -> Terms 0 m -> (s ** (Fin n -> Fin s, Fin m -> Fin s, Terms 0 s))
merge xs ys =
  let ((s' ** (terms, lte)), conflict) = mergeHelper xs ys

      fn : Fin n -> Fin (n + s')
      fn = weakenN s'

      fm : Fin m -> Fin (n + s')
      fm x = case conflict of
        Nothing => weakenLTE x lte
        Just idx => if x < idx then weakenLTE x lte else shiftLTE x lte

   in ((n + s') ** (fn, fm, terms))

-- [a, b, r]
-- [a, b, c, d, e]
-- [a, b, r, c, d, e]
