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
data Fn : Nat -> Type -> Type where
  MkFn : {arity : _} -> Vect arity FullShape -> a -> Fn arity a

public export
data Node : Nat -> Type where
  FromLiteral : PrimitiveRW dtype ty => {shape : _} -> Literal shape ty -> Node n
  Parameter : FullShape -> Nat -> Node n
  Tuple : List (Fin n) -> Node n
  GetTupleElement : Nat -> Fin n -> Node n
  MinValue : Primitive dtype => Node n
  MaxValue : Primitive dtype => Node n
  MinFiniteValue : Primitive dtype => Node n
  MaxFiniteValue : Primitive dtype => Node n
  ConvertElementType : Primitive dtype => Fin n -> Node n
  Reshape : Shape -> Shape -> Fin n -> Node n
  Slice : List Nat -> List Nat -> List Nat -> Fin n -> Node n
  DynamicSlice : List (Fin n) -> List Nat -> Fin n -> Node n
  Concat : Nat -> Fin n -> Fin n -> Node n
  Diag : Fin n -> Node n
  Triangle : (lower : Bool) -> Fin n -> Node n
  Transpose : List Nat -> Fin n -> Node n
  Identity : Primitive dtype => Nat -> Node n
  Broadcast : Primitive dtype => Shape -> Shape -> Fin n -> Node n
  -- Map : Fn a (Node n) -> Vect a (Fin n) -> Shape -> Node n
  Reduce : Fn 2 (Node (S (S n))) -> Fin n -> List Nat -> Fin n -> Node n
  Sort : Fn 2 (Node (S (S n))) -> Nat -> Bool -> List (Fin n) -> Node n
  Reverse : List Nat -> Fin n -> Node n
  Eq : Fin n -> Fin n -> Node n
  Ne : Fin n -> Fin n -> Node n
  Add : Fin n -> Fin n -> Node n
  Sub : Fin n -> Fin n -> Node n
  Mul : Fin n -> Fin n -> Node n
  Div : Fin n -> Fin n -> Node n
  Pow : Fin n -> Fin n -> Node n
  Lt : Fin n -> Fin n -> Node n
  Gt : Fin n -> Fin n -> Node n
  Le : Fin n -> Fin n -> Node n
  Ge : Fin n -> Fin n -> Node n
  And : Fin n -> Fin n -> Node n
  Or : Fin n -> Fin n -> Node n
  Min : Fin n -> Fin n -> Node n
  Max : Fin n -> Fin n -> Node n
  Not : Fin n -> Node n
  Neg : Fin n -> Node n
  Reciprocal : Fin n -> Node n
  Abs : Fin n -> Node n
  Ceil : Fin n -> Node n
  Floor : Fin n -> Node n
  Log : Fin n -> Node n
  Exp : Fin n -> Node n
  Logistic : Fin n -> Node n
  Erf : Fin n -> Node n
  Square : Fin n -> Node n
  Sqrt : Fin n -> Node n
  Sin : Fin n -> Node n
  Cos : Fin n -> Node n
  Tan : Fin n -> Node n
  Asin : Fin n -> Node n
  Acos : Fin n -> Node n
  Atan : Fin n -> Node n
  Sinh : Fin n -> Node n
  Cosh : Fin n -> Node n
  Tanh : Fin n -> Node n
  Asinh : Fin n -> Node n
  Acosh : Fin n -> Node n
  Atanh : Fin n -> Node n
  Argmin : Primitive out => Nat -> Fin n -> Node n
  Argmax : Primitive out => Nat -> Fin n -> Node n
  Select : Fin n -> Fin n -> Fin n -> Node n
  Cond : Fin n -> Fn 1 (Node (S n)) -> Fin n -> Fn 1 (Node (S n)) -> Fin n -> Node n
  Dot : Fin n -> Fin n -> Node n
  Cholesky : Fin n -> Node n
  TriangularSolve : Fin n -> Fin n -> Bool -> Node n
  UniformFloatingPoint : Fin n -> Fin n -> Fin n -> Fin n -> Shape -> Node n
  NormalFloatingPoint : Fin n -> Fin n -> Shape -> Node n

export
Prelude.Eq (Node n) where
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

||| A graph as a topologically-sorted adjacency list. A `Graph lo hi` has up a maximum of `hi` nodes, with all but `lo`
||| filled.
public export
data Graph : Nat -> Nat -> Type where
  Nil : Graph n n
  (::) : Node lower -> Graph (S lower) upper -> Graph lower upper

||| The node at the specified index.
export
index : (i : Fin upper) -> Graph 0 upper -> Node (finToNat i)
index i xs = impl 0 i xs where
  impl : (lower : Nat) -> (i : Fin rem) -> Graph lower (lower + rem) -> Node (lower + finToNat i)
  impl 0 FZ (x :: _) = x 
  impl 0 (FS i) (_ :: xs) = impl 1 i xs
  impl (S lower) FZ (x :: _) = rewrite plusZeroRightNeutral lower in x 
  impl (S lower) (FS {k} i) (_ :: xs) =
    rewrite sym $ plusSuccRightSucc lower (finToNat i) in
            impl (S (S lower)) i (rewrite plusSuccRightSucc lower k in xs)

||| Append a new node to the graph, at the end of the adjacency list.
export
snoc : Node n -> Graph 0 n -> Graph 0 (S n)
snoc x xs = impl x xs where
  impl : Node hi -> Graph lo hi -> Graph lo (S hi)
  impl x [] = [x] 
  impl x (y :: ys) = y :: impl x ys

reindex : (n : Nat) -> Nat -> Graph lo hi -> Graph (lo + n) (hi + n)
reindex n bound [] = []
reindex n bound (x :: xs) =
  rewrite plusCommutative lo n in impl x :: rewrite sym $ plusCommutative lo n in reindex n bound xs

  where

  shift' : Fin m -> Fin (n + m)
  shift' x = if cast x >= bound then shift n x else rewrite sym $ plusCommutative m n in weakenN n x

  impl : Node p -> Node (n + p)
  impl (FromLiteral {dtype} x) = FromLiteral {dtype} x
  impl (Parameter spec k) = Parameter spec k
  impl (Tuple ys) = Tuple [shift' y | y <- ys]
  impl (GetTupleElement k x) = GetTupleElement k (shift' x)
  impl (MinValue {dtype}) = MinValue {dtype} 
  impl (MaxValue {dtype}) = MaxValue {dtype}
  impl (MinFiniteValue {dtype}) = MinFiniteValue {dtype}
  impl (MaxFiniteValue {dtype}) = MaxFiniteValue {dtype}
  impl (ConvertElementType {dtype} x) = ConvertElementType {dtype} (shift' x)
  impl (Reshape ks js x) = Reshape ks js (shift' x)
  impl (Slice ks js is x) = Slice ks js is (shift' x)
  impl (DynamicSlice ys ks x) = DynamicSlice [shift' y | y <- ys] ks (shift' x)
  impl (Concat k x y) = Concat k (shift' x) (shift' y)
  impl (Diag x) = Diag (shift' x)
  impl (Triangle lower x) = Triangle lower (shift' x)
  impl (Transpose ks x) = Transpose ks (shift' x)
  impl (Identity {dtype} k) = Identity {dtype} k
  impl (Broadcast {dtype} ks js x) = Broadcast {dtype} ks js (shift' x)
  impl (Reduce x y ks z) = ?reduce
  impl (Sort x k y ys) = ?sort
  impl (Reverse ks x) = Reverse ks (shift' x)
  impl (Eq x y) = Eq (shift' x) (shift' y)
  impl (Ne x y) = Ne (shift' x) (shift' y)
  impl (Add x y) = Add (shift' x) (shift' y)
  impl (Sub x y) = Sub (shift' x) (shift' y)
  impl (Mul x y) = Mul (shift' x) (shift' y)
  impl (Div x y) = Div (shift' x) (shift' y)
  impl (Pow x y) = Pow (shift' x) (shift' y)
  impl (Lt x y) = Lt (shift' x) (shift' y)
  impl (Gt x y) = Gt (shift' x) (shift' y)
  impl (Le x y) = Le (shift' x) (shift' y)
  impl (Ge x y) = Ge (shift' x) (shift' y)
  impl (And x y) = And (shift' x) (shift' y)
  impl (Or x y) = Or (shift' x) (shift' y)
  impl (Min x y) = Min (shift' x) (shift' y)
  impl (Max x y) = Max (shift' x) (shift' y)
  impl (Not x) = Not (shift' x)
  impl (Neg x) = Neg (shift' x)
  impl (Reciprocal x) = Reciprocal (shift' x)
  impl (Abs x) = Abs (shift' x)
  impl (Ceil x) = Ceil (shift' x)
  impl (Floor x) = Floor (shift' x)
  impl (Log x) = Log (shift' x)
  impl (Exp x) = Exp (shift' x)
  impl (Logistic x) = Logistic (shift' x)
  impl (Erf x) = Erf (shift' x)
  impl (Square x) = Square (shift' x)
  impl (Sqrt x) = Sqrt (shift' x)
  impl (Sin x) = Sin (shift' x)
  impl (Cos x) = Cos (shift' x)
  impl (Tan x) = Tan (shift' x)
  impl (Asin x) = Asin (shift' x)
  impl (Acos x) = Acos (shift' x)
  impl (Atan x) = Atan (shift' x)
  impl (Sinh x) = Sinh (shift' x)
  impl (Cosh x) = Cosh (shift' x)
  impl (Tanh x) = Tanh (shift' x)
  impl (Asinh x) = Asinh (shift' x)
  impl (Acosh x) = Acosh (shift' x)
  impl (Atanh x) = Atanh (shift' x)
  impl (Argmin {out} k x) = Argmin {out} k (shift' x)
  impl (Argmax {out} k x) = Argmax {out} k (shift' x)
  impl (Select x y z) = Select (shift' x) (shift' y) (shift' z)
  impl (Cond x y z w v) = ?cond
  impl (Dot x y) = Dot (shift' x) (shift' y)
  impl (Cholesky x) = Cholesky (shift' x)
  impl (TriangularSolve x y z) = TriangularSolve (shift' x) (shift' y) z
  impl (UniformFloatingPoint x y z w ks) =
    UniformFloatingPoint (shift' x) (shift' y) (shift' z) (shift' w) ks
  impl (NormalFloatingPoint x y ks) = NormalFloatingPoint (shift' x) (shift' y) ks

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

mergeHelper : {n, m : _} -> Graph 0 n -> Graph 0 m -> ((s ** (Graph 0 (n + s), LTE m (n + s))), Maybe (Fin m))
mergeHelper xs ys = impl xs ys where

  extend : Graph lo p -> Graph p hi -> Graph lo hi
  extend xs [] = xs
  extend [] ys = ys
  extend (x :: xs) ys = x :: extend xs ys

  impl :
    {lo, nx, ny : Nat} ->
    Graph lo (lo + nx) ->
    Graph lo (lo + ny) ->
    ((s ** (Graph lo (lo + nx + s), LTE (lo + ny) (lo + nx + s))), Maybe (Fin (lo + ny)))
  impl {ny = 0} xs _ =
    let lte = rewrite plusZeroRightNeutral lo in rewrite plusZeroRightNeutral (lo + nx) in lteAddRight lo
        terms = rewrite plusZeroRightNeutral (lo + nx) in xs
     in ((0 ** (terms, lte)), Nothing)
  impl {nx = 0} _ ys =
    ((ny ** (rewrite plusZeroRightNeutral lo in ys, rewrite plusZeroRightNeutral lo in reflexive)), Nothing)
  impl {nx = S nx} {ny = S ny} (x :: xs) (y :: ys) =
    if x == y
    then let ys : Graph (S lo) (lo + S ny) = ys
             ((s' ** (terms, lte)), conflict) =
               impl {lo = S lo} {nx, ny} (rewrite plusSuccRightSucc lo nx in xs) (rewrite plusSuccRightSucc lo ny in ys)
             lte = rewrite sym $ succNested lo nx s' in rewrite sym $ plusSuccRightSucc lo ny in lte
          in ((s' ** (rewrite sym $ succNested lo nx s' in x :: terms, lte)), rewrite sym $ plusSuccRightSucc lo ny in conflict)
    else let ys = reindex (S nx) lo (y :: ys)  -- do we always reindex? what if an index points to a value that's not conflicted?
             terms : Graph lo (lo + S nx + S ny) = rewrite sym $ plusCommutativeLeftParen lo (S ny) (S nx) in extend (x :: xs) ys
          in ((S ny ** (terms, ltePlusMiddle lo (S ny) (S nx))), Just (rewrite sym $ plusSuccRightSucc lo ny in weakenN ny $ last {n = lo}))

||| Shift a `Fin p` by adding q and subtracting p.
shiftLTE : {p, q : _} -> Fin p -> LTE p q -> Fin q
shiftLTE FZ (LTESucc _) = last 
shiftLTE (FS x) (LTESucc lte) = FS (shiftLTE x lte)

||| Merge two topologically sorted graphs into one topologically sorted graph. The input graphs can share nodes. The
||| resulting graph starts with all nodes in the LHS, as they appear in the LHS. It then contains any nodes in the RHS
||| not in the LHS. If the input graphs diverge at some node (where two nodes `x`, `y` do not satisfy `x` == `y`), then
||| all nodes from that node forward are appended after the nodes of the LHS. For example, in pseudo-syntax:
|||
||| If the input graphs are the same
||| merge [a, b, c] [a, b, c] produces [a, b, c]
|||
||| If one input graph is a subset of the other
||| merge [a, b, c] [a, b] produces [a, b, c]
||| merge [a, b] [a, b, c] produces [a, b, c]
|||
||| If the two graphs diverge
||| merge [a, b, c, d] [a, b, e] produces [a, b, c, d, e]
||| merge [a, b, c] [a, b, d, e] produces [a, b, c, d, e]
|||
||| `merge` also returns functions to update indices into the first two graphs so that they are valid in the resulting
||| graph.
export
merge : {n, m : _} -> Graph 0 n -> Graph 0 m -> (s ** (Fin n -> Fin s, Fin m -> Fin s, Graph 0 s))
merge xs ys =
  let ((s' ** (terms, lte)), conflict) = mergeHelper xs ys

      fn : Fin n -> Fin (n + s')
      fn = weakenN s'

      fm : Fin m -> Fin (n + s')
      fm x = case conflict of
        Nothing => weakenLTE x lte
        Just idx => if x < idx then weakenLTE x lte else shiftLTE x lte

   in ((n + s') ** (fn, fm, terms))
