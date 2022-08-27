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
import Data.Hashable
import Data.SnocList

import Compiler.LiteralRW
import Compiler.Xla.TensorFlow.Compiler.Xla.XlaData
import Literal
import Primitive
import Util
import Util.Hashable
import Compiler.SnocShape

namespace Concat
  public export
  data Concat : (xs, init, rest : List a) -> Type where
    Nil : Concat xs [] xs
    (::) : (x : a) -> Concat xs init rest -> Concat (x :: xs) (x :: init) rest

  public export
  ceq1 : Concat xs init [m] -> Concat xs init' [m'] -> (init' = init, m' = m)
  ceq1 [] [] = (Refl, Refl)
  ceq1 (x :: xs) (x :: ys) = let (ieq, meq) = ceq1 xs ys in (cong (x ::) ieq, meq)

  initEmpty : Concat xs [] ys -> xs = ys
  initEmpty [] = Refl

  public export
  ceq2 : Concat xs init [m, n] -> Concat xs init' [m', n'] -> (init' = init, m' = m, n' = n)
  ceq2 [] [] = (Refl, Refl, Refl)
  ceq2 [] (_ :: ys) with (initEmpty {xs} [])
    ceq2 [] (_ :: _ :: _) | Refl impossible
  ceq2 (_ :: ys) [] with (initEmpty {xs} [])
    ceq2 (_ :: _ :: _) [] | Refl impossible
  ceq2 (x :: xs) (x :: ys) = let (ieq, meq, neq) = ceq2 xs ys in (cong (x ::) ieq, meq, neq)

  public export
  eqIfHalvesEqual : Concat xs init rest -> Concat xs' init rest -> xs' = xs
  eqIfHalvesEqual [] [] = Refl
  eqIfHalvesEqual (x :: xs) (x :: ys) = cong (x ::) (eqIfHalvesEqual xs ys)

data ElemAt : Nat -> a -> List a -> Type where
  Tail : ElemAt 0 x (x :: xs)
  Init : (x : a) -> ElemAt n y ys -> ElemAt (S n) y (x :: ys)

namespace Pivot
  public export
  data Pivot : Nat -> List a -> List a -> a -> List a -> Type where
    Nil : Pivot 0 (x :: xs) [] x xs
    (::) : (x : a) -> Pivot n xs lead mid tail -> Pivot (S n) (x :: xs) (x :: lead) mid tail

mutual
  public export
  data Expr : Shape -> Type where
    FromLiteral : PrimitiveRW dtype ty => {shape : _} -> Literal shape ty -> Expr shape
    Parameter : Primitive dtype =>
                (position : Nat) ->
                {shape : _} ->
                String ->
                Expr shape
    -- GetTupleElement : {0 sts : Vect (S n) (Shape, Shape, Type)} -> (idx : Fin (S n)) ->
    --                   let (leading, operational,) = index idx sts
    --                   in TupleExpr {n} sts -> Expr leading operational
    MinFiniteValue : Primitive dtype => Expr shape
    MaxFiniteValue : Primitive dtype => Expr shape
    ConvertElementType : Primitive dtype => Expr shape -> Expr shape
    Reshape : {from, to : _} ->
              {auto 0 sizesEqual : product from = product to} ->
              Expr from ->
              Expr to
    Slice : (starts : List Nat) ->
            (stops : List Nat) ->
            {auto 0 _ : length starts = length shape} ->
            {auto 0 _ : length stops = length shape} ->
            {auto 0 _ : Each LTE starts stops} ->
            {auto 0 _ : Each LTE stops shape} ->
            Expr shape ->
            Expr shape
    DynamicSlice : (starts : List (Expr [])) ->
                   {shape : _} ->
                   Expr shape ->
                   (sizes : List Nat) ->
                  --  {auto 0 _ : length starts = length shape} ->
                   {auto 0 _ : length sizes = length shape} ->
                   {auto 0 _ : Each LTE sizes shape} ->
                   Expr sizes
    Concat : (axis : Nat) ->
             {front, end : _} ->
             Expr front ->
             Expr end ->
             {auto 0 _ : InBounds axis front} ->
             {auto 0 _ : InBounds axis end} ->
             {auto 0 shapesConcatenable : deleteAt axis front = deleteAt axis end} ->
             {auto 0 retType : res = replaceAt axis (index axis front + index axis end) front} ->
             Expr res
    Diag : {from : _} -> Expr from -> Concat from leading [n, n] => Concat to leading [n] => Expr to
    Triangle : (lower : Bool) ->
               {from : _} ->
               Expr from ->
               Concat from leading [n, n] =>
               Concat to leading [n, n] =>
               Expr to
    Transpose : List Nat -> Expr shape -> Expr shape
    Identity : Primitive dtype => {n : _} -> Expr [n, n]
    Broadcast : Primitive dtype =>
                {from, to : _} ->
                {auto 0 _ : Broadcastable from to} ->
                Expr from ->
                Expr to
    -- Map : Fn n Expr -> Vect n Expr -> Shape -> Expr  -- do we really care about map? It's the only n-arity Fn
    Reduce : (Expr [], Expr [], Expr []) ->
             Expr [] ->
             (axes : List Nat) ->
             {from : _} ->
             Expr from ->
             Expr to
    Sort : (Expr [], Expr [], Expr []) ->
           (axis : Nat) ->
           (isStable : Bool) ->
           List (Expr shape) ->
           Expr shape
    Reverse : (axes : List Nat) -> Expr shape -> Expr shape
    Eq : Expr shape -> Expr shape -> Expr shape
    Ne : Expr shape -> Expr shape -> Expr shape
    Lt : Expr shape -> Expr shape -> Expr shape
    Gt : Expr shape -> Expr shape -> Expr shape
    Le : Expr shape -> Expr shape -> Expr shape
    Ge : Expr shape -> Expr shape -> Expr shape
    Add : Expr shape -> Expr shape -> Expr shape
    Sub : Expr shape -> Expr shape -> Expr shape
    Mul : Expr shape -> Expr shape -> Expr shape
    Div : Expr shape -> Expr shape -> Expr shape
    Pow : Expr shape -> Expr shape -> Expr shape
    And : Expr shape -> Expr shape -> Expr shape
    Or : Expr shape -> Expr shape -> Expr shape
    Min : Expr shape -> Expr shape -> Expr shape
    Max : Expr shape -> Expr shape -> Expr shape
    Not : Expr shape -> Expr shape
    Neg : Expr shape -> Expr shape
    Reciprocal : Expr shape -> Expr shape
    Abs : Expr shape -> Expr shape
    Ceil : Expr shape -> Expr shape
    Floor : Expr shape -> Expr shape
    Log : Expr shape -> Expr shape
    Exp : Expr shape -> Expr shape
    Logistic : Expr shape -> Expr shape
    Erf : Expr shape -> Expr shape
    Square : Expr shape -> Expr shape
    Sqrt : Expr shape -> Expr shape
    Sin : Expr shape -> Expr shape
    Cos : Expr shape -> Expr shape
    Tan : Expr shape -> Expr shape
    Asin : Expr shape -> Expr shape
    Acos : Expr shape -> Expr shape
    Atan : Expr shape -> Expr shape
    Sinh : Expr shape -> Expr shape
    Cosh : Expr shape -> Expr shape
    Tanh : Expr shape -> Expr shape
    Asinh : Expr shape -> Expr shape
    Acosh : Expr shape -> Expr shape
    Atanh : Expr shape -> Expr shape
    Select : Expr shape ->
             Expr shape ->
             Expr shape ->
             Expr shape
    Cond : {fs, ts : _} -> Expr [] -> (Expr fs, Expr s) -> Expr fs -> (Expr ts, Expr s) -> Expr ts -> Expr s
    Dot : {shape, shape' : _} -> Expr shape -> Expr shape' -> Expr shape''
    Cholesky : Expr shape -> Concat shape leading [S n, S n] => Expr shape
    TriangularSolve : {l, r : _} ->
                      Expr l ->
                      Expr r ->
                      Concat l leading [n, n] =>
                      Concat r leading [n, k] =>
                      (lower : Bool) ->
                      Expr r

  data TupleExpr : Vect (S n) Shape -> Type where
    -- Tuple : (xs : Vect (S n) (s ** Expr s)) -> TupleExpr {n} (map fst xs)
    UniformFloatingPoint : Expr [] ->
                          Expr [] ->
                          {shape : _} ->
                          Expr shape ->
                          Expr shape ->
                          TupleExpr [[], shape]
    NormalFloatingPoint : Expr [] -> Expr [] -> {shape : _} -> TupleExpr [[], shape]

mutual
  export
  Prelude.Eq (Expr shape) where
    (FromLiteral lit) == (FromLiteral lit') = hash lit == hash lit'
    (Parameter position name) == (Parameter position' name') = (position, name) == (position', name')
    -- (GetTupleElement idx tuple) == (GetTupleElement idx' tuple') = ?eq
    MinFiniteValue == MinFiniteValue = True
    MaxFiniteValue == MaxFiniteValue = True
    (ConvertElementType operand) == (ConvertElementType operand') = operand == operand'
    (Reshape {from} x) == (Reshape {from=from'} x') = case decEq from from' of
      Yes eq => x == rewrite eq in x'
      No _ => False
    (Slice starts stops x) == (Slice starts' stops' x') =
      (starts, stops) == (starts', stops') && x == x'
    (DynamicSlice starts {shape} x _) == (DynamicSlice starts' {shape=shape'} x' _) =
      case decEq shape shape' of
        Yes eq => (assert_total $ starts == starts') && x == rewrite eq in x'
        No _ => False
    (Concat {front=f} {end=e} axis x y) == (Concat {front=f'} {end=e'} axis' x' y') =
      -- we *could* use proofs here if we really wanted. It wouldn't speed it up much though cos
      -- we're working with shapes which are small things.
      axis == axis' && case decEq f f' of
        Yes eq => (x == rewrite eq in x') && case decEq e e' of
          Yes eq => y == rewrite eq in y'
          No _ => False
        No _ => False
    (Diag @{pfrom} @{pto} x) == (Diag @{pfrom'} @{pto'} x') =
      let (leq, neq) = ceq1 pto pto'
          pfrom' = rewrite sym neq in rewrite sym leq in pfrom'
       in x == rewrite sym (eqIfHalvesEqual pfrom pfrom') in x'
    (Triangle @{pfrom} @{pto} lower x) == (Triangle @{pfrom'} @{pto'} lower' x') =
      let (leq, neq, _) = ceq2 pto pto'
          pfrom' = rewrite sym neq in rewrite sym leq in pfrom'
       in lower == lower' && x == rewrite sym (eqIfHalvesEqual pfrom pfrom') in x'
    (Transpose ordering x) == (Transpose ordering' x') = ordering == ordering' && x == x'
    Identity == Identity = True
    (Broadcast {from=f} x) == (Broadcast {from=f'} x') = case decEq f f' of
      Yes eq => x == rewrite eq in x'
      No _ => False
    -- (Map {n} (MkFn params f) xs dims) == (Map {n=n'} (MkFn params' f') xs' dims') =
    --   case decEq n n' of
    --     Yes eq =>
    --       (assert_total $ params == rewrite eq in params')
    --       && f == f'
    --       && (assert_total $ xs == rewrite eq in xs')
    --       && dims == dims'
    --     No _ => False
    (Reduce (p0, p1, monoid) neutral axes {from=f} x) ==
      (Reduce (p0', p1', monoid') neutral' axes' {from=f'} x') =
        p0 == p0' && p1 == p1' && monoid == monoid' && neutral == neutral' && axes == axes' &&
          case decEq f f' of
            Yes eq => x == rewrite eq in x'
            No _ => False
    (Sort (p0, p1, comparator) dimension isStable operands) ==
      (Sort (p0', p1', comparator') dimension' isStable' operands') =
        p0 == p0'
        && p1 == p1'
        && comparator == comparator'
        && dimension == dimension'
        && isStable == isStable'
        && (assert_total $ operands == operands')
    (Reverse axes expr) == (Reverse axes' expr') = axes == axes' && expr == expr'
    (Eq l r) == (Eq l' r') = l == l' && r == r'
    (Ne l r) == (Ne l' r') = l == l' && r == r'
    (Lt l r) == (Lt l' r') = l == l' && r == r'
    (Gt l r) == (Gt l' r') = l == l' && r == r'
    (Le l r) == (Le l' r') = l == l' && r == r'
    (Ge l r) == (Ge l' r') = l == l' && r == r'
    (Add l r) == (Add l' r') = l == l' && r == r'
    (Sub l r) == (Sub l' r') = l == l' && r == r'
    (Mul l r) == (Mul l' r') = l == l' && r == r'
    (Div l r) == (Div l' r') = l == l' && r == r'
    (Pow l r) == (Pow l' r') = l == l' && r == r'
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
    (Select pred f t) == (Select pred' f' t') = pred == pred' && f == f' && t == t'
--    (Cond pred (pt, fTrue) true (pf, fFalse) false) ==
--      (Cond pred' (pt', fTrue') true' (pf', fFalse') false') =
--        pred == pred'
--        && pt == pt'
--        && fTrue == fTrue'
--        && true == true'
--        && pf == pf'
--        && fFalse == fFalse'
--        && false == false'
    -- (Dot x y) == (Dot x' y') = x == x' && y == y'
    (Cholesky x) == (Cholesky x') = x == x'
    (TriangularSolve @{pl} @{pr} x y lower) == (TriangularSolve @{pl'} @{pr'} x' y' lower') =
      let (leq, neq, _) = ceq2 pr pr'
          pl' = rewrite sym neq in rewrite sym leq in pl'
       in x == (rewrite sym (eqIfHalvesEqual pl pl') in x') && y == y' && lower == lower'
    _ == _ = False

  export
  Prelude.Eq (TupleExpr sts) where
    (UniformFloatingPoint key initialState minval maxval) ==
      (UniformFloatingPoint key' initialState' minval' maxval') =
        key == key' && initialState == initialState' && minval == minval' && maxval == maxval'
    (NormalFloatingPoint key initialState) == (NormalFloatingPoint key' initialState') =
      key == key' && initialState == initialState'
    _ == _ = False

mutual
  export
  Hashable (Expr shape) where
    hashWithSalt salt (FromLiteral {shape} {dtype} lit) =
      salt `hashWithSalt` ("FromLiteral", typeString {dtype}, shape, lit)
    hashWithSalt salt (Parameter {dtype} position {shape} name) =
      salt `hashWithSalt` ("Parameter", typeString {dtype}, shape, position, name)
    -- hashWithSalt salt (GetTupleElement idx tuple) = ?gtehws
--      salt `hashWithSalt` ("GetTupleElement", idx) `hashWithSalt` tuple
    hashWithSalt salt (MinFiniteValue {dtype}) =
      salt `hashWithSalt` ("MinFiniteValue", typeString {dtype})
    hashWithSalt salt (MaxFiniteValue {dtype}) =
      salt `hashWithSalt` ("MaxFiniteValue", typeString {dtype})
    hashWithSalt salt (ConvertElementType {dtype} operand) =
      salt `hashWithSalt` ("ConvertElementType", typeString {dtype}) `hashWithSalt` operand
    hashWithSalt salt (Reshape {from, to} x) =
      salt `hashWithSalt` ("Reshape", from, to) `hashWithSalt` x
    hashWithSalt salt (Slice starts stops x) =
      salt `hashWithSalt` ("Slice", starts, stops) `hashWithSalt` x
    hashWithSalt salt (DynamicSlice starts sizes x) =
      let salt = salt `hashWithSalt` "DynamicSlice"
          salt = assert_total $ salt `hashWithSalt` starts
      in salt `hashWithSalt` sizes `hashWithSalt` x
    hashWithSalt salt (Concat axis x y) =
      salt `hashWithSalt` ("Concat", axis) `hashWithSalt` x `hashWithSalt` y
    hashWithSalt salt (Diag x) = salt `hashWithSalt` "Diag" `hashWithSalt` x
    hashWithSalt salt (Triangle lower x) = salt `hashWithSalt` ("Triangle", lower) `hashWithSalt` x
    hashWithSalt salt (Transpose ordering x) =
        salt `hashWithSalt` ("Transpose", ordering) `hashWithSalt` x
    hashWithSalt salt (Identity {dtype} {n}) = salt `hashWithSalt` ("Identity", typeString {dtype}, n)
    hashWithSalt salt (Broadcast {from, to} x) =
      salt `hashWithSalt` ("Broadcast", from, to) `hashWithSalt` x
    -- hashWithSalt salt (Map (MkFn params f) xs dims) =
    --   let salt = salt `hashWithSalt` "Map"
    --       salt = assert_total $ salt `hashWithSalt` params
    --       salt = salt `hashWithSalt` f
    --       salt = assert_total $ salt `hashWithSalt` xs
    --    in salt `hashWithSalt` dims
    hashWithSalt salt (Reduce (p0, p1, monoid) neutral axes x) = salt
      `hashWithSalt` "Reduce"
      `hashWithSalt` p0
      `hashWithSalt` p1
      `hashWithSalt` monoid
      `hashWithSalt` neutral 
      `hashWithSalt` axes
      `hashWithSalt` x
    hashWithSalt salt (Sort (p0, p1, comparator) dimension isStable operands) =
      let salt = salt
            `hashWithSalt` "Sort"
            `hashWithSalt` p0
            `hashWithSalt` p1
            `hashWithSalt` (dimension, isStable)
      in assert_total $ salt `hashWithSalt` operands
    hashWithSalt salt (Reverse axes operand) =
      salt `hashWithSalt` ("Reverse", axes) `hashWithSalt` operand
    hashWithSalt salt (Eq l r) = salt `hashWithSalt` "Eq" `hashWithSalt` l `hashWithSalt` r
    hashWithSalt salt (Ne l r) = salt `hashWithSalt` "Ne" `hashWithSalt` l `hashWithSalt` r
    hashWithSalt salt (Lt l r) = salt `hashWithSalt` "Lt" `hashWithSalt` l `hashWithSalt` r
    hashWithSalt salt (Gt l r) = salt `hashWithSalt` "Gt" `hashWithSalt` l `hashWithSalt` r
    hashWithSalt salt (Le l r) = salt `hashWithSalt` "Le" `hashWithSalt` l `hashWithSalt` r
    hashWithSalt salt (Ge l r) = salt `hashWithSalt` "Ge" `hashWithSalt` l `hashWithSalt` r
    hashWithSalt salt (Add l r) = salt `hashWithSalt` "Add" `hashWithSalt` l `hashWithSalt` r
    hashWithSalt salt (Sub l r) = salt `hashWithSalt` "Sub" `hashWithSalt` l `hashWithSalt` r
    hashWithSalt salt (Mul l r) = salt `hashWithSalt` "Mul" `hashWithSalt` l `hashWithSalt` r
    hashWithSalt salt (Div l r) = salt `hashWithSalt` "Div" `hashWithSalt` l `hashWithSalt` r
    hashWithSalt salt (Pow l r) = salt `hashWithSalt` "Pow" `hashWithSalt` l `hashWithSalt` r
    hashWithSalt salt (And l r) = salt `hashWithSalt` "And" `hashWithSalt` l `hashWithSalt` r
    hashWithSalt salt (Or l r) = salt `hashWithSalt` "Or" `hashWithSalt` l `hashWithSalt` r
    hashWithSalt salt (Min l r) = salt `hashWithSalt` "Min" `hashWithSalt` l `hashWithSalt` r
    hashWithSalt salt (Max l r) = salt `hashWithSalt` "Max" `hashWithSalt` l `hashWithSalt` r
    hashWithSalt salt (Not expr) = salt `hashWithSalt` "Not" `hashWithSalt` expr
    hashWithSalt salt (Neg expr) = salt `hashWithSalt` "Neg" `hashWithSalt` expr
    hashWithSalt salt (Reciprocal expr) = salt `hashWithSalt` "Reciprocal" `hashWithSalt` expr
    hashWithSalt salt (Abs expr) = salt `hashWithSalt` "Abs" `hashWithSalt` expr
    hashWithSalt salt (Ceil expr) = salt `hashWithSalt` "Ceil" `hashWithSalt` expr
    hashWithSalt salt (Floor expr) = salt `hashWithSalt` "Floor" `hashWithSalt` expr
    hashWithSalt salt (Log expr) = salt `hashWithSalt` "Log" `hashWithSalt` expr
    hashWithSalt salt (Exp expr) = salt `hashWithSalt` "Exp" `hashWithSalt` expr
    hashWithSalt salt (Logistic expr) = salt `hashWithSalt` "Logistic" `hashWithSalt` expr
    hashWithSalt salt (Erf expr) = salt `hashWithSalt` "Erf" `hashWithSalt` expr
    hashWithSalt salt (Square expr) = salt `hashWithSalt` "Square" `hashWithSalt` expr
    hashWithSalt salt (Sqrt expr) = salt `hashWithSalt` "Sqrt" `hashWithSalt` expr
    hashWithSalt salt (Sin expr) = salt `hashWithSalt` "Sin" `hashWithSalt` expr
    hashWithSalt salt (Cos expr) = salt `hashWithSalt` "Cos" `hashWithSalt` expr
    hashWithSalt salt (Tan expr) = salt `hashWithSalt` "Tan" `hashWithSalt` expr
    hashWithSalt salt (Asin expr) = salt `hashWithSalt` "Asin" `hashWithSalt` expr
    hashWithSalt salt (Acos expr) = salt `hashWithSalt` "Acos" `hashWithSalt` expr
    hashWithSalt salt (Atan expr) = salt `hashWithSalt` "Atan" `hashWithSalt` expr
    hashWithSalt salt (Sinh expr) = salt `hashWithSalt` "Sinh" `hashWithSalt` expr
    hashWithSalt salt (Cosh expr) = salt `hashWithSalt` "Cosh" `hashWithSalt` expr
    hashWithSalt salt (Tanh expr) = salt `hashWithSalt` "Tanh" `hashWithSalt` expr
    hashWithSalt salt (Asinh expr) = salt `hashWithSalt` "Asinh" `hashWithSalt` expr
    hashWithSalt salt (Acosh expr) = salt `hashWithSalt` "Acosh" `hashWithSalt` expr
    hashWithSalt salt (Atanh expr) = salt `hashWithSalt` "Atanh" `hashWithSalt` expr
    hashWithSalt salt (Select pred f t) =
      salt `hashWithSalt` "Select" `hashWithSalt` pred `hashWithSalt` f `hashWithSalt` t
    hashWithSalt salt (Cond pred (pt, fTrue) true (pf, fFalse) false) = salt
      `hashWithSalt` "Cond"
      `hashWithSalt` pred
      `hashWithSalt` pt
      `hashWithSalt` fTrue
      `hashWithSalt` true
      `hashWithSalt` pf
      `hashWithSalt` fFalse
      `hashWithSalt` false
    hashWithSalt salt (Dot x y) = salt `hashWithSalt` "Dot" `hashWithSalt` x `hashWithSalt` y
    hashWithSalt salt (Cholesky x) = salt `hashWithSalt` "Cholesky" `hashWithSalt` x
    hashWithSalt salt (TriangularSolve x y lower) =
      salt `hashWithSalt` "TriangularSolve" `hashWithSalt` x `hashWithSalt` y `hashWithSalt` lower

  export
  Hashable (TupleExpr shapes) where
    hashWithSalt salt x = ?hws
    -- hashWithSalt salt (Tuple xs) =
    --   let salt = salt `hashWithSalt` "Tuple"
    --    in assert_total $ hashWithSalt salt xs
   -- hashWithSalt salt (UniformFloatingPoint key initialState minval maxval shape) = salt
   --     `hashWithSalt` "UniformFloatingPoint"
   --     `hashWithSalt` key
   --     `hashWithSalt` initialState
   --     `hashWithSalt` minval
   --     `hashWithSalt` maxval
   --     `hashWithSalt` shape
   -- hashWithSalt salt (NormalFloatingPoint key initialState shape) = salt
   --     `hashWithSalt` "NormalFloatingPoint"
   --     `hashWithSalt` key
   --     `hashWithSalt` initialState
   --     `hashWithSalt` shape
