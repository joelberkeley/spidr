module Unit.Compiler.TestExpr

import Decidable.Equality
import Data.Nat

import Compiler.Expr
import Literal
import Primitive

import Utils.Cases
import Utils.Comparison

Show (Expr n) where
  show (FromLiteral {shape} _) = "FromLiteral {shape = \{show shape}}"
  show (Concat axis x y) = "Concat {axis = \{show axis}} \{show x} \{show y}"
  show (Eq x y) = "Eq \{show x} \{show y}"
  show (Add x y) = "Add \{show x} \{show y}"
  show (Min x y) = "Min \{show x} \{show y}"
  show (Select p t f) = "Select \{show p} \{show t} \{show f}"
  show _ = "???"

Show (Terms m n) where
  show Nil = "Nil"
  show (x :: xs) = "\{show x} :: \{show xs}"

Prelude.Eq (Terms m n) where
  Nil == Nil = True
  (x :: xs) == (y :: ys) = x == y && xs == ys
  _ == _ = False

u32 : Nat -> Expr n
u32 = FromLiteral {dtype = U32} . Scalar

mergeNoConflict : Property
mergeNoConflict = fixedProperty $ do
  let (s ** (_, _, terms)) = merge [] []
  case decEq 0 s of
    Yes eq => terms === (rewrite sym eq in [])
    No _ => failure

  let (s ** (fm, fn, terms)) = merge [u32 0] []
  case decEq 1 s of
    Yes eq => do
      fm 0 === (rewrite sym eq in FZ)
      terms === (rewrite sym eq in [u32 0])
    No _ => failure

{-
  let ((s ** (terms, _)), conflict) = Expr.mergeHelper [] [u32 0]
  case decEq 1 s of
    Yes eq => terms === (rewrite sym eq in [u32 0])
    No _ => failure
  conflict === False

  let ((s ** (terms, _)), conflict) = Expr.mergeHelper [u32 0] [u32 0]
  case decEq 0 s of
    Yes eq => terms === (rewrite sym eq in [u32 0])
    No _ => failure
  conflict === False

  let ((s ** (terms, _)), conflict) = Expr.mergeHelper [u32 0, u32 0] [u32 0]
  case decEq 0 s of
    Yes eq => terms === (rewrite sym eq in [u32 0, u32 0])
    No _ => failure
  conflict === False

  let ((s ** (terms, _)), conflict) = Expr.mergeHelper [u32 0] [u32 0, u32 0]
  case decEq 1 s of
    Yes eq => terms === (rewrite sym eq in [u32 0, u32 0])
    No _ => failure
  conflict === False

  let ((s ** (terms, _)), conflict) = Expr.mergeHelper [u32 0, u32 0] [u32 0, u32 0]
  case decEq 0 s of
    Yes eq => terms === (rewrite sym eq in [u32 0, u32 0])
    No _ => failure
  conflict === False

mergeWithConflict : Property
mergeWithConflict = fixedProperty $ do
  let ((s ** (terms, _)), conflict) = Expr.mergeHelper [u32 0] [u32 1]
  case decEq 1 s of
    Yes eq => terms === (rewrite sym eq in [u32 0, u32 1])
    No _ => failure
  conflict === True

  let ((s ** (terms, _)), conflict) = Expr.mergeHelper [u32 0, u32 0] [u32 1]
  case decEq 1 s of
    Yes eq => terms === (rewrite sym eq in [u32 0, u32 0, u32 1])
    No _ => failure
  conflict === True

  let ((s ** (terms, _)), conflict) = Expr.mergeHelper [u32 0, u32 1] [u32 1]
  case decEq 1 s of
    Yes eq => terms === (rewrite sym eq in [u32 0, u32 1, u32 1])
    No _ => failure
  conflict === True

mergeWithIndicesNoConflict : Property
mergeWithIndicesNoConflict = fixedProperty $ do
  let terms = [u32 0, u32 1, Add 0 1]
      ((s ** (merged, _)), conflict) = Expr.mergeHelper terms terms
  case decEq 0 s of
    Yes eq => merged === (rewrite sym eq in terms)
    No _ => failure
  conflict === False

mergeWithIndicesWithConflict : Property
mergeWithIndicesWithConflict = fixedProperty $ do
  let ((s ** (merged, _)), conflict) = Expr.mergeHelper [u32 0, u32 1, Eq 0 1] [u32 0, u32 1, Add 0 1]
  case decEq 1 s of
    Yes eq => merged === (rewrite sym eq in [u32 0, u32 1, Eq 0 1, Add 0 1])
    No _ => failure
  conflict === True

  let ((s ** (merged, _)), conflict) = Expr.mergeHelper [u32 0, u32 1, Eq 0 1] [u32 0, u32 2, Add 0 1]
  case decEq 2 s of
    Yes eq => merged === (rewrite sym eq in [u32 0, u32 1, Eq 0 1, u32 2, Add 0 3])
    No _ => failure
  conflict === True
-}
export
group : Group
group = MkGroup "Compiler.Expr" $ [
      ("merge no conflict", mergeNoConflict)
  ]
{-
      , ("mergeHelper with conflict", mergeWithConflict)
      , ("mergeHelper with indices no conflict", mergeWithIndicesNoConflict)
      , ("mergeHelper with indices with conflict", mergeWithIndicesWithConflict)
  ]
-}