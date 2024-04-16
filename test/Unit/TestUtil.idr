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
module Unit.TestUtil

import Decidable.Equality

import Util

import Utils.Cases
import Utils.Comparison

namespace Vect
  export
  range : Property
  range = fixedProperty $ do
    Vect.range 0 === []
    Vect.range 1 === [0]
    Vect.range 3 === [0, 1, 2]

  export
  enumerate : Property
  enumerate = fixedProperty $ do
    Vect.enumerate {a=()} [] === []
    Vect.enumerate [5] === [(0, 5)]
    Vect.enumerate [5, 7, 9] === [(0, 5), (1, 7), (2, 9)]

namespace List
  export
  range : Property
  range = fixedProperty $ do
    List.range 0 === []
    List.range 1 === [0]
    List.range 3 === [0, 1, 2]

  export
  enumerate : Property
  enumerate = fixedProperty $ do
    List.enumerate {a=()} [] === []
    List.enumerate [5] === [(0, 5)]
    List.enumerate [5, 7, 9] === [(0, 5), (1, 7), (2, 9)]

  uniqueEmpty : Eq a => unique (the (List a) []) = True
  uniqueEmpty = Refl

  multiIndex : HList [
        multiIndex {inBounds = []} [] (the (List Nat) []) = the (List Nat) []
      , multiIndex [] [the Nat 0] = the (List Nat) []
      , multiIndex [0] [the Nat 1] = the (List Nat) [1]
      , multiIndex [] [the Nat 2, 3] = the (List Nat) []
      , multiIndex [0] [the Nat 2, 3] = the (List Nat) [2]
      , multiIndex [1] [the Nat 2, 3] = the (List Nat) [3]
      , multiIndex [0, 0] [the Nat 2, 3] = the (List Nat) [2, 2]
      , multiIndex [0, 1] [the Nat 2, 3] = the (List Nat) [2, 3]
      , multiIndex [1, 0] [the Nat 2, 3] = the (List Nat) [3, 2]
      , multiIndex [1, 1] [the Nat 2, 3] = the (List Nat) [3, 3]
      , multiIndex [] [the Nat 3, 4, 5] = the (List Nat) []
      , multiIndex [0] [the Nat 3, 4, 5] = the (List Nat) [3]
      , multiIndex [1] [the Nat 3, 4, 5] = the (List Nat) [4]
      , multiIndex [2] [the Nat 3, 4, 5] = the (List Nat) [5]
      , multiIndex [1, 2, 0] [the Nat 3, 4, 5] = the (List Nat) [4, 5, 3]
    ]
  multiIndex = %search

  deleteAt : Property
  deleteAt = property $ do
    ys <- forAll $ list (constant 0 10) (nat (constant 0 100))
    y <- forAll $ nat (constant 0 100)

    deleteAt [] ys === ys

    let xs : List Nat
        xs = y :: ys

    (i0 ** p0) <- forAll $ index xs
    (i1 ** p1) <- forAll $ index xs

    let x0 = index i0 xs
        x1 = index i1 xs

    deleteAt [i0] xs === deleteAt i0 xs
    deleteAt [i0, i0] xs === deleteAt i0 xs
    deleteAt [i0, i1] xs === deleteAt [i1, i0] xs

    let inBoundsDelete : {i : _} ->
                         (prf : InBounds j xs) ->
                         LT i j ->
                         InBounds i (deleteAt {prf} j xs)
        inBoundsDelete {i = Z}   (InLater _)  (LTESucc LTEZero) = InFirst
        inBoundsDelete {i = S k} (InLater ib) (LTESucc lt)      = InLater (inBoundsDelete ib lt)

        inj : Not (Prelude.S m = Prelude.S n) -> Not (m = n)
        inj ne refl = absurd $ ne $ cong S refl

        notEqLteIsLt : {i, j : Nat} -> Not (i = j) -> LTE i j -> LTE (S i) j
        notEqLteIsLt {i = Z}   {j = Z}   ne _             = absurd (ne Refl)
        notEqLteIsLt {i = Z}   {j = S _} _  LTEZero       = LTESucc LTEZero
        notEqLteIsLt {i = S _} {j = Z}   _  (LTESucc lte) impossible
        notEqLteIsLt {i = S _} {j = S _} ne (LTESucc lte) = LTESucc (notEqLteIsLt (inj ne) lte)

    case decEq i0 i1 of
      Yes _ => pure ()  -- tested above
      No ne => case isLTE i0 i1 of
        Yes lte   => let 0 prf = inBoundsDelete p1 (notEqLteIsLt ne lte)
                      in deleteAt [i0, i1] xs === deleteAt {prf} i0 (deleteAt i1 xs)
        No notLte => let 0 prf = inBoundsDelete p0 (notLTEImpliesGT notLte)
                      in deleteAt [i0, i1] xs === deleteAt {prf} i1 (deleteAt i0 xs)

  repeatedNotLT : Sorted LT [x, x] -> Void
  repeatedNotLT SNil impossible
  repeatedNotLT SOne impossible
  repeatedNotLT (SCons _ ok _) = succNotLTEpred ok

  repeatedLaterNotLT : Sorted LT [x, y, y] -> Void
  repeatedLaterNotLT SNil impossible
  repeatedLaterNotLT SOne impossible
  repeatedLaterNotLT (SCons _ _ tail) = repeatedNotLT tail

  ltNotReflexive : (x, y : Nat) -> LT x y -> LT y x -> Void
  ltNotReflexive 0 0 LTEZero _ impossible
  ltNotReflexive 0 0 (LTESucc _) _ impossible
  ltNotReflexive 0 (S _) (LTESucc _) LTEZero impossible
  ltNotReflexive 0 (S _) (LTESucc _) (LTESucc _) impossible
  ltNotReflexive (S _) 0 LTEZero _ impossible
  ltNotReflexive (S _) 0 (LTESucc _) _ impossible
  ltNotReflexive (S x) (S y) (LTESucc xlty) (LTESucc yltx) = ltNotReflexive x y xlty yltx

  repeatedDispersedNotLT : Sorted LT [y, x, y] -> Void
  repeatedDispersedNotLT (SCons y yltx (SCons x xlty SOne)) = ltNotReflexive x y xlty yltx

  increasingLT : (x : Nat) -> Sorted LT [x, S x, S (S x)]
  increasingLT x = SCons x (reflexive {ty=Nat}) (SCons (S x) (reflexive {ty=Nat}) SOne)

  succNotLT : (x : Nat) -> LT (S x) x -> Void
  succNotLT 0 LTEZero impossible
  succNotLT 0 (LTESucc _) impossible
  succNotLT (S x) (LTESucc lte) = succNotLT x lte

  decreasingNotLT : (x : Nat) -> Sorted LT (S x :: x :: xs)  -> Void
  decreasingNotLT _ SNil impossible
  decreasingNotLT _ SOne impossible
  decreasingNotLT x (SCons (S x) ok _) = succNotLT x ok

export
group : Group
group = MkGroup "Util" $ [
      ("Vect.range", Vect.range)
    , ("Vect.enumerate", Vect.enumerate)
    , ("List.range", List.range)
    , ("List.enumerate", List.enumerate)
  ]
