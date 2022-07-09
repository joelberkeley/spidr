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
module Unit.Util.TestList

import Util.List

import Utils.Cases
import Utils.Comparison

indicesNil : indices Prelude.Nil === [] 
indicesNil = Refl

indices : Property
indices = property $ do
  xs <- forAll (list (linear 0 10) (constant ()))
  indices (() :: xs) === indices xs ++ [length xs]

enumerate : Property
enumerate = fixedProperty $ do
  enumerate {a=()} [] === []
  enumerate [5] === [(0, 5)]
  enumerate [5, 7, 9] === [(0, 5), (1, 7), (2, 9)]

insertAt : Property
insertAt = fixedProperty $ do
  List.insertAt 0 9 [6, 7, 8] === [9, 6, 7, 8]
  List.insertAt 1 9 [6, 7, 8] === [6, 9, 7, 8]
  List.insertAt 3 9 [6, 7, 8] === [6, 7, 8, 9]
  List.insertAt 0 9 [] === [9]

deleteAt : Property
deleteAt = fixedProperty $ do
  deleteAt 0 [6, 7, 8] === [7, 8]
  deleteAt 1 [6, 7, 8] === [6, 8]
  deleteAt 2 [6, 7, 8] === [6, 7]
  deleteAt 0 [6] === []

replaceAt : Property
replaceAt = fixedProperty $ do
  replaceAt 0 5 [6, 7, 8] === [5, 7, 8]
  replaceAt 1 5 [6, 7, 8] === [6, 5, 8]
  replaceAt 2 5 [6, 7, 8] === [6, 7, 5]
  replaceAt 0 5 [6] === [5]

namespace Many
  deleteAtForNoIndices : (xs : List a) -> deleteAt [] xs = xs
  deleteAtForNoIndices [] = Refl
  deleteAtForNoIndices (_ :: _) = Refl

  deleteAtHead : (x : a) -> (xs : List a) -> deleteAt [0] (x :: xs) = xs
  deleteAtHead _ [] = Refl
  deleteAtHead _ (_ :: _) = Refl

  deleteAtLater : (x, y : a) -> (xs : List a) -> deleteAt [1] (x :: y :: xs) = (x :: xs)
  deleteAtLater _ _ [] = Refl
  deleteAtLater _ _ (_ :: _) = Refl

  deleteAtHeadAndLater :
    (x, y, z : a) -> (xs : List a) -> deleteAt [0, 2] (x :: y :: z :: xs) = (y :: xs)
  deleteAtHeadAndLater _ _ _ [] = Refl
  deleteAtHeadAndLater _ _ _ (_ :: _) = Refl

export
group : Group
group = MkGroup "Util" $ [
      ("indices", indices)
    , ("enumerate", enumerate)
    , ("insertAt", insertAt)
    , ("deleteAt", deleteAt)
  ]
