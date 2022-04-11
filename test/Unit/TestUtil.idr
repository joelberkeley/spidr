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

import Util

import Utils.Property

namespace Vect
  export
  test_range : Property
  test_range = property $ do
    Vect.range 0 === []
    Vect.range 1 === [0]
    Vect.range 3 === [0, 1, 2]

  export
  test_enumerate : Property
  test_enumerate = property $ do
    Vect.enumerate {a=()} [] === []
    Vect.enumerate [5] === [(0, 5)]
    Vect.enumerate [5, 7, 9] === [(0, 5), (1, 7), (2, 9)]

namespace List
  export
  test_range : Property
  test_range = property $ do
    List.range 0 === []
    List.range 1 === [0]
    List.range 3 === [0, 1, 2]

  export
  test_enumerate : Property
  test_enumerate = property $ do
    List.enumerate {a=()} [] === []
    List.enumerate [5] === [(0, 5)]
    List.enumerate [5, 7, 9] === [(0, 5), (1, 7), (2, 9)]

  export
  test_insertAt : Property
  test_insertAt = property $ do
    List.insertAt 0 9 [6, 7, 8] === [9, 6, 7, 8]
    List.insertAt 1 9 [6, 7, 8] === [6, 9, 7, 8]
    List.insertAt 3 9 [6, 7, 8] === [6, 7, 8, 9]
    List.insertAt 0 9 [] === [9]

  export
  test_deleteAt : Property
  test_deleteAt = property $ do
    List.deleteAt 0 [6, 7, 8] === [7, 8]
    List.deleteAt 1 [6, 7, 8] === [6, 8]
    List.deleteAt 2 [6, 7, 8] === [6, 7]
    List.deleteAt 0 [6] === []

  export
  test_replaceAt : Property
  test_replaceAt = property $ do
    List.replaceAt 0 5 [6, 7, 8] === [5, 7, 8]
    List.replaceAt 1 5 [6, 7, 8] === [6, 5, 8]
    List.replaceAt 2 5 [6, 7, 8] === [6, 7, 5]
    List.replaceAt 0 5 [6] === [5]

export
root : Group
root = MkGroup "Util" $ [
      ("Vect.range", Vect.test_range)
    , ("Vect.enumerate", Vect.test_enumerate)
    , ("List.range", List.test_range)
    , ("List.enumerate", List.test_enumerate)
    , ("insertAt", test_insertAt)
    , ("deleteAt", test_deleteAt)
  ]
