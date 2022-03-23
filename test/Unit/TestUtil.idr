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
import Utils

namespace Vect
  export
  test_range : IO ()
  test_range = do
    assert "Vect range 0" $ Vect.range 0 == []
    assert "Vect range 1" $ Vect.range 1 == [0]
    assert "Vect range 3" $ Vect.range 3 == [0, 1, 2]

  export
  test_enumerate : IO ()
  test_enumerate = do
    assert "Vect enumerate 0" $ Vect.enumerate [] == []
    assert "Vect enumerate 1" $ Vect.enumerate [5] == [(0, 5)]
    assert "Vect enumerate 3" $ Vect.enumerate [5, 7, 9] == [(0, 5), (1, 7), (2, 9)]

namespace List
  export
  test_range : IO ()
  test_range = do
    assert "List range 0" $ List.range 0 == []
    assert "List range 1" $ List.range 1 == [0]
    assert "List range 3" $ List.range 3 == [0, 1, 2]

  export
  test_enumerate : IO ()
  test_enumerate = do
    assert "List enumerate 0" $ List.enumerate [] == []
    assert "List enumerate 1" $ List.enumerate [5] == [(0, 5)]
    assert "List enumerate 3" $ List.enumerate [5, 7, 9] == [(0, 5), (1, 7), (2, 9)]

  export
  test_insertAt : IO ()
  test_insertAt = do
    assert "insertAt can insert at front" $ (List.insertAt 0 9 [6, 7, 8]) == [9, 6, 7, 8]
    assert "insertAt can insert in middle" $ (List.insertAt 1 9 [6, 7, 8]) == [6, 9, 7, 8]
    assert "insertAt can insert at end" $ (List.insertAt 3 9 [6, 7, 8]) == [6, 7, 8, 9]
    assert "insertAt for empty list" $ (List.insertAt 0 9 []) == [9]

  export
  test_deleteAt : IO ()
  test_deleteAt = do
    assert "deleteAt can delete from front" $ (List.deleteAt 0 [6, 7, 8]) == [7, 8]
    assert "deleteAt can delete from middle" $ (List.deleteAt 1 [6, 7, 8]) == [6, 8]
    assert "deleteAt can delete from end" $ (List.deleteAt 2 [6, 7, 8]) == [6, 7]
    assert "deleteAt for length one list" $ (List.deleteAt 0 [6]) == []

  export
  test_replaceAt : IO ()
  test_replaceAt = do
    assert "replaceAt can replace at front" $ (List.replaceAt 0 5 [6, 7, 8]) == [5, 7, 8]
    assert "replaceAt can replace at middle" $ (List.replaceAt 1 5 [6, 7, 8]) == [6, 5, 8]
    assert "replaceAt can replace at end" $ (List.replaceAt 2 5 [6, 7, 8]) == [6, 7, 5]
    assert "replaceAt for length one list" $ (List.replaceAt 0 5 [6]) == [5]

export
test : IO ()
test = do
  Vect.test_range
  Vect.test_enumerate
  List.test_range
  List.test_enumerate
  test_insertAt
  test_deleteAt
