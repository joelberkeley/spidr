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
