{--
Copyright 2021 Joel Berkeley

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
import XLA.Client.XlaBuilder

import Types

assert : Bool -> IO ()
assert x = putStrLn $ if x then "PASS" else "FAIL"

test_XlaBuilder_name : IO ()
test_XlaBuilder_name = do assert $ name (mkXlaBuilder "foo") == "foo"
                          assert $ name (mkXlaBuilder "-1") == "-1"
                          assert $ name (mkXlaBuilder "") == ""

test_add : IO ()
test_add = do let builder = mkXlaBuilder "foo"
              let one = const builder 1
              let two = const builder 2
              let minus_seven = const builder (-7)
              assert $ eval_int builder (one + two) == 3
              assert $ eval_int builder (two + minus_seven) == -5
              delete one
              delete two
              delete builder

test_opToString : IO ()
test_opToString = do let builder = mkXlaBuilder "foo"
                     let one = const builder 1
                     assert $ opToString builder one == "constant, shape=[], metadata={:0}"
                     delete one
                     delete builder

test : IO ()
test = do test_add
          test_XlaBuilder_name
          test_opToString
