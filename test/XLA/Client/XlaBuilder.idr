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
test_add = do let b = mkXlaBuilder ""
              x <- const {shape=[2, 3]} {dtype=Int} b [[1, 15, 5], [-1, 7, 6]]
              y <- const {shape=[2, 3]} {dtype=Int} b [[11, 5, 7], [-3, -4, 0]]
              sum <- eval {shape=[2, 3]} {dtype=Int} (x + y)
              assert $ sum == [[12, 20, 12], [-4, 3, 6]]
              delete x
              delete y
              x <- const {shape=[]} {dtype=Int} b 3
              y <- const {shape=[]} {dtype=Int} b (-7)
              sum <- eval {shape=[]} {dtype=Int} (x + y)
              assert $ sum == -4

test_opToString : IO ()
test_opToString = do let builder = mkXlaBuilder "foo"
                     one <- const {shape=[1]} {dtype=Int} builder [1]
                     assert $ opToString builder one == "constant, shape=[1], metadata={:0}"
                     delete one
                     delete builder

test : IO ()
test = do test_add
