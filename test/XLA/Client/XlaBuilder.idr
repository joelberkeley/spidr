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
import XLA.XlaData
import XLA.Literal

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
              x <- const {shape=[3, 1]} {dtype=Double} b [[1.8], [1.3], [4.0]]
              y <- const {shape=[3, 1]} {dtype=Double} b [[-3.3], [0.0], [0.3]]
              sum <- eval {shape=[3, 1]} {dtype=Double} (x + y)
              assert $ abs (index 0 (index 0 sum) - (-1.5)) < 0.000001
              assert $ abs (index 0 (index 1 sum) - 1.3) < 0.000001
              assert $ abs (index 0 (index 2 sum) - 4.3) < 0.000001
              delete x
              delete y
              x <- const {shape=[]} {dtype=Int} b 3
              y <- const {shape=[]} {dtype=Int} b (-7)
              sum <- eval {shape=[]} {dtype=Int} (x + y)
              assert $ sum == -4
              delete x
              delete y
              delete b

test_opToString : IO ()
test_opToString = do let b = mkXlaBuilder ""
                     x <- const {shape=[]} {dtype=Int} b 1
                     assert $ opToString b x == "constant, shape=[], metadata={:0}"
                     delete x
                     x <- const {shape=[3]} {dtype=Int} b [1, 2, 3]
                     assert $ opToString b x == "constant, shape=[3], metadata={:0}"
                     delete x
                     delete b

test : IO ()
test = do test_XlaBuilder_name
          test_add
          test_opToString
