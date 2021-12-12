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
test_XlaBuilder_name = do x <- name (mkXlaBuilder "foo"); assert $ x == "foo"
                          x <- name (mkXlaBuilder "-1 "); assert $ x == "-1 "
                          x <- name (mkXlaBuilder ""); assert $ x == ""

test_add : IO ()
test_add = do let b = mkXlaBuilder ""
              sum <- eval_int (const b 1 + const b 2); assert $ sum == 3
              let b = mkXlaBuilder ""
              sum <- eval_int (const b 3 + const b (-7)); assert $ sum == -4

test_opToString : IO ()
test_opToString = do s <- opToString $ const (mkXlaBuilder "foo") 1
                     assert $ s == "constant, shape=[], metadata={:0}"

test : IO ()
test = do pure ()
          test_XlaBuilder_name
          test_add
          test_opToString
