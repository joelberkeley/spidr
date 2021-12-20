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

-- test_XlaBuilder_name : IO ()
-- test_XlaBuilder_name = do assert $ name (mkXlaBuilder "foo") == "foo"
--                           assert $ name (mkXlaBuilder "-1") == "-1"
--                           assert $ name (mkXlaBuilder "") == ""

-- test_add : IO ()
-- test_add = do let b = mkXlaBuilder ""
--                   x = const b 1
--                   y = const b 2
--               sum <- eval {shape=[]} (x + y)
--               assert $ sum == (the Int $ 3)
--               delete b
--               delete x
--               delete y
--               let b = mkXlaBuilder ""
--                   x = const b 3
--                   y = const b (-7)
--               sum <- eval {shape=[]} (x + y)
--               assert $ sum == (the Int $ -4)
--               delete b
--               delete x
--               delete y

-- test_opToString : IO ()
-- test_opToString = do let builder = mkXlaBuilder "foo"
--                      let one = const builder 1
--                      assert $ opToString builder one == "constant, shape=[], metadata={:0}"
--                      delete one
--                      delete builder

test_add : IO ()
test_add = do let b = mkXlaBuilder ""
              putStrLn "test_add ... construct x"
              x <- XlaBuilder.const {shape=[2, 3]} {dtype=Int} b [[1, 15, 5], [-1, 7, 6]]
              -- x <- XlaBuilder.const {shape=[3]} {dtype=Int} b [1, 15, 5]
              putStrLn "test_add ... construct y"
              y <- XlaBuilder.const {shape=[2, 3]} {dtype=Int} b [[11, 5, 7], [-3, -4, 0]]
              -- y <- XlaBuilder.const {shape=[3]} {dtype=Int} b [11, 5, 7]
              putStrLn "test_add ... add and evaluate (x + y)"
              sum <- eval {shape=[2, 3]} {dtype=Int} (x + y)
              putStrLn "test_add ... assert result"
              assert $ sum == [[12, 20, 11], [-4, 3, 6]]
              putStrLn "test_add ... done"
              -- assert $ sum == [12, 20, 12]

test : IO ()
test = do test_add
