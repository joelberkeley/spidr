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

test_array_put_get : IO ()
test_array_put_get = do ptr <- putArray [2, 3] Int [[11, 5, 7], [-3, 4, 0]]
                        primIO $ test_put ptr

test : IO ()
test = do test_array_put_get
