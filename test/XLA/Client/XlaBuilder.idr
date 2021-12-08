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

infix 0 ==?

assert : Bool -> IO ()
assert x = printLn $ the String $ if x then "PASS" else "FAIL"

test_add : IO ()
test_add = do let builder = mkXlaBuilder
              let one = const builder 1
              let two = const builder 2
              let three = one + two
              putStrLn (opToString builder (one + two == three))
              delete one
              delete two
              delete three
              delete builder
