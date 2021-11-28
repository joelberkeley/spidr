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
import XLA

infix 0 ==?

assert : Bool -> IO ()
assert x = if x then printLn "PASS" else printLn "FAIL"

test_add : IO ()
test_add = do let two = mkBignum
                  three = mkBignum
              assign two 2
              assign three 3
              assert $ three > two
              delete two
              delete three
