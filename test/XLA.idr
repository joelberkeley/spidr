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

(==?) : Eq ty => ty -> ty -> IO ()
x ==? y = printLn $ if x == y then "PASS" else "FAIL"

test_add : IO ()
test_add = do let two = mkScalar 2
                  three = mkScalar 3
                  five = add two three
              cast five ==? 5.0
              delScalar two
              delScalar three
              delScalar five
