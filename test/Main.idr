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
module Main

import Unit.TestTensor
import Unit.TestXLA

import Utils

main : IO ()
main = do
    test_sufficientlyEq
    test_sufficientlyEqEach

    test_const_eval
    test_toString
    test_broadcast
    test_elementwise_equality
    test_elementwise_inequality
    test_comparison
    test_add
    test_subtract
    test_elementwise_multiplication
    test_constant_multiplication
    test_elementwise_division
    test_elementwise_and
    test_elementwise_or
    test_elementwise_notEach
    test_absE
    test_negate

    test_parameter_addition

    putStrLn "Tests passed"
