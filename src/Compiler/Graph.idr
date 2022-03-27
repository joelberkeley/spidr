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
module Compiler.Graph

import Data.Stream
import Types
import Util

||| A `Graph` represents the graph computation of a tensor value. It is equivalent to the
||| computation graph used at runtime, but might not be an exact representation. Specifically, given
||| two `Graph`s gx and gy that compute tensors x and y respectively, if gx is equal to gy, then x
||| is equal to y, but the computations used to compute x and y may be different.
public export
data Graph : Type where
  ||| Represents a function application.
  |||
  ||| @name The name of the operation.
  ||| @arguments The arguments the operation is called on.
  ||| @shape The shape of the resulting tensor.
  ||| @type A string representation of the data type of the resulting tensor.
  Operation :
    (name : String) ->
    (arguments : List Graph) ->
    (shape : Shape) ->
    (type : String) ->
    Graph

  ||| Represents a tensor value. This tensor can have a concrete value, or correspond to a
  ||| function argument.
  |||
  ||| @name The name of the method of instantiating the tensor.
  ||| @id_ An identifier to differentiate this tensor from other tensors.
  ||| @shape The shape of this tensor.
  ||| @type A string representation of the data type of the tensor.
  Leaf : (name : String) -> (id_ : Bits64) -> (shape : Shape) -> (type : String) -> Graph

export covering
Eq Graph where
  Operation lname largs lshape ltype == Operation rname rargs rshape rtype =
    lname == rname && largs == rargs && lshape == rshape && ltype == rtype
  Leaf lname lhash lshape ltype == Leaf rname rhash rshape rtype =
    lname == rname && lhash == rhash && lshape == rshape && ltype == rtype
  _ == _ = False

export covering
Show Graph where
  show xs = impl 0 xs where
    impl : Nat -> Graph -> String
    impl depth =
      let indent = pack $ take (2 * depth) (repeat ' ')
       in \case
            Operation name args shape type =>
              let init = indent ++ "\{type}\{show shape} \{name}"
               in foldl (\acc, g => acc ++ "\n" ++ impl (S depth) g) init args
            Leaf name hash shape type => indent ++ "\{type}\{show shape} \{name}"
