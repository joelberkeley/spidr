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

import Types
import Util

public export
data Graph =
  Operation String (List Graph) Shape String
  | Leaf String Bits64 Shape String

export covering
Eq Graph where
  Operation lname largs lshape ltype == Operation rname rargs rshape rtype =
    lname == rname && largs == rargs && lshape == rshape && ltype == rtype
  Leaf lname lhash lshape ltype == Leaf rname rhash rshape rtype =
    lname == rname && lhash == rhash && lshape == rshape && ltype == rtype
  _ == _ = False

export covering
Show Graph where
  show (Operation name args shape type) = "\{type}\{show shape} \{name}: \{show args}"
  show (Leaf name hash shape type) = "\{type}\{show shape} \{name}"
