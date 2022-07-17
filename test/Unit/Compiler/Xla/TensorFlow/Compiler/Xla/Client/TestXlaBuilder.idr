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
module Unit.Compiler.Xla.TensorFlow.Compiler.Xla.Client.TestXlaBuilder

import Compiler.Computation
import Compiler.LiteralRW
import Compiler.Xla.TensorFlow.Compiler.Xla.Literal
import Compiler.Xla.TensorFlow.Compiler.Xla.Shape
import Compiler.Xla.TensorFlow.Compiler.Xla.XlaData
import Compiler.Xla.TensorFlow.Compiler.Xla.Client.XlaBuilder
import Literal

import Utils.Comparison
import Utils.Cases

export
xlaOpShapeDebugString : Property
xlaOpShapeDebugString = fixedProperty $ do
  let str : String = unsafePerformIO $ do
        lit <- write {dtype=S32} [[0, 1, 2], [3, 4, 5]]
        builder <- mkXlaBuilder ""
        op <- constantLiteral builder lit
        shape <- getShape builder op
        pure (debugString shape)

  str ===
    "element_type: S32\n" ++
    "dimensions: 2\n" ++
    "dimensions: 3\n" ++
    "layout {\n" ++
    "  minor_to_major: 1\n" ++
    "  minor_to_major: 0\n" ++
    "  format: DENSE\n" ++
    "}\n" ++
    "is_dynamic_dimension: false\n" ++
    "is_dynamic_dimension: false\n"

export covering
group : Group
group = MkGroup "Debug" $ [
      ("XlaOp shape debug string", xlaOpShapeDebugString)
  ]
