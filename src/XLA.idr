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
||| This is the root module for the Idris API to XLA.
module XLA

import public XLA.Client as XLA
import public XLA.FFI as XLA
import public XLA.Literal as XLA
import public XLA.Shape as XLA
import public XLA.ShapeUtil as XLA
import public XLA.XlaData as XLA
