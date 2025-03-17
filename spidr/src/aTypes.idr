{--
Copyright 2025 Joel Berkeley

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
||| Common library types.
module Types

import public Data.Nat
import public Data.Fin
import public Data.Vect
import public Data.Quantifiers.All

export
data Cache : Vect n (Nat, Type) -> Type -> Type where
  MkCache : HVect (snd <$> types) -> Cache types


