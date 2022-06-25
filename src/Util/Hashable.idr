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
module Util.Hashable

import public Data.Hashable

import Util
import Types

export
Hashable Double where
  hashWithSalt s x = s `combine` hash (show x)

export
(Hashable a, Hashable b) => Hashable (a, b) where
    hashWithSalt s (a, b) = s `hashWithSalt` a `hashWithSalt` b

export
(Hashable a, Hashable b) => Hashable (Either a b) where
  hashWithSalt salt (Left l) = salt `hashWithSalt` 0 `hashWithSalt` l
  hashWithSalt salt (Right r) = salt `hashWithSalt` 1 `hashWithSalt` r
