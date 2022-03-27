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
  hashWithSalt s x =
    let hash : Bits64 =
          if x /= x then 111111 else
          if x == 1 / 0 then 222222 else
          if x == -1 / 0 then -222222 else
          cast x
     in s `combine` hash

export
hashWithSalt : Hashable a => {shape : _} -> Bits64 -> Array shape a -> Bits64
hashWithSalt {shape=[]} salt x = hashWithSalt salt x
hashWithSalt {shape=(0 :: _)} salt [] = Data.Hashable.hashWithSalt salt 0
hashWithSalt {shape=(S d :: ds)} salt (x :: xs) =
  hashWithSalt {shape=(d :: ds)} (
    hashWithSalt {shape=ds} (
      Data.Hashable.hashWithSalt salt 1
    ) x
  ) xs
