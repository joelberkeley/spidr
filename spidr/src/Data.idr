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
||| Dataset types.
module Data

import Tensor

||| Observed pairs of data points from feature and target domains. Data sets such as this are
||| commonly used in supervised learning settings.
|||
||| @features The shape of the feature domain.
||| @targets The shape of the target domain.
public export
data Dataset : (features, targets : Shape) -> Type where
  MkDataset : {s : Nat} ->
              Tensor (S s :: features) F64 -@
              (Tensor (S s :: targets) F64 -@
              Dataset features targets)

public export
size : Dataset f t -> Nat
size (MkDataset {s} f t) = S s

||| The feature data
export
(.features) : (1 dataset : Dataset features targets) -> Tensor (size dataset :: features) F64
(MkDataset fs ts).features = discarding ts fs

||| The target data
export
(.targets) : (1 dataset : Dataset features targets) -> Tensor (size dataset :: targets) F64
(MkDataset fs ts).targets = discarding fs ts

||| Concatenate two datasets along their leading axis.
export
concat : Dataset features targets -@ (Dataset features targets -@ Dataset features targets)
concat (MkDataset {s = s} x y) (MkDataset {s = s'} x' y') =
  MkDataset {s = s + S s'} (concat 0 x x') (concat 0 y y')

export
Copy (Dataset f t) where
  copy (MkDataset f t) = do
    MkBang f <- copy f
    MkBang t <- copy t
    pure $ MkBang $ MkDataset f t
  discard (MkDataset f t) =
    let () = discard f
        () = discard t
     in ()
