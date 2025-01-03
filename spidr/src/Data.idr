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

%prefix_record_projections off

||| Observed pairs of data points from feature and target domains. Data sets such as this are
||| commonly used in supervised learning settings.
|||
||| @features The shape of the feature domain.
||| @targets The shape of the target domain.
public export
record Dataset (0 featureShape, targetShape : Shape) where
  constructor MkDataset
  {s : Nat}

  ||| The feature data
  features : Tensor (S s :: featureShape) F64

  ||| The target data
  targets : Tensor (S s :: targetShape) F64

%prefix_record_projections on

||| Concatenate two datasets along their leading axis.
export
concat : Dataset features targets -> Dataset features targets -> Dataset features targets
concat (MkDataset {s = s} x y) (MkDataset {s = s'} x' y') =
  MkDataset {s = s + S s'} (concat 0 x x') (concat 0 y y')

export
Taggable (Dataset f t) where
  tag (MkDataset f t) = [| MkDataset (tag f) (tag t) |]
