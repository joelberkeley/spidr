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
||| This module contains definitions and utilities for datasets.
module Data

import Tensor

||| Observed pairs of data points from feature and target domains. Data sets such as this are
||| commonly used in supervised learning settings.
|||
||| @features The shape of the feature domain.
||| @targets The shape of the target domain.
public export
data Dataset : (0 features : Shape) -> (0 targets : Shape) -> Type where
  MkDataset :
    {s : _} ->
    Tensor (S s :: features) F64 ->
    Tensor (S s :: targets) F64 ->
    Dataset features targets

||| Concatenate two datasets along their leading axis.
export
SemigroupT Ref (Dataset features targets) where
  (MkDataset {s=s} x y) <+> (MkDataset {s=s'} x' y') =
    MkDataset {s=s + S s'} !(concat 0 x x') !(concat 0 y y')
