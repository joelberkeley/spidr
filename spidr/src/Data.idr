{--
Copyright (C) 2025  Joel Berkeley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
