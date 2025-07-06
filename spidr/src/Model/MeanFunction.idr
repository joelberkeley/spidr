{--
Copyright (C) 2021  Joel Berkeley

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
||| This module contains definitions for and implementations of mean functions, particularly for
||| use in Gaussian processes.
module Model.MeanFunction

import Tensor

||| A `MeanFunction` maps a point in feature space to the mean value of a corresponding
||| distribution in target space.
|||
||| @features The shape of the feature domain.
public export 0
MeanFunction : (0 features : Shape) -> Type
MeanFunction features = {sm : _} -> Tensor (sm :: features) F64 -> Tag $ Tensor [sm] F64

||| A mean function where the mean is zero in all target dimensions.
export
zero : MeanFunction features
zero _ = pure $ fill 0
