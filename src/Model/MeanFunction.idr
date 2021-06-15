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
||| This module contains definitions for and implementations of mean functions, particularly for
||| use in Gaussian processes.
module Model.MeanFunction

import Tensor

||| A `MeanFunction` maps a point in feature space to the mean value of a corresponding
||| distribution in target space.
public export
MeanFunction : (features : Shape) -> Type
MeanFunction features = {sm : Nat} -> Tensor (sm :: features) Double -> Tensor [sm] Double

||| A mean function where the mean is zero in all target dimensions.
export
zero : MeanFunction features
