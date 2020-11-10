module MeanFunction

import Data.Vect
import Tensor

public export
MeanFunction : (features : Shape) -> Type
MeanFunction features = {samples : Nat} -> Tensor (samples :: features) Double -> Tensor [samples] Double
-- MeanFunction samples features = Tensor (samples :: features) Double -> Tensor [samples] Double

export
zero : MeanFunction features
zero {samples} (MkTensor x) = replicate [samples] $ MkTensor 0
