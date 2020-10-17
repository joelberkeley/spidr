module MeanFunction

import Data.Vect
import Tensor

public export
MeanFunction : (samples : Nat) -> (features : Shape) -> Type
-- MeanFunction features = {samples : Nat} -> Tensor (samples :: features) Double -> Tensor [samples] Double
MeanFunction samples features = Tensor (samples :: features) Double -> Tensor [samples] Double
