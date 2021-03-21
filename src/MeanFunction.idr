module MeanFunction

import Data.Vect
import Tensor

public export
MeanFunction : (features : Shape) -> Type
MeanFunction features = {samples : Nat} -> Tensor (samples :: features) Double -> Tensor [samples] Double

-- todo is it possible to implement this without using `samples`? If so, we can make samples erased
export
zero : MeanFunction features
zero {samples} (MkTensor x) = replicate [samples] $ MkTensor 0
