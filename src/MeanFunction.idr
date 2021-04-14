module MeanFunction

import Data.Vect
import Tensor

public export
MeanFunction : (features : Shape) -> Type
MeanFunction features = {sm : Nat} -> Tensor (sm :: features) Double -> Tensor [sm] Double

-- todo is it possible to implement this without using `sm`? If so, we can make `sm` erased
export
zero : MeanFunction features
zero {sm} (MkTensor x) = replicate {over=[sm]} $ MkTensor 0
