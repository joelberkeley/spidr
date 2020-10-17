import Tensor

import Data.Vect
import GaussianProcess
import Kernel

-- note having to specify types isn't much of a problem, because they'll usually
--   be inferred from type signatures
two_by_four : Tensor [2, 4] Double
two_by_four = MkTensor [[-1, -2, -3, -4], [1, 2, 3, 4]]

four_by_three : Tensor [4, 3] Double
four_by_three = MkTensor [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

one_by_one_by_two : Tensor [1, 1, 2] Double
one_by_one_by_two = MkTensor [[[2, 1]]]

three_by_one_by_two : Tensor [3, 1, 2] Double
three_by_one_by_two = MkTensor [[[1, 2]], [[3, 4]], [[5, 6]]]

three_by_three : Tensor [3, 3] Double
three_by_three = MkTensor [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

bc_prf : Broadcastable [1, 1, 1] [3, 1, 2]
bc_prf = Extend (One Refl _) (Extend (Eq Refl) (Extend (One Refl _) Empty))

main : IO ()
main = printLn $ two_by_four @@ four_by_three
