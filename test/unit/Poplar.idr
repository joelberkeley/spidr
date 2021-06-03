import System

import Poplar

infix 0 ==?

(==?) : Eq ty => ty -> ty -> IO ()
x ==? y = if x == y then exitSuccess else exitFailure

test_add : IO ()
test_add = add 2 3 ==? 5
