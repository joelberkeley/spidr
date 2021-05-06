%foreign "C:add,libffi"
add : Int -> Int -> Int

main : IO()
main = printLn $ add 1 2
