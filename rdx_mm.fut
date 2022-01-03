let radix_sort_step [ n ] ( xs : [ n ] u32 ) ( b : i32 ) : [ n ] u32 =
  let bits = map (\ x -> ( i32 . u32 ( x >> u32 . i32 b ) ) & 1) xs
  let bits_neg = map (1 -) bits
  let offs = reduce (+) 0 bits_neg
  let idxs0 = map2 (*) bits_neg ( scan (+) 0 bits_neg )
  let idxs1 = map2 (*) bits ( map (+ offs ) ( scan (+) 0 bits ) )
  let idxs2 = map2 (+) idxs0 idxs1
  let idxs = map (\ x - >x -1) idxs2
  let xs = scatter ( copy xs ) ( map i64 . i32 idxs ) xs
  in xs

let radix_sort [ n ] ( xs : [ n ] u32 ) : [ n ] u32 =
  loop xs for i < 32 do radix_sort_step xs i
