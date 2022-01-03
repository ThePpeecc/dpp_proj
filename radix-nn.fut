
let radix_sort_step_nn [n] (xs : [n]u32) (b : i32) : [n]u32 =
  let bits = map (\x -> (i32.u32 (x >> u32.i32 b)) & 1) xs -- [1,0,1,0,1,0]
  let bits_neg = map (1-) bits -- [0,1,0,1,0,1]
  let offs = reduce (+) 0 bits_neg -- 3


  let temp1 = (scan (+) 0 bits) -- [1,1,2,2,3,3]
  let temp2 = (scan (+) 0 bits_neg) -- [0,1,1,2,2,3]
  let temp_off = map (+offs) temp1 -- [4,4,5,5,6,6] 

  let idxs0 = map2 (*) bits_neg temp2 -- [0,1,0,2,0,3]
  let idxs1 = map2 (*) bits temp_off -- [4,0,5,0,6,0]
 
  let idxs2 = map2 (+) idxs0 idxs1 -- [4,1,5,2,6,3]
  let idxs = map (\x -> x -1) idxs2 -- [3,0,4,1,5,2]


  in scatter (copy xs) (map i64.i32 idxs) xs -- sorted
  



let radix_sort_nn [n] (xs : [n]u32) : [n]u32 =
  loop xs for i < 32 do radix_sort_step_nn xs i
