
let radix_sort_step_nn [n] (xs : [n]u32) (b : i32) : [n]u32 =
  let bits = map (u32.get_bit b) xs   -- [1,0,1,0,1,0]
  let bits_neg = map (1-) bits        -- [0,1,0,1,0,1]


  let temp1 = (scan (+) 0 bits)       -- [1,1,2,2,3,3]
  let temp2 = (scan (+) 0 bits_neg)   -- [0,1,1,2,2,3]
  let offs = temp2[n-1]               -- 3

  let idxs = map3 (\b t1 t2 -> if b == 1 
                               then i64.i32 (t1-1+offs)
                               else i64.i32 (t2-1)) bits temp1 temp2    -- [3,0,4,1,5,2]

  in scatter (copy xs) idxs xs -- sorted
  



let radix_sort_nn [n] (xs : [n]u32) : [n]u32 =
  loop xs for i < 32 do radix_sort_step_nn xs i
