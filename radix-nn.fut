
let radix_sort_step_nn [n] (xs : [n]u32) (b : u32) (bits : u32) : [n]u32 =
  let bits_len = i64.u32 (1<<bits)
  let get_bits x = x >> (b*bits) & ((1<<bits)-1)
  let pairwise op arr1 arr2 = map2 (op) arr1 arr2
  let bins = xs |> map get_bits
  let flags = tabulate_2d n bits_len (\i j -> 
                                 if i64.u32 bins[i] == j then 1
                                 else 0)
  let offsets = scan (pairwise (+)) (replicate bits_len 0) flags

  let get_last xs = last xs

  let offs = concat [0] (scan (+) 0 (get_last offsets))[:bits_len-1]
   
  let f bin arr = offs[i32.u32 bin]+arr[i32.u32 bin]-1
  let is = map2 f bins offsets
  in scatter (copy xs) is xs
  

let radix_sort_step_nn_fast_big [n] (xs : [n]u32) (b : i32) : [n]u32 =
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
  if n <= 100000 then 
    if n <= 1000 then
      loop xs for i < 8 do radix_sort_step_nn xs (u32.i32 i) 4
    else 
      loop xs for i < 16 do radix_sort_step_nn xs (u32.i32 i) 2
  else 
    loop xs for i < 32 do radix_sort_step_nn_fast_big xs i
