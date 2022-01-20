let sgmscan 't [n] (op: t->t->t) (ne: t) (flg : [n]i64) (arr : [n]t) : [n]t =
  let flgs_vals =
    scan ( \ (f1, x1) (f2,x2) ->
            let f = f1 | f2 in
            if f2 != 0 then (f, x2)
            else (f, op x1 x2) )
         (0,ne) (zip flg arr)
  let (_, vals) = unzip flgs_vals
  in vals


let radix_sort_intragroup [n] (xs: [n]u32) : [n]u32 = 
  let block_size = 256
  let tdxs = iota (block_size)
  let bit_mask = 15
  let rest = block_size - ( n % block_size)
  let modXS = if n % block_size != 0 then xs ++ (replicate (rest) u32.highest) else xs
  -- Helper function begining

  let radix_sort_MM [m] (xs : [m]u32) (b : u32) : [m]u32 =
    let block_num = m / block_size
    -- Due to the fact that we have padded the input array
    let kernel_1 = map (\bid -> 
      let data  = xs[bid*block_size : (bid+1)*block_size] :> [block_size]u32
      let radix = map (\elem -> i32.u32 (elem >> b*4 & bit_mask)) data :> [block_size]i32  
      let masks = map (\i -> 
        map (\r -> if i == i64.i32 r then 1 else 0) radix
      ) (iota 16)
      let scannedMasks = map (scan (+) 0) masks -- This is a seg scan, but i'm lazy
      let block_d_sums = map last scannedMasks
      let merged_mask_scan = map2 (\r tdx -> if tdx > 0 then scannedMasks[r, tdx - 1] else 0) radix tdxs
      let histogram = [0] ++ scan (+) 0 block_d_sums --exclusive scan
      let t_prefix_sums = map (\tdx -> merged_mask_scan[tdx]) tdxs  
      let new_poss = map2 (\t_prefix_sum r -> t_prefix_sum + histogram[r]) t_prefix_sums radix
      let sorted = scatter (replicate block_size 0) new_poss data
      in (sorted, merged_mask_scan, block_d_sums)
    ) (iota block_num)
    let local_sorted = flatten (map (.0) kernel_1)
    let prefixes = flatten (map (.1) kernel_1)
    let block_d_sums_permuted = map (.2) kernel_1
    -- This have very shit locality
    let block_d_sums = map (\idx -> 
      let bdx = idx / 16
      let i = idx % 16
      in block_d_sums_permuted[bdx,i]
    ) (iota (block_num * 16))
    let scanned_block_d_sums = [0] ++ scan (+) 0 block_d_sums -- Scan kernel
    let sorted = map (\idx -> -- Block shuffle Kernel 
      let data = xs[idx]
      let global_radix = (i64.u32 (data >> b*4 & bit_mask)) * block_num + (idx / block_size)
      let prefix = prefixes[idx]
      let data_global_pos = scanned_block_d_sums[global_radix] + prefix
      in xs[data_global_pos]
    ) (iota m)
    in sorted
        
  let sortedModXS = loop modXS for i < 8 do radix_sort_MM modXS (u32.i32 i)
  in sortedModXS[:n]
  

-- let xs = [23u32, 45u32, 12u32, 34564u32, 2112u32, 3222u32, 123u32, 3211u32, 534u32, 2u32, 3u32, 112u32]
-- let b = 1u32
-- let tdxs = (iota (length xs)) 
-- let bit_mask = 15u32
-- let radix = map (\elem -> i32.u32 (elem >> b*4 & bit_mask)) xs
-- let masks = map (\i -> map (\r -> if i == i64.i32 r then 1 else 0) radix) (iota 16)
-- let scannedMasks = map (scan (+) 0) masks
-- let block_d_sums = map last scannedMasks
-- let merged_mask_scan = map2 (\r tdx -> if tdx > 0 then scannedMasks[tdx - 1, r] else 0) radix tdxs
      

-- ==
-- random input { [512]u32 }
--  input {  [23u32, 45u32, 12u32, 34564u32, 2112u32, 3222u32, 123u32, 3211u32, 534u32, 2u32, 3u32, 112u32] }
--  output { [2u32, 3u32, 12u32, 23u32, 45u32, 112u32, 123u32, 534u32, 2112u32, 3211u32, 3222u32, 34564u32] }
let main [n] (xs : [n]u32) : [n]u32 = 
  radix_sort_intragroup xs
