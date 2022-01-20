-- I have it on good authority that this readable code
let radix_sort_intragroup [n] (xs: [n]u32) : [n]u32 = 
  let block_size = 4i64
  let tdxs = iota (block_size)
  let bit_mask = 15u32
  let rest = block_size - ( n % block_size)
  let modXS = if n % block_size != 0 then xs ++ (replicate (rest) u32.highest) else xs
  -- Helper function begining

  let radix_sort_MM [m] (xs : [m]u32) (b : u32) : [m]u32 =
    let block_num = m / block_size 
    -- Due to the fact that we have padded the input array
    let kernel_1 = map (\bid -> 
      let data  = xs[bid*block_size : (bid+1)*block_size]
      let radix = map (\elem -> i32.u32 (elem >> b*4 & bit_mask)) data :> [block_size]i32  
      let loop_mask  = replicate block_size 0
      let block_d_init = replicate 16 0
      let merged_mask_scan_block_d_sums = loop (loop_mask, block_d_init) for i < 16 do
          let mask = map (\r -> if r == i then 1 else 0) radix
          let exclusive_scanned_mask = [0] ++ scan (+) 0 mask
          let masksum = last exclusive_scanned_mask
          let block_d_updated = map(\j -> if  i == (i32.i64 j) then masksum else block_d_init[i] ) (iota 16)
          let loop_mask_updated = map (\tdx -> if i == radix[tdx] then exclusive_scanned_mask[tdx] else loop_mask[tdx]) tdxs 
          in (loop_mask_updated, block_d_updated)
      let merged_mask_scan = merged_mask_scan_block_d_sums.0
      let block_d_sums = merged_mask_scan_block_d_sums.1
      let histogram = [0] ++ (scan (+) 0 block_d_sums)
      let t_prefix_sums = map (\tdx -> merged_mask_scan[tdx]) tdxs  
      let new_poss = map2 (\t_prefix_sum r -> t_prefix_sum + histogram[r]) t_prefix_sums radix
      let sorted = map (\new_pos -> data[new_pos]) new_poss
      let prefixes = map (\new_pos ->  merged_mask_scan[new_pos]) new_poss
      in (sorted, prefixes, block_d_sums)
    ) (iota block_num)
    let local_sorted = flatten (map (.0) kernel_1)
    let prefixes = flatten (map (.1) kernel_1)
    let block_d_sums_permuted = flatten (map (.2) kernel_1)
    -- This have very shit locality
    let offsets = map (\idx -> 
      let bdx = idx / block_num
      let i = idx % block_num
      in block_num * i + bdx
    ) (iota (length block_d_sums_permuted))
    let block_d_sums = map (\offset -> block_d_sums_permuted[offset]) offsets
    let block_d_sums_length_m1 = (length block_d_sums) - 1
    let scanned_block_d_sums = [0] ++ scan (+) 0 block_d_sums[:block_d_sums_length_m1] -- Exclusive scan kernel
    let _ = trace scanned_block_d_sums
    let sorted = map (\idx -> 
      let data = local_sorted[idx]
      let global_radix = (i64.u32 (data >> b*4 & bit_mask)) * block_num + (idx / block_size)
      let data_global_pos = scanned_block_d_sums[global_radix] + prefixes[idx]
      in xs[data_global_pos]
    ) (iota m)
    in sorted
        
  let sortedModXS = loop modXS for i < 8 do radix_sort_MM modXS (u32.i32 i)
  in sortedModXS[:n]

  
-- ==
-- random input { [512]u32 }
--  input {  [23u32, 45u32, 12u32, 34564u32, 2112u32, 3222u32, 123u32, 3211u32, 534u32, 2u32, 3u32, 112u32] }
--  output { [2u32, 3u32, 12u32, 23u32, 45u32, 112u32, 123u32, 534u32, 2112u32, 3211u32, 3222u32, 34564u32] }
let main [n] (xs : [n]u32) : [n]u32 = 
  radix_sort_intragroup xs