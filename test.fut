import "radix-4-way"
import "radix-8-way"
import "lib/github.com/diku-dk/sorts/radix_sort"
import "radix-intragroup"

-- ==
-- entry: test_radix_4_way
-- random input { [10]u32 }
-- output { 10 }
-- random input { [17]u32 }
-- output { 17 }
-- random input { [1391]u32 }
-- output { 1391 }

entry test_radix_4_way [n] (xs : [n]u32) = 
  let our_res = radix_sort_4_way xs
  let correct_res = radix_sort 32 u32.get_bit xs
  let comp = map2 (==) our_res correct_res
  let converted_comp = map (i32.bool) comp
  in reduce (+) 0 converted_comp

-- ==
-- entry: test_radix_intragroup
-- random input { [10]u32 }
-- output { 10 }
-- random input { [17]u32 }
-- output { 17 }
-- random input { [1391]u32 }
-- output { 1391 }

entry test_radix_intragroup [n] (xs : [n]u32) = 
  let our_res = radix_sort_intragroup xs
  let correct_res = radix_sort 32 u32.get_bit xs
  let comp = map2 (==) our_res correct_res
  let converted_comp = map (i32.bool) comp
  in reduce (+) 0 converted_comp

-- ==
-- entry: test_radix_intragroup
-- random input { [10]u32 }
-- output { 10 }
-- random input { [17]u32 }
-- output { 17 }
-- random input { [1391]u32 }
-- output { 1391 }

entry test_radix_8_way [n] (xs : [n]u32) = 
  let our_res = radix_sort_8_way xs
  let correct_res = radix_sort 32 u32.get_bit xs
  let comp = map2 (==) our_res correct_res
  let converted_comp = map (i32.bool) comp
  in reduce (+) 0 converted_comp

