
# The ||: stops it from failing
clean: 
	rm bench bench.c 2&> /dev/null ||:
	rm test test.c 2&> /dev/null ||:
	rm -rf data


test:
	futhark test test.fut
	make clean
