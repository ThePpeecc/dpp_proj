
# The ||: stops it from failing
clean: 
	rm bench 2&> /dev/null ||:
	rm bench.c 2&> /dev/null ||:
	rm test test.c 2&> /dev/null ||:
	rm *.actual 2&> /dev/null ||:
	rm *.expected 2&> /dev/null ||:
	rm -rf data


test:
	futhark test test.fut
	make clean
