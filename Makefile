
# The ||: stops it from failing
clean: 
	rm test test.c 2&> /dev/null ||:
	rm *.actual 2&> /dev/null ||:
	rm *.expected 2&> /dev/null ||:
	rm -rf data 
	rm *.tuning 2&> /dev/null ||:
	rm bench.c && rm bench


test:
	futhark test test.fut
	make clean
