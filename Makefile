-include ./HEWrapper/hewrapper/make/env.mk
CC=			g++
CFLAGS=		-g -Wall -O2 -fPIC -fopenmp -std=c++17
CFLAGS_LIB=	#-ansi -pedantic -Wno-long-long # ANSI C does not have inline which affects performance a little bit
INCLUDES :=	-I. -I./HEWrapper/hewrapper/include  -I./HEWrapper/hewrapper/src/seawrapper -I./HEWrapper/hewrapper/src/sealwrapper/include/SEAL-$(SEAL_VER)
EXE=		examples/test examples/lenet examples/loader_test examples/save_load_test examples/layer_timer#examples/mnist-cnn examples/inspect
LIBS=		-L./HEWrapper/hewrapper -l$(HW_LINK) -L./HEWrapper/hewrapper/src/sealwrapper/lib -lseal -lpthread -lz -lm

.SUFFIXES: .cpp .o
.PHONY: all clean depend hewrapper

.cpp.o:
		$(CC) -c $(CFLAGS) $(INCLUDES) $(CPPFLAGS) $< -o $@

all: clean hewrapper kautodiff.o kann.o kann_extra/kann_data.o util.o $(EXE)

util.o: util.cpp
		$(CC) -c $(CFLAGS) $(CFLAGS_LIB) $(INCLUDES) $(CPPFLAGS) -o $@ $<

kautodiff.o: kautodiff.cpp hewrapper
		$(CC) -c $(CFLAGS) $(CFLAGS_LIB) $(INCLUDES) $(CPPFLAGS) -o $@ $<

kann.o: kann.cpp hewrapper
		$(CC) -c $(CFLAGS) $(CFLAGS_LIB) $(INCLUDES) $(CPPFLAGS) -o $@ $<

hewrapper:
		$(MAKE) -C HEWrapper/hewrapper

kann_extra/kann_data.o: kann_extra/kann_data.cpp hewrapper
		$(CC) -c $(CFLAGS) -DHAVE_ZLIB $< -o $@

examples/test: examples/test.o kautodiff.o kann.o kann_extra/kann_data.o util.o
		$(CC) $(CFLAGS) -o $@ examples/test.o kautodiff.o kann.o kann_extra/kann_data.o util.o $(LIBS)

examples/lenet: examples/lenet.o kautodiff.o kann.o kann_extra/kann_data.o util.o
		$(CC) $(CFLAGS) -o $@ examples/lenet.o kautodiff.o kann.o kann_extra/kann_data.o util.o $(LIBS)

examples/layer_timer: examples/layer_timer.o kautodiff.o kann.o kann_extra/kann_data.o util.o
		$(CC) $(CFLAGS) -o $@ examples/layer_timer.o kautodiff.o kann.o kann_extra/kann_data.o util.o $(LIBS)

examples/loader_test: examples/loader_test.o kautodiff.o kann.o kann_extra/kann_data.o util.o
		$(CC) $(CFLAGS) -o $@ examples/loader_test.o kautodiff.o kann.o kann_extra/kann_data.o util.o $(LIBS)

examples/save_load_test: examples/save_load_test.o kautodiff.o kann.o kann_extra/kann_data.o util.o
		$(CC) $(CFLAGS) -o $@ examples/save_load_test.o kautodiff.o kann.o kann_extra/kann_data.o util.o $(LIBS)

# examples/inspect:examples/inspect.o kautodiff.o kann.o
# 		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# examples/mnist-cnn:examples/mnist-cnn.o kautodiff.o kann.o kann_extra/kann_data.o
#		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
		rm -fr *.o */*.o a.out */a.out *.a *.dSYM */*.dSYM $(EXE)

depend:
		(LC_ALL=C; export LC_ALL; makedepend -Y -- $(CFLAGS) $(DFLAGS) -- *.c kann_extra/*.c examples/*.c)

# DO NOT DELETE

examples/inspect.o: kann.h kautodiff.h
examples/mnist-cnn.o: kann_extra/kann_data.h kann.h kautodiff.h
