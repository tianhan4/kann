-include ./HEWrapper/hewrapper/make/env.mk
CC=			g++
CFLAGS=		-g -Wall -O2 -fPIC -fopenmp -std=c++17
CFLAGS_LIB=	#-ansi -pedantic -Wno-long-long # ANSI C does not have inline which affects performance a little bit
INCLUDES :=	-I. -I./boost_1_75_0 -I./HEWrapper/hewrapper/include  -I./HEWrapper/hewrapper/src/seawrapper -I./HEWrapper/hewrapper/src/sealwrapper/include/SEAL-$(SEAL_VER)
EXE=		kann_client examples/test_kann_server examples/test examples/lenet examples/loader_test examples/save_load_test examples/layer_timer#examples/mnist-cnn examples/inspect
LIBS=		-L./HEWrapper/hewrapper -l$(HW_LINK) -L./HEWrapper/hewrapper/src/sealwrapper/lib -lseal -lpthread -lz -lm

.SUFFIXES: .cpp .o
.PHONY: all clean depend

.cpp.o:
		$(CC) -c $(CFLAGS) $(INCLUDES) $(CPPFLAGS) $< -o $@

all: HEWrapper/hewrapper/$(HW_LIB) kann_client.o kautodiff.o kann.o kann_extra/kann_data.o util.o $(EXE)

util.o: util.cpp
		$(CC) -c $(CFLAGS) $(CFLAGS_LIB) $(INCLUDES) $(CPPFLAGS) -o $@ $<

kautodiff.o: kautodiff.cpp HEWrapper/hewrapper/$(HW_LIB)
		$(CC) -c $(CFLAGS) $(CFLAGS_LIB) $(INCLUDES) $(CPPFLAGS) -o $@ $<

kann.o: kann.cpp HEWrapper/hewrapper/$(HW_LIB)
		$(CC) -c $(CFLAGS) $(CFLAGS_LIB) $(INCLUDES) $(CPPFLAGS) -o $@ $<

HEWrapper/hewrapper/$(HW_LIB):
		$(MAKE) -C HEWrapper/hewrapper

kann_extra/kann_data.o: kann_extra/kann_data.cpp HEWrapper/hewrapper/$(HW_LIB)
		$(CC) -c $(CFLAGS) -DHAVE_ZLIB $< -o $@

examples/test_kann_server: tcp/NetIO.hpp tcp/IOChannel.hpp examples/test_kann_server.o kautodiff.o kann.o kann_extra/kann_data.o util.o
		$(CC) $(CFLAGS) -o $@ examples/test_kann_server.o kautodiff.o kann.o kann_extra/kann_data.o util.o $(LIBS)

kann_client: tcp/NetIO.hpp tcp/IOChannel.hpp kann_client.o kautodiff.o kann.o kann_extra/kann_data.o util.o
		$(CC) $(CFLAGS) -o $@ kann_client.o kautodiff.o kann.o kann_extra/kann_data.o util.o $(LIBS)

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
		rm HEWrapper/hewrapper/$(HW_LIB)

depend:
		(LC_ALL=C; export LC_ALL; makedepend -Y -- $(CFLAGS) $(DFLAGS) -- *.cpp examples/*.cpp)

# DO NOT DELETE

kann.o: kann.h kautodiff.h util.h kann_extra/kann_data.h
kann_client.o: kann.h kautodiff.h kann_extra/kann_data.h util.h tcp/NetIO.hpp
kann_client.o: tcp/IOChannel.hpp
kautodiff.o: kautodiff.h util.h kann.h kann_extra/kann_data.h
util.o: kann.h kautodiff.h kann_extra/kann_data.h util.h
examples/layer_timer.o: kann.h kautodiff.h kann_extra/kann_data.h util.h
examples/lenet.o: kann.h kautodiff.h kann_extra/kann_data.h util.h
examples/loader_test.o: kann.h kautodiff.h kann_extra/kann_data.h util.h
examples/mnist-cnn.o: kann_extra/kann_data.h kann.h kautodiff.h
examples/save_load_test.o: kann.h kautodiff.h kann_extra/kann_data.h util.h
examples/test.o: util.h kann.h kautodiff.h kann_extra/kann_data.h
examples/test_kann_server.o: kann.h kautodiff.h kann_extra/kann_data.h util.h
examples/test_kann_server.o: tcp/NetIO.hpp tcp/IOChannel.hpp
