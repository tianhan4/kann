-include ./HEWrapper/hewrapper/make/env.mk
CC=			g++
CFLAGS=		-g -Wall -Wextra -Wc++-compat -O2
CFLAGS_LIB=	#-ansi -pedantic -Wno-long-long # ANSI C does not have inline which affects performance a little bit
INCLUDES :=	-I. -I./HEWrapper/hewrapper/include -I./HEWrapper/hewrapper/include -I./HEWrapper/hewrapper/src/seawrapper -I./HEWrapper/hewrapper/src/sealwrapper/include/SEAL-$(SEAL_VER)
EXE=		examples/mlp examples/mnist-cnn examples/inspect
LIBS=		-L./HEWrapper/hewrapper -l$(HW_LINK) -L./HEWrapper/hewrapper/src/sealwrapper/lib -lseal -lpthread -lz -lm

.SUFFIXES:.c .o
.PHONY:all clean depend hewrapper

.c.o:
		$(CC) -c $(CFLAGS) $(INCLUDES) $(CPPFLAGS) $< -o $@

all:hewrapper kautodiff.o kann.o kann_extra/kann_data.o $(EXE)

kautodiff.o:kautodiff.c hewrapper
		$(CC) -c $(CFLAGS) $(CFLAGS_LIB) $(INCLUDES) $(CPPFLAGS) -o $@ $<

kann.o:kann.c hewrapper
		$(CC) -c $(CFLAGS) $(CFLAGS_LIB) $(INCLUDES) $(CPPFLAGS) -o $@ $<

hewrapper:
		$(MAKE) -C HEWrapper/hewrapper

kann_extra/kann_data.o:kann_extra/kann_data.c hewrapper
		$(CC) -c $(CFLAGS) -DHAVE_ZLIB $< -o $@

examples/mlp:examples/mlp.o kautodiff.o kann.o kann_extra/kann_data.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples/inspect:examples/inspect.o kautodiff.o kann.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples/mnist-cnn:examples/mnist-cnn.o kautodiff.o kann.o kann_extra/kann_data.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
		rm -fr *.o */*.o a.out */a.out *.a *.dSYM */*.dSYM $(EXE)

depend:
		(LC_ALL=C; export LC_ALL; makedepend -Y -- $(CFLAGS) $(DFLAGS) -- *.c kann_extra/*.c examples/*.c)

# DO NOT DELETE

kann.o: kann.h kautodiff.h
kautodiff.o: kautodiff.h
kann_extra/kann_data.o: kann_extra/kseq.h kann_extra/kann_data.h
examples/inspect.o: kann.h kautodiff.h
examples/mlp.o: kann.h kautodiff.h kann_extra/kann_data.h
examples/mnist-cnn.o: kann_extra/kann_data.h kann.h kautodiff.h
