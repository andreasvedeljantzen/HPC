TARGET  = libmatmult.so
LIBSRCS = func.c
LIBOBJS = $(LIBSRCS:.c=.o)

OPT     = -g
PIC     = -fPIC
ISA	= -std=c99

CC	= gcc
CFLAGS= $(OPT) $(PIC) $(XOPTS) $(ISA)

SOFLAGS = -shared
XLIBS   = 

$(TARGET): $(LIBOBJS)
	$(CC) -o $@ $(SOFLAGS) $(LIBOBJS) $(XLIBS)

clean:
	@/bin/rm -f core core.* $(LIBOBJS)
