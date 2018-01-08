SRCS	= main.c func.c
OBJS	= $(SRCS:.c=.o)
TARGET	= pi_cal.$(CC)

CC	= gcc
INCS	= 
DBG	= -g
OPT	= -O3
PARA	= 
CFLAGS	= $(DBG) $(OPT) $(PARA) $(XOPTS)

LIBS	= 

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	@/bin/rm -f $(OBJS) *~ core*

realclean: clean
	@/bin/rm -f $(TARGET)

