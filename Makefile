CC = gcc
CFLAGS = -g -Wall -Werror
ASFLAGS = -g

# Detect if the OS is 64 bits.  If so, request 32-bit builds.
LBITS := $(shell getconf LONG_BIT)
ifeq ($(LBITS),64)
  CFLAGS += -m32
  ASFLAGS += -32
endif


all: radiation test_glm


clean:
	rm -rf *.o *~ radiation test_glm


	
radiation: radiation.cc radiation.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_glm: test_glm.cc test_glm.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)


.PHONY: all clean

