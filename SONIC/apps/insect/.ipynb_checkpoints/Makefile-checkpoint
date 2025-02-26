OBJECTS = main.o
EXEC = insect

DEPS = libmspbuiltins libfixed libmat libdnn libalpaca libio libmsp libmspdriver libconsole

ifeq ($(BACKEND), tails)
DEPS += libdsp libmspdriver
endif

include tools/maker/Makefile
