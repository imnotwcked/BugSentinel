LIB = libmspbuiltins

OBJECTS =

include $(MAKER_ROOT)/Makefile.version.clang

ifneq ($(call version-lt,$(CLANG_VERSION),5.0.0),)
OBJECTS += arithmetic.o
endif # Clang < 5.0

override SRC_ROOT = ../../src

override CFLAGS += \
	-I $(SRC_ROOT)/include/$(LIB) \

include $(MAKER_ROOT)/Makefile.$(TOOLCHAIN)
