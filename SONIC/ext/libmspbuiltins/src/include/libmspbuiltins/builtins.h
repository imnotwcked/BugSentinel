#ifndef _MSP_BUILTINS_H
#define _MSP_BUILTINS_H

#include <stdint.h>

#define __delay_cycles(n) \
    __asm__ volatile ( \
      "pushm.a #1, r13\n" \
      "mov     %[count], r13\n" \
      "dec     r13\n" \
      "jnz     $-2\n" \
      "popm.a  #1, r13\n" \
      "nop\n" \
      : : [count] "i" ((n) / 3 - 3) \
    )

#endif // _MSP_BUILTINS_H
