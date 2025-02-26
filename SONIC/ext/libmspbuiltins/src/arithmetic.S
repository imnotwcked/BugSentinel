/* Builtins not (yet) implemented in Clang's compiler-rt. */

.global __mulhi3hw_noint
	.type __mulhi3hw_noint,@function
__mulhi3hw_noint:
    ; these addresses are for MSP430FR{5{8,9},6}xx
	MOV R15, &0x04C0
	MOV R14, &0x04C8
	MOV &0x04CA, R15
	RET
