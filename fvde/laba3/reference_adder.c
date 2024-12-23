#include <stdint.h>
#include <stdio.h>

// Reference adder function for IEEE 754 double precision
void reference_adder(long long a, long long b, long long *result) {
    // Interpret inputs as double using type punning
    double operand_a, operand_b, res;
    operand_a = *(double*)&a;
    operand_b = *(double*)&b;

    // Perform addition
    res = operand_a + operand_b;

    // Cast the result back to long long
    *result = *(long long*)&res;

    // Optional: Print debug information
    printf("Reference Adder: Operand A = %e, Operand B = %e, Result = %e\n",
           operand_a, operand_b, res);
}
