from math import pi, sin

from sympy import init_printing, diff, var, solve
from sympy import sin as sympy_sin
init_printing()

g_v, beta, gamma_a, gamma_b, gamma_c = var("g_v beta gamma_a gamma_b gamma_c")
a, b, c  = var(" a b c")

eq = g_v*sympy_sin(beta)*a*b*c + 2*gamma_a*b*c + 2*gamma_b*a*c + 2*gamma_c*a*b

deq_da = diff(eq, a)
deq_db = diff(eq, b)
deq_dc = diff(eq, c)

solution = solve([deq_da,deq_db,deq_dc],[a,b,c])[1]
print(solution)