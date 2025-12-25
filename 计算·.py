from scipy.optimize import fsolve, root
import numpy as np
r=3.3

def equation(x):
    return 23/r+29/x+4*x/(r**2)-17.5

def equation1(y):
    return (3.3**2)*3.3*(6*3.3+4*y)*((12*np.pi)**2)-1

initial_guess = 1
solution = fsolve(equation, initial_guess)
print(f"数值解：{solution[0]:.6f}")

solution2 = fsolve(equation1, initial_guess)
print(f"：{solution2[0]:.6f}")