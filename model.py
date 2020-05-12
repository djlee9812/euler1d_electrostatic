import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
plt.style.use("seaborn")
mpl.rcParams['axes.labelsize'] = 13
mpl.rcParams['axes.titlesize'] = 15
mpl.rcParams['figure.titlesize'] = 18

steps = 1000
length = 0.01 # [m]
delta_x = length / steps
# rho_c = 1e-6 # [C/m^3] Charge density
eps = 8.854e-12 # [F/m] Absolute Dielectric Constant
b = 2.43e-4 # [m^2/(V*sec)] Ion mobility
k = 2.534e-2 # [W/(m*K)] Thermal conductivity of air at sea level
R = 287 # [J/(kg*K)] Gas constant
Cv = 718 # [J/(kg*K)] Specific heat of air at constant volume (Cv*T+.5v^2 = e)
E_init = 0 # [V/m] Electric field at x=0
p_init = 101325 # [Pa] Pressure at x=0
rho_init = 1.225 # [kg/m^3] air density at x=0
v_init = 50 # [m/s] Carrier fluid velocity
u = 10000 # [V] voltage between emitter and collector
u_star = 0
n_stages = 100

def func_j(guess, E0, v0):
    """
    Relation of j, E, and U -> f(j) = 0
    :params guess: Guess for value of j
    """
    c1 = 2*length/(eps*b)
    c2 = (E0 + v0/b) ** 2
    c3 = 3*(u-u_star)/(eps*b) + 3*v0*length/(eps*b**2)
    c4 = (E0+v0/b)**3
    return (c1*guess+c2)**(3/2) - c3*guess - c4

def deriv_j(guess, E0, v0):
    """
    Derivative of func_j() wrt j
    :params guess: Guess for value of j
    """
    c1 = 2*length/(eps*b)
    c2 = (E0 + v0/b) ** 2
    c3 = 3*(u-u_star)/(eps*b) + 3*v0*length/(eps*b**2)
    return 3/2 * c1 * (c1*guess+c2)**(1/2) - c3

def solve_j(guess, E0, v0, reltol=1e-6):
    """
    Solves for positive solution for current density (in case 2 exist)
    """
    nfev = 0
    last_guess = 0
    value = 1
    while abs(last_guess - guess) > reltol or abs(func_j(guess, E0, v0)) > 1e3:
        last_guess = guess
        value = func_j(guess, E0, v0)
        deriv = deriv_j(guess, E0, v0)
        # Use abs value because function of function shape:
        # Should increase if f < 0, decrease if f > 0
        guess -= value / abs(deriv)/2
        nfev += 1
        if nfev >= 1000:
            print("Current density not found")
            print("j:", guess, "Function:", value, "Derivative:", deriv)
            break
    return guess

def E_analytical(x, j, E0, v0):
    return (2*j*x/(eps*b) + (E0 + v0/b)**2)**0.5 - v0/b

def incomp_p(x, j, E0, v0, p0):
    inner = (2*j*x/(eps*b) + (E0 + v0/b) ** 2) **(1/2) - (E0 + v0/b)
    outer = 2*j*x/(eps*b) - 2*v0/b * inner
    return eps/2 * outer + p0

def incomp_rho_c(x, j, E0, v0):
    return (j/b) / ( (2*j*x)/(eps*b) + (E0+(v0/b))**2 )**(.5)

def model_incomp(E0, v0):
    xs = np.array([delta_x * i for i in range(steps+1)])
    E_num = np.zeros(steps+1)
    p_num = np.zeros(steps+1)
    E_num[0] = E0
    p_num[0] = p0
    j = solve_j(1, E0, v0)
    rho_c_an = np.array([incomp_rho_c(x, j, E0, v0) for x in xs])
    p_an = np.array([incomp_p(x, j, E0, v0, p0) for x in xs])
    E_an = np.array([E_analytical(x, j, E0, v0) for x in xs])

    for i in range(1, steps+1):
        x = xs[i]
        p_num[i] = p_num[i-1] + rho_c_an[i]* E_num[i-1] * delta_x
        E_num[i] = E_num[i-1] + j / (eps * b) / (E_num[i-1] + v0/b) * delta_x

    plt.figure()
    plt.plot(xs, E_num, label="Numerical E(x)")
    plt.plot(xs, E_an, label="Analytical E(x)")
    plt.xlabel(r"$x$ [m]")
    plt.ylabel(r"$E(x)$ [V/m]")
    plt.title("Incompressible Model Electric Field")
    plt.legend()

    plt.figure()
    plt.plot(xs, p_num, label="Numerical p(x)")
    plt.plot(xs, p_an, label="Analytical p(x)")
    plt.xlabel(r"$x$ [m]")
    plt.ylabel(r"$p(x)$ [Pa]")
    plt.title("Incompressible Model Pressure")
    plt.legend()

    plt.figure()
    plt.plot(xs, rho_c_an)
    plt.xlabel(r"$x$ [m]")
    plt.ylabel(r"$\rho_c$ [C/m^3]")
    plt.title("Charge Density")
    plt.show()

def plot_iterations(results, step_size=3):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
    for i in range(0, len(results), step_size):
        ax1.plot(xs, results[i,:,0], label="Trial {0}".format(i))
        ax2.plot(xs, results[i,:,1], label="Trial {0}".format(i))
        ax3.plot(xs, results[i,:,2], label="Trial {0}".format(i))
    ax1.set_title("Pressure [Pa]")
    ax2.set_title("Air Density [kg / m^3]")
    ax3.set_title("Drift Velocity [m/s]")
    ax1.set_xlabel("x [m]")
    ax2.set_xlabel("x [m]")
    ax3.set_xlabel("x [m]")
    plt.legend()
    plt.tight_layout()
    plt.show()

def f_eval(x, u, rho0, v0, p0):
    """
    Evaluate f(x,u) = 0
    N = n_steps - 1 (index of last element)
    :param x: vector of n_steps nodes (rows) with p, rho, v as columns (n_steps, 3)
    :param u: vector of E and rho_c at every node (n_steps, 2)
    """
    p, rho, v = x.T
    T = p / (rho * R)
    e = Cv * T + 0.5 * v ** 2
    e0 = Cv * p0 / (rho0*R) + 0.5 * v0 ** 2
    rho_c, E = u.T
    N = x.shape[0] - 1
    # function evals as (N+1, 3) matrix
    f = np.zeros(x.shape)
    # Calculate first row using forward difference
    f[0] = [(rho[0]*v[0] - rho0*v0) / delta_x,
            ((rho[0]*v[0]**2+p[0]) - (rho0*v0**2+p0))/delta_x - rho_c[0]*E[0],
            (v[0]*(rho[0]*e[0]+p[0]) - v0*(rho0*e0+p0))/delta_x - rho_c[0]*E[0]*v[0]]

    # # Calculate 1 to N-1 using central difference
    f[1:N] = np.array([(rho[2:]*v[2:] - rho[:-2]*v[:-2]) / (2*delta_x),
            ((rho[2:]*v[2:]**2+p[2:]) - (rho[:-2]*v[:-2]**2+p[:-2])) / (2*delta_x) - rho_c[1:-1]*E[1:-1],
            (v[2:]*(rho[2:]*e[2:]+p[2:]) - v[:-2]*(rho[:-2]*e[:-2]+p[:-2])) / (2*delta_x) - rho_c[1:-1]*E[1:-1]*v[1:-1]
            - (T[2:]-2*T[1:-1]+T[:-2])/(delta_x**2)]).T

    # Calculate last row using backward difference
    f[N] = [(rho[N]*v[N] - rho[N-1]*v[N-1]) / delta_x ,
            ( ((rho[N]*v[N]**2+p[N]) - (rho[N-1]*v[N-1]**2+p[N-1]))/delta_x - rho_c[N]*E[N] ),
            ( (v[N]*(rho[N]*e[N]+p[N]) - v[N-1]*(rho[N-1]*e[N-1]+p[N-1]))/delta_x - rho_c[N]*E[N]*v[N] )]
    return f

def jac_est(x, u, rho0, v0, p0, eps=1e-3):
    """
    Numerical jacobian estimator for f
    Flattens x array into [p_0,...,p_i, rho_i, v_i,...,v_N+1] then iterates
    each term computing (f(x_i + eps) - f(x_i)) / eps
    Makes 3*(N+1) calls to f_eval
    """
    # Dimensions of jac matrix = ( 3*(N+1), 3*(N+1) )
    dim = x.flatten().shape[0]
    jac = np.zeros((dim, dim))

    # Iterate through each term in x = [p_0,...,p_i, rho_i, v_i,...,v_N+1]
    # and increment to evaluate ith column of jacobian
    for i, x_i in enumerate(x.flatten()):
        # inc = [0,...,eps,...,0] where only ith term = eps and others are 0
        inc = np.zeros(dim)
        inc[i] = eps
        inc = inc.reshape(x.shape)
        # Evaluate f at x+inc and at x for ith column of jacobian
        col = (f_eval(x+inc, u, rho0, v0, p0) - f_eval(x, u, rho0, v0, p0)) / eps
        jac[:,i] = col.flatten()
    return jac


def model(E0, rho0, v0, p0, tol=1, plot=False):
    # Use Incompressible analytical solution for E and rho_c
    xs = np.array([delta_x * i for i in range(steps)])
    j = solve_j(1, E0, v0)
    rho_c_an = np.array([incomp_rho_c(x, j, E0, v0) for x in xs])
    E_an = np.array([E_analytical(x, j, E0, v0) for x in xs])
    # u = [charge density.T, E-field.T] (N, 2)
    u = np.array([rho_c_an, E_an]).T
    # Initialize empty arrays and set initial conditions
    x = np.ones((steps, 3))
    x[:,0] *= p0
    x[:,1] *= rho0
    x[:,2] *= v0
    f = f_eval(x, u, rho0, v0, p0)
    # start = time.time()
    # for i in range(5000):
    #     f = f_eval(x, u, rho0, v0, p0)
    # print(str((time.time()-start)/5000) + "seconds each")
    # return
    nit = 0
    results = [x.copy()]
    while np.linalg.norm(f) > tol and nit < 100:
        jac = jac_est(x, u, rho0, v0, p0)
        try:
            dx = np.linalg.solve(jac, -f.flatten())
        except:
            print("Jacobian is singular")
            break
        x += dx.reshape(x.shape)
        f = f_eval(x, u, rho0, v0, p0)
        nit += 1
        if nit%3 == 0: print(nit, np.linalg.norm(f))
        results.append(x.copy())
    if nit >= 1e4: print("Too many iterations")
    results = np.array(results)

    if plot:
        plot_iterations(results)

    return x, np.linalg.norm(f), nit

if __name__ == "__main__":
    # plt.figure()
    # js = np.linspace(0, 0.2, 50)
    # plt.plot(js, func_j(js))
    # plt.show()

    p0, p_incomp0, rho0, v0, E0 = (p_init, p_init, rho_init, v_init, E_init)
    # model_incomp(E0, v0)

    x = np.empty((0,3))
    xs = np.array([])
    p_an = np.array([])
    for i in range(n_stages):
        # Solve j given initial E and v
        j = solve_j(1, E0, v0)
        # Append to xs arrray and analytical incomp pressure solution
        xs_i = np.array([delta_x * j for j in range(steps)])
        p_an_i = np.array([incomp_p(x, j, E0, v0, p_incomp0) for x in xs_i])
        xs = np.append(xs, xs_i+length*i)
        p_an = np.append(p_an, p_an_i)
        # Solve compressible values and append to x
        x_i, err, nit = model(E0, rho0, v0, p0)
        x = np.append(x, x_i, axis=0)
        # Reset boundary conditions for next stage
        p0, rho0, v0 = x_i[-1]
        p_incomp0 = p_an[-1]


    p, rho, v = x.T

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9,7))
    fig.suptitle("Change in Flow Quantities across Ion Thruster")
    ax1.plot(xs*1e3, p-p_init, label="Compressible")
    ax1.plot(xs*1e3, p_an-p_init, label="Incompressible")
    ax2.plot(xs*1e3, rho-rho_init)
    ax3.plot(xs*1e3, v-v_init)
    # ax4.plot(xs, p - p_an)
    ax4.plot(xs, p/(rho * R) - p_init/(rho_init*R))

    ax1.set_title("Pressure")
    ax2.set_title("Air Density")
    ax3.set_title("Bulk Velocity")
    # ax4.set_title("Pressure Difference")
    ax4.set_title("Temperature")

    ax1.set_xlabel(r"$x$ [mm]")
    ax2.set_xlabel(r"$x$ [mm]")
    ax3.set_xlabel(r"$x$ [mm]")
    # ax4.set_xlabel(r"$x$ [m]")
    ax4.set_xlabel(r"$x$ [mm]")

    ax1.set_ylabel(r"$\Delta p$ [Pa]")
    ax2.set_ylabel(r"$\Delta \rho$ [kg/m^3]")
    ax3.set_ylabel(r"$\Delta v$ [m/s]")
    # ax4.set_ylabel(r"$p_{comp}-p_{incomp}$ [Pa]")
    ax4.set_ylabel(r"$\Delta T$ [K]")

    ax1.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.figure()
    plt.plot(xs*1e3, p - p_an)
    plt.xlabel(r"$x$ [mm]")
    plt.ylabel(r"$p_{comp}-p_{incomp}$ [Pa]")
    plt.title("Pressure Difference")
    plt.tight_layout()
    plt.show()

    print("Final function value:", np.round(err, 3), ", iterations:", nit)
