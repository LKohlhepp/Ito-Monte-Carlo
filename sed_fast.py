import numpy as np 
from numba import cuda
from numba.cuda import random
import math


"""
Shock simulations of cosmic-ray acceleration by Ito equations

Following: 
Krülls, W. M.; Achterberg, A. Computation of cosmic-ray acceleration by Ito's stochastic differential equations. 
Astronomy and Astrophysics, Vol. 286, p. 314-327. 1994


All shock conditions discussed further in Krülls (1994) fulfill the following conditons:
- functions are x, y independent / constant: 
- derivatives are always zero: a_1, a_2, kappa
- beta always has the form: a - b * tanh(x/x_sh)

These simplifications allow the simulation to be coded more ridgitly and thus more efficiently

"""


def build_funcs(a, b, x_sh, a_1, a_2, kappa, k_syn):

    """
    Create beta and its derivate as well as the shorthand notations as cuda device functions
    :param a: float = a from eq. 18
    :param b: float = b from eq. 18, height of the potential wall
    :param x_sh: float = x_sh from eq. 18, with of the potential wall
    :param a_1: float = shock conditions set by accordance to 4.
    :param a_2: float = shock conditions set by accordance to 4.
    :param kappa: float = shock conditions set by accordance to 4.
    :param k_syn: float = shock conditions set by accordance to 4.
    :return: Tuple(function(x, y), function(x, y), function(x, y), function(x, y))
    """
    @cuda.jit(device=True)
    def beta(x):
        return a - b * math.tanh(x/x_sh)

    @cuda.jit(device=True)
    def beta_div(x):
        return -b * 1/ math.cosh(x)**2

    @cuda.jit(device=True)
    def aa_1(x, y):
        return beta(x) + 3 * a_1

    @cuda.jit(device=True)
    def aa_2(x, y):
        return a_2 * 4 * y - k_syn * y**2 - 1/3 * beta_div(x) * y

    @cuda.jit(device=True)
    def b_11(x, y):
        return (2 * kappa)**0.5

    @cuda.jit(device=True)
    def b_22(x, y):
        return (2 * a_2)**0.5 * y 

    return aa_1, aa_2, b_11, b_22


def run_cuda(a, b, x_sh, a_1, a_2, kappa, k_syn, dt, iterations, amount, seed):

    """
    Create beta and its derivate as well as the shorthand notations as cuda device functions
    :param a: float = a from eq. 18
    :param b: float = b from eq. 18, height of the potential wall
    :param x_sh: float = x_sh from eq. 18, with of the potential wall
    :param a_1: float = shock conditions set by accordance to 4.
    :param a_2: float = shock conditions set by accordance to 4.
    :param kappa: float = shock conditions set by accordance to 4.
    :param k_syn: float = shock conditions set by accordance to 4.
    :param dt: float = size of the time step
    :param iterations: Union[int, List[int]] = Number of iterations per simulation, if List[int], a snapshot of all simulations is created for each entry
    :param amount: int = number of simulations run parallel
    :param seed: int = seed for the RNG
    :return: Union[np.ndarray(shape=(amount, 2)), np.ndarray(shape=(len(iterations), amount, 2))]
    """
    aa_1, aa_2, b_11, b_22 = build_funcs(a, b, x_sh, a_1, a_2, kappa, k_syn)

    @cuda.jit
    def solve_sed_cuda(initial, dt, iterations, rng_states):

        """
        This is the main simulation loop, written to be executed on a CUDA-GPU
        :param initial: np.ndarray(shape=(amount, 2)) = the initial conditions of the particals
        :param dt: float = size of the time step
        :param iterations: int = number of iterations to run for the simulation
        :param rng_states: numba.cuda.cudadrv.devicearray.DeviceNDArray(len=2 * amount) = states of the RNG
        :return: np.ndarray(shape=(amount, 2))
        """

        # get coordinates of the thread in 2D
        x, y = cuda.grid(2)
        if x < initial.shape[0]:
            # iterate over time steps
            for i in range(iterations):
                x_val, y_val = initial[x, 0], initial[x, 1]
                # as x(t) and y(t) are dependent on both x(t-1) and y(t-1), both threads must finish t-1 before t can be startet
                cuda.syncthreads()
                if y == 0:
                    initial[x, 0] = x_val + dt * aa_1(x_val, y_val) + dt**0.5 * b_11(x_val, y_val) * random.xoroshiro128p_normal_float64(rng_states, x * 2 + y)
                else:
                    initial[x, 1] = y_val + dt * aa_2(x_val, y_val) + dt**0.5 * b_22(x_val, y_val) * random.xoroshiro128p_normal_float64(rng_states, x * 2 + y)
                cuda.syncthreads()

    # set kernel size and calculate required amount of blocks for GPU
    kernel = (32, 2)
    blockspergrid = math.ceil(amount / kernel[1]), 1
    # initilise the rngs
    rng_states = random.create_xoroshiro128p_states(2 * amount, seed=seed)

    if type(iterations) == int:
        # set intitial conditions
        array = np.ones((amount, 2))
        # run simulation on GPU
        solve_sed_cuda[blockspergrid, kernel](array, dt, iterations, rng_states)
        return array
    else:
        results = np.zeros((len(iterations), amount, 2))
        initial = np.ones((amount, 2))
        last_iter = 0
        # if iterations is a list, it gives a snapshot for each run at the given iterations by stoping the simulation
        # copying the results and continuing the simulation at the given point. 
        for i in range(len(iterations)):
            solve_sed_cuda[blockspergrid, kernel](initial, dt, iterations[i]-last_iter, rng_states)
            results[i, :, :] = initial[:, :]
            last_iter = iterations[i]
        return results


if __name__ == '__main__':
    from time import time
    start = time()
    res = run_cuda(1, 1, 0.1, 0, 0, 1, 0, 10E-3, 1000, 10000, 666)
    print(time()-start)
    #print(res)
    import matplotlib.pyplot as plt 
    plt.plot(res[:, 0], res[:, 1], linewidth=0, marker='+')
    plt.show()
