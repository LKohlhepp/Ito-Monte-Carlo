import numpy as np 

"""
Shock simulations of cosmic-ray acceleration by Ito equations

Following: 
Krülls, W. M.; Achterberg, A. Computation of cosmic-ray acceleration by Ito's stochastic differential equations. 
Astronomy and Astrophysics, Vol. 286, p. 314-327. 1994
"""


class DevFunc2d:

    """
    Container for grouping a 2D-function and its derivatives in x and y directions
    """
    def __init__(self, func, dev_x, dev_y):
        """
        Initilize the container
        :param func: function(x, y) = primary function
        :param dev_x: function(x, y) = derivative in x-direction of the primary function
        :param dev_y: function(x, y) = derivative in y-direction of the primary function
        :return: DevFunc2d
        """
        self.func = func
        self.dev_x = dev_x
        self.dev_y = dev_y

    def __call__(self, x, y):
        return self.func(x, y)

class DevFunc1d:

    """
    Container for grouping a 2D-function and its derivative
    """
    def __init__(self, func, dev):
        """
        Initilize the container
        :param func: function(x) = primary function
        :param dev: function(x) = derivative of the primary function
        """
        self.func = func
        self.dev = dev 

    def __call__(self, x):
        return self.func(x)


def build_scaling_func(kappa, beta, a_1, a_2, k_syn):

    """
    Wrapper function for creating the shorthand notation of eq. 17
    :param kappa: DevFunc2d = kappa from eq. 13
    :param beta: DevFunc1d = beta from eq. 13
    :param a_1: DevFunc2d = a_1 from eq. 13
    :param a_2: DevFunc2d = a_2 from eq. 13
    :param k_syn: DevFunc1d = k_syn from eq. 13
    :return: Tuple[function(x, y), function(x, y), function(x, y), function(x, y)]
    """

    def aa_1(x, y):
        return kappa.dev_x(x, y) + beta(x) + y * a_1.dev_y(x, y) + 3* a_1(x, y)

    def aa_2(x, y):
        return y**2 * a_2.dev_y(x, y) + 4 * a_2(x, y) * y - a_1.dev_x(x, y) * y - k_syn(x) * y**2 - 1/3 * beta.dev(x) * y

    def b_11(x, y):
        return (2*kappa(x, y))**0.5

    def b_22(x, y):
        return (2*a_2(x, y))**0.5 * y

    return aa_1, aa_2, b_11, b_22


"""
These are the implementations of the different physical models
"""

class PureShock:

    """
    Container for generating parameters corresponding to a simulation with shocks only. 
    following chapter 4.1
    """
    def __init__(self, a, b, x_sh):
        """
        Initilize parameters for a simulation with shocks only
        :param a: float = a from eq. 18
        :param b: float = b from eq. 18
        :param x_sh: float = x_sh from eq. 18
        :return: PureShock
        """
        def zero_func(x, y):
            return 0
        self.a_1 = DevFunc2d(zero_func, zero_func, zero_func)
        self.a_2 = DevFunc2d(zero_func, zero_func, zero_func)
        self.k_syn = DevFunc1d(lambda x: 0, lambda x: 0)
        self.beta = DevFunc1d(lambda x: a-b* np.tanh(x/x_sh), lambda x: -b * 1/ np.cosh(x)**2)
        self.kappa = DevFunc2d(lambda x, y: 1, zero_func, zero_func) # ???


class ShockSynchrotron:

    """
    Container for generating parameters corresponding to a simulation with shocks and syncrotron radiation. 
    following chapter 4.2
    """
    def __init__(self, a, b, x_sh):
        """
        Initilize parameters for a simulation with shocks and syncrotron radiation
        :param a: float = a from eq. 18
        :param b: float = b from eq. 18
        :param x_sh: float = x_sh from eq. 18
        :return: ShockSynchrotron
        """
        self.a_1 = DevFunc2d(lambda x: 9, zero_func, zero_func)
        self.a_2 = DevFunc2d(lambda x: 2, zero_func, zero_func)
        # I don't understand a = beta^2 / 4 kappa k_syn
        self.k_syn = DevFunc1d(lambda x: 0, lambda x: 0)
        self.beta = DevFunc1d(lambda x: a-b* np.tanh(x/x_sh), lambda x: -b * 1/ np.cosh(x)**2)
        self.kappa = DevFunc2d(lambda x, y: 1, zero_func, zero_func) # ???


class SecondFermiSynchrotron:

    """
    Container for generating parameters corresponding to a simulation with Fermi-shocks of 1st and 2nd order and syncrotron radiation. 
    following chapter 4.3
    """
    def __init__(self, a, b, x_sh):
        """
        Initilize parameters for a simulation with Fermi-shocks of 1st and 2nd order and syncrotron radiation
        :param a: float = a from eq. 18
        :param b: float = b from eq. 18
        :param x_sh: float = x_sh from eq. 18
        :return: SecondFermiSynchrotron
        """
        self.a_1 = DevFunc2d(lambda x: 9, zero_func, zero_func)
        self.a_2 = DevFunc2d(lambda x: 10E-2, zero_func, zero_func)
        # I don't understand a = beta^2 / 4 kappa k_syn
        self.k_syn = DevFunc1d(lambda x: 10E-3, lambda x: 0)
        self.beta = DevFunc1d(lambda x: a-b* np.tanh(x/x_sh), lambda x: -b * 1/ np.cosh(x)**2)
        self.kappa = DevFunc2d(lambda x, y: 1, zero_func, zero_func) # ???
     

def iterate_sed(n, dt, shock_params):

    """
    Does one simulation of the Ito sDGL
    :param n: int = number of time steps to run 
    :param dt: float = size of the time steps
    :shock_params: Union[PureShock, ShockSynchrotron, SecondFermiSynchrotron] = Type of simulation and boundry conditions
    :return: np.ndarray(shape=(2, n))
    """

    # create the shorthand notation from the type of simulation and boundry conditions
    aa_1, aa_2, b_11, b_22 = build_scaling_func(shock_params.kappa, shock_params.beta, shock_params.a_1, shock_params.a_2, shock_params.k_syn)
    # pre generate the random numbers
    array = np.random.standard_normal(size=(2, n))
    array[0, 0], array[1, 0] = 1, 1 #x_0, y_0 = 1, 1

    # iterate over the time steps
    for i in range(1, n):
        array[0, i] = array[0, i-1] + aa_1(array[0, i-1], array[1, i-1]) * dt + b_11(array[0, i-1], array[1, i-1]) * array[0, i] * dt**0.5
        array[1, i] = array[1, i-1] + aa_2(array[0, i-1], array[1, i-1]) * dt + b_22(array[0, i-1], array[1, i-1]) * array[1, i] * dt**0.5

    return array


def bulk_iterate_sed(n, dt, shock_params, amount):

    """
    Does multiple runs of the simulation for the same parameters, as it is a Monte-Carlo simulation
    this will still result in different results
    :param n: int = number of time steps to run 
    :param dt: float = size of the time steps
    :param shock_params: PureShock, ShockSynchrotron, or SecondFermiSynchrotron = Type of simulation and boundry conditions
    :param amount: int = number of simulations to be run
    :return: np.ndarray(shape=(amount, 2, n))
    """
    results = np.zeros((amount, 2, n))
    for i in range(amount):
        results[i][:, :] = iterate_sed(n, dt, shock_params)[:, :]
    return results


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n = 10000
    res = []
    for i in range(n):
        res.append(iterate_sed(1000, 10E-3, PureShock(1, 1, 0.1)))
    print(res)
    for i in range(n):
        plt.plot(res[i][0], res[i][1], linewidth=0, marker='+')
    #plt.yscale('log')
    plt.show()
    time_sampled_y = [res[i][1][640] for i in range(n)]
    plt.hist(time_sampled_y, bins=100, histtype='step')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


