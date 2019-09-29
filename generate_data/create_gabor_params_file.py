import pickle

gabor_parameters_list = [
    [{
        'x0': 0,
        'y0': 0,
        'theta_deg': 60,
        'amp': 1,
        'sigma': 4.0,
        'lambda1': 8,
        'psi': 30,
        'gamma': 1
    }],
    [{
        'x0': 0,
        'y0': 0,
        'theta_deg': 130,
        'amp': 1,
        'sigma': 4.0,
        'lambda1': 7,
        'psi': 0,
        'gamma': 1
    }],
    [{
        'x0': 0,
        'y0': 0,
        'theta_deg': 110,
        'amp': 1,
        'sigma': 4.0,
        'lambda1': 11,
        'psi': 180,
        'gamma': 1
    }],
    [{
        'x0': 0,
        'y0': 0,
        'theta_deg': 150,
        'amp': 1,
        'sigma': 4.0,
        'lambda1': 9,
        'psi': 0,
        'gamma': 1
    }],
    [{
        'x0': 0,
        'y0': 0,
        'theta_deg': 20,
        'amp': 1,
        'sigma': 4.0,
        'lambda1': 10,
        'psi': 0,
        'gamma': 1
    }],
]

pickle_file = "bw_5_gabors_params.pickle"

with open(pickle_file,  'wb') as handle:
    pickle.dump(gabor_parameters_list, handle)

with open(pickle_file, 'rb') as handle:
    a = pickle.load(handle)

import pdb
pdb.set_trace()