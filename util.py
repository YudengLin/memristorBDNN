import numpy as np
import scipy.stats
import pandas as pd

def JS_div(arr1,arr2,num_bins):

    max0 = max(np.max(arr1),np.max(arr2))
    min0 = min(np.min(arr1),np.min(arr2))
    bins = np.linspace(min0-1e-4, max0-1e-4, num=num_bins)
    PDF1 = pd.cut(arr1,bins).value_counts() / len(arr1)
    PDF2 = pd.cut(arr2,bins).value_counts() / len(arr2)

    return JS_divergence(PDF1.values,PDF2.values)

def JS_divergence(p,q):

    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)


def compute_probs(data, n=10):
    h, e = np.histogram(data, n)
    p = h / data.shape[0]
    return e, p


def support_intersection(p, q):
    sup_int = (
        list(
            filter(
                lambda x: (x[0] != 0) & (x[1] != 0), zip(p, q)
            )
        )
    )
    return sup_int


def get_probs(list_of_tuples):
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q


def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))


def js_divergence(p, q):
    m = (1. / 2.) * (p + q)
    return (1. / 2.) * kl_divergence(p, m) + (1. / 2.) * kl_divergence(q, m)


def compute_kl_divergence(train_sample, test_sample, n_bins=10):
    """
    Computes the KL Divergence using the support
    intersection between two different samples
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)

    return kl_divergence(p, q)


def compute_js_divergence(train_sample, test_sample, n_bins=20):
    """
    Computes the JS Divergence using the support
    intersection between two different samples
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)

    return js_divergence(p, q)
