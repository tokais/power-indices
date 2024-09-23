from math import factorial, floor
import itertools
import matplotlib.pyplot as plt
import numpy as np

def generate_coalitions(parties):
    """Returns list of all possible coalitions for given list of parties"""
    coalitions = []
    for i in range(len(parties)):
        coalitions.extend(list(itertools.combinations(parties, i+1)))
    return coalitions


def v(coalition, seats, threshold):
    """Examines whether coalition exceeds the majority threshold"""
    votes =0
    for _,party in enumerate(coalition):
        votes += seats[party]
    return 1 if votes >= threshold else 0


def calculate_threshold(sum):
    """Returns 50% of sum + 1 """
    return floor(sum / 2) + 1


def shapley_shubik(seats, threshold):
    """Returns Shapley-Shubik power indices"""
    parties = seats.keys()
    n = len(parties)
    coalitions = generate_coalitions(parties)
    shapley_values = {}

    for _,party in enumerate(parties):
        shapley_value = 0
        coalitions_with_party = [coalition for coalition in coalitions if party in coalition]
        for _,coalition in enumerate(coalitions_with_party):
            s = len(coalition)
            factor = (factorial(s-1) * factorial(n-s)) / factorial(n)
            shapley_value += factor * (v(coalition, seats, threshold) - v([x for x in coalition if x is not party], seats, threshold))
        shapley_values[party] = shapley_value

    return shapley_values


def banzhaf(seats, threshold):
    """Returns Banzhaf power indices"""
    parties = seats.keys()
    n = len(parties)
    coalitions = generate_coalitions(parties)
    banzhaf_values = {}

    for _,party in enumerate(parties):
        banzhaf_value = 0
        coalitions_with_party = [coalition for coalition in coalitions if party in coalition]
        for _,coalition in enumerate(coalitions_with_party):
            banzhaf_value += v(coalition, seats, threshold) - v([x for x in coalition if x is not party], seats, threshold)
        banzhaf_values[party] = banzhaf_value * (1 / (pow(2, n-1)))

    return banzhaf_values


def banzhaf_normalized(banzhaf_values):
    """Normalizes Banzhaf power indices"""
    B = 0
    for b in banzhaf_values.values():
        B += b

    banzhaf_normalized = {}

    for _,banzhaf_value in enumerate(banzhaf_values.items()):
        banzhaf_normalized[banzhaf_value[0]] = float(banzhaf_value[1]) / B

    return banzhaf_normalized


def deegan_packel(seats, threshold):  
    """Returns Deegan-Packel power indices"""
    parties = seats.keys()
    coalitions = generate_coalitions(parties)

    M = []
    for _, coalition in enumerate(coalitions):
        if v(coalition, seats, threshold) == 1:
            sub_coalitions = generate_coalitions(coalition)
            is_minimal = True
            for _, sub_coalition in enumerate(sub_coalitions):
                if sub_coalition != coalition and v(sub_coalition, seats, threshold) == 1:
                    is_minimal = False
            if is_minimal is True:
                M.append(coalition)
    
    deegan_packel_values = {}

    for _, party in enumerate(parties):
        deegan_packel_value = 0
        coalitions_with_party = [coalition for coalition in M if party in coalition]
        for _,coalition in enumerate(coalitions_with_party):
            deegan_packel_value += 1 / len(coalition)
        deegan_packel_values[party] = deegan_packel_value / len(M)
    
    return deegan_packel_values


def visualize_indices(shapley_shubik, banzhaf, deegan_packel):
    """Plots calculated indices"""
    parties = shapley_shubik.keys()
    x = np.arange(len(parties))
    
    plt.figure(figsize=(14, 8))
    
    bar1 = plt.bar(x - 0.25, shapley_shubik.values(), width=0.25, label='Shapley-Shubik')
    bar2 = plt.bar(x, banzhaf.values(), width=0.25, label='Banzhaf Normalized')
    bar3 = plt.bar(x + 0.25, deegan_packel.values(), width=0.25, label='Deegan-Packel')
    
    plt.xlabel('Parties')
    plt.ylabel('Power indices')
    
    plt.xticks(x, parties)
    
    def add_value_labels(bars):
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')
    
    add_value_labels(bar1)
    add_value_labels(bar2)
    add_value_labels(bar3)
    
    plt.legend()
    plt.tight_layout()
    plt.show()


def calculate_indices(seats, threshold=None):
    """Calculate and plot power indices for given seat allocation"""
    if threshold is None:
        threshold = calculate_threshold(sum(seats.values()))

    shapley_shubik_values = shapley_shubik(seats, threshold)
    banzhaf_values = banzhaf(seats, threshold)
    banzhaf_values_normalized = banzhaf_normalized(banzhaf_values)
    deegan_packel_values = deegan_packel(seats, threshold)

    visualize_indices(shapley_shubik_values, banzhaf_values_normalized, deegan_packel_values)