"""
Visualize Genetic Algorithm to find a maximum point in a function.
"""
import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 10            # Довжина ДНК
POP_SIZE = 100           # Розмір популяції
CROSS_RATE = 0.8         # Вірогідність схрещування (ДНК crossover)
MUTATION_RATE = 0.003    # Вірогідність мутаії
N_GENERATIONS = 200
X_BOUND = [0, 5]         # x верхня і нижня границя

def F(x): return np.sin(10*x)*x + np.cos(2*x)*x     # Знайти максимум в цій функції


# Знайти ненульовий фітнес для вибору
def get_fitness(pred): return pred + 1e-3 - np.min(pred)


# Перетворити двійкову ДНК у десяткову та нормалізувати його до діапазону (0, 5)
def translateDNA(pop): return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]


def select(pop, fitness):    # природній відбір wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]


def crossover(parent, pop):     # схрещування(genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)                             # вибирає іншу особу із вибірки
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # вибрати варіант схрещування точки
        parent[cross_points] = pop[i_, cross_points]                            # схрущування і виробництво
    return parent


def mutate(child): #мутація
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA

plt.ion()       # something about plotting
x = np.linspace(*X_BOUND, 200)
plt.plot(x, F(x))

for _ in range(N_GENERATIONS):
    F_values = F(translateDNA(pop))    # розрахування функції значення шляхом вилучення ДНК

    # something about plotting
    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)

    # GA part (evolution)
    fitness = get_fitness(F_values)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :])
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child       # parent is replaced by its child

plt.ioff(); plt.show()
