import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_input():
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for i in range(len(t)):
        t[i] = t[i]*500
    # Plotowanie wejścia U
    axes[0].plot(t, U[0], label='$u_1$')
    axes[0].plot(t, U[1], label='$u_2$')
    axes[0].set_ylabel('$\mathbf{u}$')
    axes[0].legend(loc='upper right')
    axes[0].yaxis.label.set_rotation(0)  # 0 stopni - napis jest poziomy
    axes[0].yaxis.label.set_ha('right')  # Ustawienie prawostronnego umieszczenia napisu
    axes[0].set_title('Sygnały wejściowe $u_1, u_2$')  # Dodanie tytułu dla pierwszego subplotu

    # Plotowanie wyjścia Y
    axes[1].plot(t, Y[0], label='$y_1$')
    axes[1].plot(t, Y[1], label='$y_2$')
    axes[1].set_xlabel('Czas')
    axes[1].set_ylabel('$\mathbf{y}$')
    axes[1].legend(loc='upper right')
    axes[1].yaxis.label.set_rotation(0)  # 0 stopni - napis jest poziomy
    axes[1].yaxis.label.set_ha('right')  # Ustawienie prawostronnego umieszczenia napisu

    axes[1].set_title('Sygnały wyjściowe $y_1, y_2$')  # Dodanie tytułu dla drugiego subplotu

    plt.show()


def plot_Q():
    plt.figure(figsize=(8, 6))
    plt.plot(u1_values, Qlist, label='Q(u1)')
    #plt.scatter(u1_values, Qlist, color='red', label='Minimum Q')
    plt.title('Wartości funkcji kosztów $Q(u_1, u_2)$ od najlepszego $u_1$')
    plt.xlabel('$u_1$')
    plt.ylabel('$Q(u_1, u_2)$')
    print(len(u1))
    print(len(Qlist))
    plt.grid(True)
    plt.show()


def plot_circle():
    # Tworzenie danych do okręgu
    theta = np.linspace(0, 2 * np.pi, 100)
    u1 = np.cos(theta)
    u2 = np.sin(theta)

    # Narysowanie okręgu
    plt.figure(figsize=(6, 6))
    if Ubest is not None:
        plt.scatter(1.98, 2.97851562, color='red', label='Optymalne sterowanie')
        plt.scatter(Ubest[0], Ubest[1], color='green', label='Optymalne sterowanie z ograniczeniami')

    plt.plot(u1, u2, label=r'$u_1^2 + u_2^2 \leq 1$')
    plt.fill(u1, u2, alpha=0.25)  # Wypełnienie obszaru w okręgu
    plt.title('Sterowanie optymalne wartościami $u_1$ i $u_2$')
    plt.xlabel('$u_1$')
    plt.ylabel('$u_2$')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.axis('equal')  # Równe skale osi x i y
    plt.show()

#tworzenie macierzy U
t = np.linspace(0, 1, 500, endpoint=False)
fala1 = 4 * np.abs(np.mod(t - 0.25, 0.5) - 0.25)
fala2 = 4 * np.abs(np.mod(t -0.15- 0.25, 0.5) - 0.25)
for i in range(len(fala2)):
    fala2[i] = 0.71
    fala1[i] = 0.7
U = np.array([fala1, fala2])

#Z zakłóceń
a: float = 0.01
Z1 = np.random.triangular(-a, 0, a, len(t))
Z2 = Z1.copy()
random.shuffle(Z2)
Z = np.array([Z1, Z2])

#tworzenie macierzy A i B i H
A = np.array([[1/2, 0], [0, 1/4]])
B = np.array([[1, 0], [0, 1]])
H = np.array([[0, 1], [1, 0]])

#tworzenie macierzy Y
inverse_matrix = np.linalg.inv(np.eye(A.shape[0]) - A @ H)
term1 = inverse_matrix @ B @ U
term2 = inverse_matrix @ Z
Y = term1 + term2

# WYKRES U i Y
plot_input()

#etap1
#tworzenie macierzy W
W1 = np.array([Y[1], U[0]])
W2 = np.array([Y[0], U[1]])

#estymawanie a1 i b1
E1 = Y[0]@W1.T@np.linalg.inv(W1@W1.T)

#estymawanie a2 i b2
E2 = Y[1]@W2.T@np.linalg.inv(W2@W2.T)
print(E1, E2)

#etap 2
#algorytm
Ubest = np.array([0, 0])
Qbest = np.inf
A_estimated = np.array([[E1[0], 0], [0, E2[0]]])
B_estimated = np.array([[E1[1], 0], [0, E2[1]]])
Qlist = []
U2list = []
T = {}
i = 0
u1_lista = []
u1_values = np.arange(-1, 1, 0.01)
for u1 in u1_values:
    u2min = -np.sqrt(1 - u1 ** 2)
    u2max = np.sqrt(1 - u1 ** 2)
    e = u2max
    Qtest = 0
    U2test = []
    while True:
        uhalf = (u2min + u2max)/2
        U1 = np.array([[u1], [uhalf-e]])
        U2 = np.array([[u1], [uhalf+e]])

        random_Z = np.random.choice(Z.flatten(), size=(2, 1))
        random_Z = random_Z.reshape((2, 1))

        inverse_matrix = np.linalg.inv(np.eye(A_estimated.shape[0]) - A_estimated @ H)
        term1 = inverse_matrix @ B_estimated @ U1
        term2 = inverse_matrix @ random_Z
        Y1 = term1 + term2

        random_Z = np.random.choice(Z.flatten(), size=(2, 1))
        random_Z = random_Z.reshape((2, 1))
        term1 = inverse_matrix @ B_estimated @ U2
        term2 = inverse_matrix @ random_Z
        Y2 = term1 + term2

        Q1 = (Y1[0] - 4) ** 2 + (Y1[1] - 4) ** 2
        Q2 = (Y2[0] - 4) ** 2 + (Y2[1] - 4) ** 2

        if i not in T:
            T[i] = []
            T[i+1] = []

        T[i].append({"U1": u1, "U2": U1, "Q": Q1})
        T[i+1].append({"U1": u1, "U2": U2, "Q": Q2})

        if Q1 <= Q2:
            if Q1 < Qbest:
                Qbest = Q1
                Ubest = U1
            u2max = uhalf
            Qtest = Q1
            Utest = U1[1]
        else:
            if Q2 < Qbest:
                Qbest = Q2
                Ubest = U2
            Qtest = Q2
            Utest = U2[1]
            u2min = uhalf
        e = e / 2
        i += 2
        if e < 0.001:
            break
    Qlist.append(Qtest)
    U2list.append(Utest)

u1 = np.arange(-1, 1, 0.01)

plot_Q()
plot_circle()

print(Ubest)
print(np.min(Qbest))
# print(np.eye(A.shape[0]) )
# u1 = np.arange(-1, 1, 0.01)
# plt.plot(u1, Qlist)
# plt.xlabel('u1')
# plt.ylabel('Q')
# plt.title('Wartość Q dla najlepszego u2 dla każdego u1')
#
# print(Ubest)

# print(E1)
# print(E2)
#print(W1.shape)
# print(a1)
# print("\n\n\n")
# print(b1)
# print(Y)
'''
plt.scatter(t, fala1, label='fala1')
plt.scatter(t, fala2, label='fala2')
plt.scatter(t, Y[0], label='Y[0]')
plt.scatter(t, Y[1], label='Y[1]')
'''
# plt.savefig(f"Mse of horizion 5000.png", dpi=300)
#
'''
plt.xlabel('t')
plt.ylabel('Y')
plt.title('Wykres zależności Y od t')
plt.legend()
plt.show()
'''
'''
plt.figure(figsize=(12, 6))


# Wykres 1 - Y[0]

plt.subplot(1, 2, 1)
plt.plot(t, Y[0], label='y1')
plt.plot(t, Y[1], label='y2')
plt.xlabel('t')
plt.ylabel('Y')
plt.title('Wykres wyjść')
plt.legend()

# Wykres 2 - Fala1 i Fala2
plt.subplot(1, 2, 2)
plt.scatter(t, fala1, label='u1')
plt.scatter(t, fala2, label='u2')
plt.xlabel('t')
plt.ylabel('U')
plt.title('Wykres wejść')
plt.legend()
'''
'''
for angle in range(0, 360, 15):
    u1 = np.arange(-1, 1, 0.01)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    u1 = np.arange(-1, 1, 0.01)
    # Dodaj punkty do wykresu 3D
    ax.scatter(u1, U2list, Qlist, c='r', marker='o')

    ax.set_xlabel('U1')
    ax.set_ylabel('U2')
    ax.set_zlabel('Wartość Q')
    ax.set_title('Q(U1, U2)')
    ax.view_init(15, angle)
    plt.draw()
    # plt.pause(0.001)
    # plt.show()
'''
