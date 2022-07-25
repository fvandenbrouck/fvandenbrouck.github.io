#Etude de la décomposition du peroxyde de ditertiobutyle dans un RPAC
#Modèle cinétique d'ordre un, cycle d'hystérésis

import numpy as np
import matplotlib.pyplot as plt

A = 1e15     #facteur préexponentiel Arrhenius (1/s)
Ea = 157e3   #énergie d'activation (J/mol)
R = 8.31     #constante des gaz parfaits (J/(K*mol))
h = 80       #coefficient de transfert thermique (échangeur de chaleur) (W/(m^2*K))
Cp = 2.1e3   #capacité thermique massique du liquide (J/(kg*K))
rho = 900    #masse volumique du réactif liquide (kg/m^3)
DrH = -150e3 #enthalpie standard de la réaction de décomposition (J/mol)
tau = 600    #temps de passage (s)
S = 3.1e-2   #surface du réacteur, sur laquelle a lieu l'échange de chaleur (m^2)
Te = 400     #température des réactifs à l'entree (K)
V = 5.2e-4   #volume du réacteur (m^3)
M = 146e-3   #masse molaire du peroxyde de ditertiobutyle (kg/mol)
Q=V/tau      #débit volumique

Ta=Ea/R      #température d'activité
alpha=-h*S*Ta*M/Q/rho/DrH  #coefficient sans dimension
beta=V*A/Q   #coefficient sans dimension

def X(T):
    # Renvoie le taux de conversion
    # T est la température adimensionnée
    q=beta*np.exp(-(1/T))
    return q/(1+q)

def phi(T):
    # Renvoie le flux thermique adimensionné évacué par le réfrigérant
    # T et T0 sont les températures adimensionnées
    return alpha*(T-T0)

T0=1e-3
T=1e-5
dTx=1e-4
dTy=1e-6
Tx=[]
Ty=[]
while T0<0.035:
    while X(T)>phi(T):
        T+=dTy
    Tx.append(T0)
    Ty.append(T)
    T0+=dTx
while T0>1e-3:
    while X(T)<phi(T):
        T+=-dTy
    Tx.append(T0)
    Ty.append(T)
    T0+=-dTx

plt.xlabel('T0/Ta')
plt.ylabel('T/Ta')
plt.plot(Tx,Ty,'g')
plt.show()
