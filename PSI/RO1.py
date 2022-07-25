import numpy as np
import matplotlib.pyplot as plt
from math import exp

# Modèle cinétique d'ordre un
A = 1e15     # facteur preexponentiel Arrhenius (1/s)
Ea = 157e3   # energie d'activation (J/mol)
R = 8.31     # constante des gaz parfaits (J/(K*mol))
h = 80       # coefficient de transfert thermique (echangeur de chaleur) (W/(m^2*K))
Cp = 2.1e3   # capacite thermique massique du liquide (J/(kg*K))
rho = 900    # masse volumique du reactif liquide (kg/m^3)
DrH = -150e3 # enthalpie standard de la reaction de decomposition (J/mol)
tau = 600    # temps de passage (s)
S = 3.1e-2   # surface du reacteur, sur laquelle a lieu l'echange de chaleur (m^2)
T0 = 273     # temperature du fluide refrigerant dans l'echangeur de chaleur (K)
Te = 400     # temperature des reactifs a l'entree (K)
V = 5.2e-4   # volume du reacteur (m^3)
M = 146e-3   # masse molaire du peroxyde de ditertiobutyle (kg/mol)
Q=V/tau      # débit volumique

Ta=Ea/R      #température d'activité
T00=T0/Ta    #température adimensionnée du réfrigérant
alpha=-h*S*Ta*M/Q/rho/DrH  #coeffient sans dimension
beta=V*A/Q   #coefficient sans dimension

def X(T):
    q=beta*np.exp(-(1/T))
    return q/(1+q)

def phi_ext(T):
    # Renvoie le flux thermique évacué par le réfrigérant
    return alpha*(T-T00)

T=np.linspace(1e-4,5e-2,101)
plt.xlabel('T/Ta')
plt.ylabel('Puissance adimensionnée')
plt.xlim(0,5e-2)
plt.ylim(-0.1,1.3)
plt.plot(T,X(T),'r',label = 'taux de conversion')
plt.plot(T,phi_ext(T),'b',label='flux thermique adimensionné')
plt.legend(loc='best')
Tf=np.array([0.0144,0.0246748,0.0315])
plt.plot(Tf,X(Tf),'ko')
plt.show()
