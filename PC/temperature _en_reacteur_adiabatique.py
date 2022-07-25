import matplotlib.pyplot as plt
from math import exp

# Modelisation de l'hydrolyse de l'anhydride acetique dans des conditions
# adiabatiques

## Systeme etudie
# 1 mol d'anhydride acetique (102 g) et 10 mol d'eau (180 g) sont introduits
# dans un recipient calorifuge sous agitation magnetique. 
# On cherche a determiner l'evolution de la temperature en fonction du temps

## Donnees (variables globales)

T = 305             # Temperature initiale (K)
Cp = 940            # Capacite thermique du melange (J/K)
Delta_rH = -56000   # Enthalpie standard de reaction (J/mol)
dt = 0.01           # Pas de temps pour le calcul (min) - à T=305K, t1/2 = 18 min.
V = 0.275           # Volume estime a 275 mL (L)

## Constante cinetique (loi d'Arrhenius)
def k(T):
    # Renvoie la constante cinetique a la temperature T (min-1)
    return 1.68e7*exp(-6078.78/T)

TT = [T]    # Initialisation de la liste des temperatures
ksi = 0     # Avancement de la reaction
t = 0       # Initialisation du temps
tt = [t]    # Initialisation de la liste des valeurs de temps
c = 1/V     # Concentration initiale d'anhydride (mol/L)

## Methode d'Euler - lorsque le temps augmente de dt...
while ksi < 0.999:
    # On s'arrete lorsque 99.9% de l'anhydride acetique est hydrolyse
    dksi = k(T)*c*V*dt          # increment de l'avancement - réaction d'ordre 1 (eau en large excès)
    dT = -Delta_rH*dksi/Cp      # increment de temperature - réacteur adiabatique

    # Incrementation des variables
    ksi += dksi # ksi augmente de dksi
    T += dT     # T augmente de dT
    t += dt     # t augmente de dt
    c -= dksi/V

    # Stockage des variables
    TT.append(T)
    tt.append(t)

## Trace du graphe
plt.title('Trace de la temperature en fonction du temps')
plt.xlabel('Temps (min)')


plt.ylabel('Temperature (K)')
plt.plot(tt,TT)
plt.show()
plt.savefig('Temperature.png')
