import matplotlib.pyplot as plt
from math import log

# Détermination et trace du taux d'avancement de la synthèse de l'ammoniac
# en fonction de la temperature pour plusieurs valeurs de pression

## Systeme etudie
# etat initial : 1 mol de diazote et 3 mol de dihydrogene (proportions stoechiometriques)

## Donnees thermodynamiques
DrH = -91800    # enthalpie standard de reaction en J/mol
DrS = -199      # entropie standard de reaction en J/mol/K
R = 8.31        # J/mol/K

def DrG(tau,T,P):
    # renvoie l'enthalpie libre standard de reaction pour le taux d'avancement tau a T,P
    #
    return DrH-DrS*T+R*T*log(4*(tau**2)*(4-2*tau)**2/(27*(1-tau)**4*P**2))

def zero_DrG(T,P):
    # renvoie l'avancement de la reaction a l'equilibre
    # la recherche de la solution a l'equation donnee par la condition
    # d'equilibre est effectuee par dichotomie

    # Initialisation des variables
    gauche = 0.0001 # tau minimal = 0 mais l'enthalpie libre de reaction diverge en 0
    droite = 0.9999 # tau maximal = 1 (idem)
    tau = (droite-gauche)/2 # on commence par tester le milieu de l'intervalle

    while abs(DrG(tau,T,P)) > 1:   # condition d'arret : abs(DrG) < 1 J/mol
        if DrG(droite,T,P)*DrG(tau,T,P) > 0:
            # si DrG ne change pas de signe dans la moitie droite de l'intervalle
            droite = tau # la solution se situe dans la moitie gauche de l'intervalle
        else:
            gauche = tau # sinon elle se situe dans la moitie droite de l'intervalle
        tau = (droite+gauche)/2 # nouveau milieu de l'intervalle
    return tau

# Valeurs de pression pour lesquelles le trace est effectue (en bar):
liste_P = [1,5,25,125]

# Intervalle de temperature etudie : 300 K - 1200 K
liste_T = [T for T in range(300,1200,10)]  # Valeurs des T (abscisse des graphes)


for P in liste_P:
    # Creation de la liste des avancements a l'equilibre a la pression P
    # L'avancement maximal est de 1 mol, le taux d'avancement est egal
    # a la valeur numerique de l'avancement exprime en mol
    liste_tau = []
    for T in liste_T:
        liste_tau.append(zero_DrG(T,P))
        titre = 'P = '+str(P)+' bar'

    # Trace de la courbe pour la pression P
    plt.plot(liste_T,liste_tau,label=titre)


plt.ylabel("taux d'avancement a l'equilibre")
plt.xlabel("temperature (K)")
plt.legend()
plt.title("Influence de la temperature sur la synthese de l'ammoniac")

# Le graphe est enregistre en fichier image .png
plt.savefig('ammoniacT.png')

plt.show()


