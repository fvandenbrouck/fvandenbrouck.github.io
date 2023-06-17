#%%
# Precision : tout ce qui suit le signe # ne participe pas au script, il s'agit simplement de commentaires 
# ayant pour but d'eclairer le script ou la physique mise en jeu.
# pour plus de lisibilite, quand les lignes de script ou de commentaire sont trop longues pour être affichées 
# dans un ecran, il arrivera que l'on passe �  la ligne suivante dans le commentaire. 
# L'utilisation de l'association #%% permet de délimiter des cellules indépendantes que l'interpréteur pourra traiter individuellement.
#%%

import numpy as np # on importe le module numpy et on le renomme np, pour pouvoir l'appeler plus simplement dans la suite
import matplotlib.pyplot as plt # on importe le module matplotlib.pyplot et l�  aussi, on le renomme

#%%
ymes=np.array([-0,-0.7,-1.5,-2.3,-3.5,-4.5,-5.9,-7.7,-8.8,-10.6,-12.3,-14.2,-16.4,-18.5,-21,-23.5])
# on définit un tableau du type array (ce qui permet de faire des opérations sur les tableaux) 
# le np. signifie qu'un tableau du type array est géré par le module numpy
print('ymes:',ymes)
# on fait afficher le tableau ymes pour vérifier qu'il correspond bien

#%%
t=np.linspace(0,1/4,16) # on définit un tableau de 16 valeurs régulièrement espacées allant de 0 �  1/4 
print('t:',t) # on le fait afficher pour vérifier

#%%
plt.plot(t,ymes) # on fait tracer ymes en fonction de t, sans aucune option de tracé. 

#%%
yreelle=ymes*2/100 # on convertit ymes en yreelle
plt.plot(t,yreelle,'ro',label="y=f(t)") # on fait tracer yreelle en fonction du temps, 
# le tracé étant des ronds (o) rouges (r), avec un label pour y
plt.xlabel("temps") # on place un label sous l'axe des abscisses
plt.ylabel("yreelle") # on place un label �  côté l'axe des ordonnées
plt.grid() # on place une grille
plt.legend() # on fait tracer toutes les légendes et les options de tracé
plt.title("chute libre") # on place un titre au dessus du graphe
plt.show() # on montre l'ensemble
