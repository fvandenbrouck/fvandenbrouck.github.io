#%%
import numpy as np # on importe numpy
import matplotlib.pyplot as plt # on importe matplotlib.pyplot
#%%
ymes=np.array([-0,-0.7,-1.5,-2.3,-3.5,-4.5,-5.9,-7.7,-8.8,-10.6,-12.3,-14.2,-16.4,-18.5,-21,-23.5])
# on crée un tableau de type array
yreelle=ymes*2/100 # que l'on convertit en unité SI
t=np.linspace(0,1/4,16) # on définit un tableau de 16 valeurs allant de 0 à 1/4 
#%%
n=np.arange(len(t)-1) 
# on définit un tableau de valeurs d'entiers prenant N-1 valeurs 
# avec N la longueur du tableau de temps définit précédemment. 
print('n:',n) # on fait afficher ce tableau, pour vérifier
#%%
x=0*t # on définit une abscisse nulle
plt.figure()
for i in n :
    # on ouvre une boucle for avec un indice qui va jusqu'à N-1 pour ne pas dépasser 
#la longueur de la liste des mesures
    plt.arrow(0,yreelle[i],0,yreelle[i+1]-yreelle[i], fc="k",ec="k",head_width=0.005,head_length=0.01)
#on fait tracer une série de vecteurs
# d'abscisse 0, de position initiale yreelle[i], de coordonnées 0 et la longueur yreelle[i+1]-yreelle[i]
# fc et ec correspondent aux couleurs de flèches (bords et intérieur), head_width est la largeur de la tête, 
# head_length est la longueur de la flèche (toutes ces options ne sont pas nécessaires pour faire tracer des flèches)


plt.ylim(-0.6,0.1) #on définit les limites selon uy du tracé pour pouvoir mieux visualiser les flèches.
plt.xlim(-0.1,0.1) # idem selon ux
plt.show()
#%%