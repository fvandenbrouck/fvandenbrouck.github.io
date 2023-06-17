#%%
import numpy as np
import matplotlib.pyplot as plt
# le début du script est identique à ce qui précède.

#%%
t=0.066*np.arange(11) # Ici le nombre de mesures est N=11
x=np.array([0.01,0.25,0.57,0.91,1.22,1.54,1.87,2.16,2.49,2.81,3.15])
y=np.array([0.01549,0.3404,0.6812,1.010,1.297,1.559,1.768,1.95,2.08,2.158,2.193])

#%%
plt.figure()
plt.plot(x,y,'ro',label="y=f(x)")
plt.xlabel("xreelle")
plt.ylabel("yreelle")
plt.grid()
plt.legend()
plt.title("Trajectoire de la balle")
plt.show()
#%%


n=np.arange(len(t)-1)
plt.figure()
for i in n :
    plt.arrow(x[i],y[i],x[i+1]-x[i],y[i+1]-y[i],head_width=0.05) #on fait tracer une série de vecteurs
# de position initiale x[i],y[i], de coordonnées x[i+1]-x[i] et y[i+1]-y[i]
# head_width est la largeur de la tête
plt.xlim(0,4)
plt.ylim(0,3)
plt.grid()
plt.title("Vecteurs déplacement")
plt.show()
#%%
m=np.arange(len(t)-2) # comme on travaille dans la suite sur des intervalles i à i+2, il faut couper la boucle à N-2

plt.figure()
for i in m :
    plt.arrow(x[i+1],y[i+1],0.1*(x[i+2]-x[i])/(t[i+2]-t[i]),0.1*(y[i+2]-y[i])/(t[i+2]-t[i]),head_width=0.05)
# tracer une série de vecteurs de position initiale x[i+1],y[i+1], de coordonnées 0.1(x[i+2]-x[i])/(t[i+2]-t[i])
# et idem selon uy.
# Ces longueurs correspondent au déplacement de i à i+2 rapporté à l'intervalle de temps correspondant.
# on multiplie par 0.1 pour que la figure soit lisible.
plt.xlim(0,4)
plt.ylim(0,3)
plt.grid()
plt.show()