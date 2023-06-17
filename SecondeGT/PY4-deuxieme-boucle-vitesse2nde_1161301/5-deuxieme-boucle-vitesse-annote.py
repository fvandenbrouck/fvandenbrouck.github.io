#%%
import numpy as np #ce script est exactement similaire au précédent
#le seul changement se situe dans la définition des flèches que l'on fait tracer. 
# ici, on rapporte la variation de position entre deux points au temps entre deux points
# ce qui est une première manière de définir la vitesse. 
import matplotlib.pyplot as plt
#%%
ymes=np.array([-0,-0.7,-1.5,-2.3,-3.5,-4.5,-5.9,-7.7,-8.8,-10.6,-12.3,-14.2,-16.4,-18.5,-21,-23.5])
yreelle=ymes*2/100
t=np.linspace(0,1/4,16)
n=np.arange(len(t)-1)
x=0*t
#%%
plt.figure()
for i in n :
    plt.arrow(0,yreelle[i],0,(yreelle[i+1]-yreelle[i])/(t[i+1]-t[i]),head_width=0.002)
#on rapporte la variation de position à la durée entre deux positions successives, le vecteurs ainsi 
# tracé correspond à la vitesse
plt.xlim(-0.02,0.02)
plt.ylim(-5,0)
plt.show()
#%%