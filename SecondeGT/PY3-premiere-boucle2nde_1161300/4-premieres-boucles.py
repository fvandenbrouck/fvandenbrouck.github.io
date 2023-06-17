#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
ymes=np.array([-0,-0.7,-1.5,-2.3,-3.5,-4.5,-5.9,-7.7,-8.8,-10.6,-12.3,-14.2,-16.4,-18.5,-21,-23.5])
yreelle=ymes*2/100
t=np.linspace(0,1/4,16)

n=np.arange(len(t)-1)
print('n:',n)
#%%

x=0*t
plt.figure()
for i in n :
    plt.arrow(0,yreelle[i],0,yreelle[i+1]-yreelle[i], fc="k",ec="k",head_width=0.005,head_length=0.01)
plt.ylim(-0.6,0.1)
plt.xlim(-0.1,0.1)
plt.show()
