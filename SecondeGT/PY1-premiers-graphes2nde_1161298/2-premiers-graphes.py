#%%
import numpy as np
import matplotlib.pyplot as plt
ymes=np.array([-0,-0.7,-1.5,-2.3,-3.5,-4.5,-5.9,-7.7,-8.8,-10.6,-12.3,-14.2,-16.4,-18.5,-21,-23.5])
print('ymes:',ymes)
t=np.linspace(0,1/4,16)
print('t:',t)

#%%
plt.figure()
plt.plot(t,ymes)

#%%
yreelle=ymes*2/100
plt.plot(t,yreelle,'ro',label="y=f(t)")
plt.xlabel("temps")
plt.ylabel("yreelle")
plt.grid()
plt.legend()
plt.title("chute libre")
plt.show()
