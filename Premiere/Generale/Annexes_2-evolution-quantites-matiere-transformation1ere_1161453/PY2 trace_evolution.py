#%%
import matplotlib.pyplot as plt
#### Tracé de l'évolution des quantités de matière ####
#### pour l'équation de réaction : vA A + vB B = vC C + vD D

def evol_qt(aA,aB,aC,aD,nA,nB,nC,nD) :
    x=0         # Initialisation del'avancement
    dx=0.001     # Incrément d'avancement
    X=[x]       # Liste stockant les valeurs successives d'avancement
    NA=[nA]     # Liste stockant les quantités des matière du réactif A
    NB=[nB]     # Idem pour le réactif B
    NC=[nC]     # Idem pour le produit C
    ND=[nD]     # Idem pour le produit C
    while NA[-1]>0 and NB[-1]>0 :
        x=x+dx
        X.append(x)
        NA.append(nA-aA*x)
        NB.append(nB-aB*x)
        NC.append(nC+aC*x)
        ND.append(nD+aD*x)
    plt.figure(1)
    plt.plot(X,NA,'r-',lw=1,label='nA')
    plt.plot(X,NB,'g-',lw=1,label='nB')
    plt.plot(X,NC,'b-',lw=1,label='nC')
    plt.plot(X,ND,'y-',lw=1,label='nD')
    plt.grid(True)
    plt.xlabel('x (mol)')
    plt.ylabel('n (mol)')
    plt.legend()
    plt.show()

#%%
evol_qt(2,5,1,5,2,3,0,0)
