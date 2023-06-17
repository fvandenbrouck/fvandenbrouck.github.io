#%%
# valeurs à entrer
I2_0 = input('quantité initiale en diiode en mol :')
nI2_0 = float(I2_0)
S2O3_0 = input('quantité initiale en thiosulfate en mol :')
nS2O3_0 = float(S2O3_0)

limitant =''  # initiallisation de la chaine de caractère correspondant au réactif limitant
x=0  # avancement initial
a=0.001 #pas d'avancement
qI2=[nI2_0]
qS2O3=[nS2O3_0]
# écriture de la boucle while
while qI2[-1] > 0 and qS2O3[-1] > 0:
    x=x+a
    qI2.append(nI2_0-x)
    qS2O3.append(nS2O3_0-2*x)
#résolution du problème et affichage du résultat
if qI2[-1]<=0:
    limitant = 'le diiode'
if qS2O3[-1]<=0:
    limitant = 'le thiosulfate'
if qI2[-1]<=0 and qS2O3[-1]<=0 :
    limitant='à la fois le thiosulfate et le diiode : le mélange est stoechiometrique'
#print(limitant)
print('Le réactif limitant est ',limitant,'\n Avancement maximum : ',round(x,2),'mol' )


#%%
#### Définition de la procédure de recherche du réactif limitant ####
#### pour l'équation de réaction : aA A + aB B = aC C + aD D
A = input('quantité initiale de A en mol :')
nA = float(A)
B = input('quantité initiale de B en mol :')
nB = float(B)
a = input('a:')
aA = float(a)
b = input('b :')
aB = float(b)
def react_lim(aA,aB,nA,nB) :
    x=0             # Initialisation de l'avancement
    dx=0.00001      # Incrément d'avancement
    qA=[nA]         # Liste stockant les quantités de matière successives de A
    qB=[nB]         # Idem pour B
    RL=[]           # Liste qui stockera le nom du réactif limitant
    while qA[-1]>0 and qB[-1]>0 :
        x=x+dx
        qA.append(nA-aA*x)
        qB.append(nB-aB*x)
    if qA[-1]<=0 :
        RL.append('A')
    if qB[-1]<=0 :
        RL.append('B')
    return(RL,round(x,2))

print(react_lim(aA,aB,nA,nB))
