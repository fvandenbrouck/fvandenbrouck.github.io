#%%
# valeurs �  entrer
I2_0 = input('quantite initiale en diiode en mol :')# on ouvre une fenetre qui demande �  l'utilisateur d'entrer
# un nombre correspondant �  la quantité initiale en I2

nI2_0 = float(I2_0) # on convertit cette entree en flottant pour pouvoir la manipuler simplement. 


S2O3_0 = input('quantite initiale en thiosulfate en mol :') # Idem pour le thiosulfate


nS2O3_0 = float(S2O3_0) #idem

limitant =''  # initiallisation de la chaine de caractere correspondant au réactif limitant
x=0  # avancement initial
a=0.001 #pas d'avancement
qI2=[nI2_0] # on crée une liste, contenant pour l'instant un seul element qui correspond �  la valeur initiale de I2


qS2O3=[nS2O3_0] # idem pour le thiosulfate
# écriture de la boucle while
while qI2[-1] > 0 and qS2O3[-1] > 0: # tant que les deux concentrations sont non nulles
    x=x+a # on remplace x par x+a : on affecte �  x la valeur précédente de x+a
    qI2.append(nI2_0-x) # on complète la liste qI2 par la valeur calculée nI2_0-x
    qS2O3.append(nS2O3_0-2*x)#on complète la liste qS2O3 par la valeur calculée nS2O3-2*x
#resolution du probleme et affichage du résultat
    
if qI2[-1]<=0: # si la dernière valeur de la liste qI2 (le -1 permet de sélectionner la dernière valeur de la liste) est nulle
    limitant = 'le diiode' # alors le réactif limitant est diiode
if qS2O3[-1]<=0: # idem dans l'autre cas
    limitant = 'le thiosulfate'
if qI2[-1]<=0 and qS2O3[-1]<=0 :
    limitant='�  la fois le thiosulfate et le diiode : le melange est stoechiometrique'
if qI2[-1]<=0 and qS2O3[-1]<=0 : #on envisage aussi le cas d'un mélange stoechiométrique
    limitant='�  la fois le thiosulfate et le diiode : le melange est stoechiometrique'

#print(limitant)
print('Le reactif limitant est ',limitant,'\n Avancement maximum : ',round(x,2),'mol' )
# on fait afficher le réactif limitant
# round(x,2) sert �  arrondir �  deux chiffres le résultat numérique précédent.
#%%


#### Definition de la procedure de recherche du reactif limitant ####
#### pour l'equation de reaction : aA A + aB B = aC C + aD D
A = input('quantite initiale de A en mol :') # on cree une entree qui demande la quantité initiale de A
nA = float(A) # on convertit l'entrée A en un flottant nA
B = input('quantite initiale de B en mol :') #idem pour B
nB = float(B)
a = input('a:') # on fait entrer le coefficient stoechiométrique de A
aA = float(a) # on le convertit en un flottant
b = input('b :') #idem pour B
aB = float(b)

def react_lim(aA,aB,nA,nB) : # on definit une fonction qui prend 4 arguments et qui va effectuer un certain nombre d'opérations
    x=0             # Initialisation de l'avancement
    dx=0.00001      # Increment d'avancement
    qA=[nA]         # Liste stockant les quantites de matiere successives de A
    qB=[nB]         # Idem pour B
    RL=[]           # Liste qui stockera le nom du réactif limitant
    while qA[-1]>0 and qB[-1]>0 : #comme precedemment, tant que les deux concentrations sont non nulles
        x=x+dx # on remplace x par x+dx
        qA.append(nA-aA*x) #on complete la liste qA par la valeur calculee nA-aA*x
        qB.append(nB-aB*x) #idem pour B
    if qA[-1]<=0 : #si la derniere valeur de qA est nulle
        RL.append('A') #completer la liste du reactif limitant RL par A
    if qB[-1]<=0 : # idem pour B
        RL.append('B')
    return(RL,round(x,2)) # retourner RL et donner la valeur de l'avancement arrondie �  2 chiffres

print(react_lim(aA,aB,nA,nB)) # effectuer la fonction reac_lim pour les grandeurs entrées par l'utilisateur 
#%%
react_lim(1,3,6,5) # test de la fonction definie precedemment pour un certain jeu de valeurs de paramètres