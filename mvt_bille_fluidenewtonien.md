# Modélisation du mouvement d'une bille dans un fluide newtonien sous l'action du champ de pesanteur terrestre


---
*Groupe de travail chargé de la rénovation des programmes de physique-chimie BCPST*


## Mise en oeuvre d'une capacité numérique spécifique du programme de physique-chimie de BCPST2

Ce document d'accompagnement des nouveaux programmes de physique-chimie en BCPST 2ème année présente une mise en oeuvre possible d'une capacité numérique particulière.

### Partie M.6 - Fluides en écoulement
#### Paragraphe M.6.1 - Description d'un fluide en écoulement 

| Notions et contenus|Capacités exigibles |
|:---|:---|
|Traînée d’une sphère en mouvement rectiligne uniforme dans un fluide newtonien : | **Capacité numérique :** résoudre, à l’aide d’un langage de programmation, l’équation différentielle vérifiée par la vitesse, |
| nombre de Reynolds Re ; coefficient de traînée $C_x$ ; graphe de $C_x$ en fonction du nombre de Reynolds; |en utilisant une modélisation fournie du coefficient de traînée Cx en fonction du nombre de Reynolds, |
| notion d’écoulement laminaire et d’écoulement turbulent. |  dans le cas de la chute d’une bille sphérique dans un fluide newtonien. |






## La chute d'une goutte de pluie

Considérons un nuage qui s'est formé à haute altitude. On suppose qu'une goutte d'eau sphérique de rayon $R = 0,075\,\mathrm{mm}$, qui s'est formée dans la partie inférieure du nuage, chute en direction du sol avec une vitesse initiale nulle. On note $\rho_{e}$ et $\rho_{a}$ les masses volumiques respectives de l'eau et de l'air et $g$ l'accélération de la pesanteur. L'étude est menée dans le référentiel terrestre galiléen suivant un axe $(Oz)$ vertical descendant.

On cherche à déterminer la vitesse terminale de chute de la goutte d'eau, soumise à l'action de la pesanteur terrestre et à l'action de l'air.

L'action mécanique de l'air sur la goutte d'eau est modélisée par une force de traînée $\vec{f}=-\frac{1}{2}\rho_a S C_x v\vec{v}$, où $v$ est la norme de la vitesse $\vec{v}$ de la goutte (par rapport au fluide, l'air, supposé immobile dans le référentiel terrestre), $S$ la "surface projetée" de la balle (autrement appelée *maître-couple*) et $C_x$ le coefficient de traînée qui est fonction du nombre de Reynolds $Re = \frac{v(2R)}{\nu}$ où $\nu$ est la viscosité cinématique de l'air.

### Mise en équation

L'équation du mouvement de la goutte d'eau se déduit de la seconde loi de Newton:

$$
m\frac{\mathrm{d}\vec{v}}{\mathrm{d}t}=-\frac{1}{2}\rho_a S C_x v\vec{v}-m\vec{g}\,,
$$
avec $S=\pi R^2$ et $m=\frac{4}{3}\pi R^3 \rho_e$.

Le mouvement de la goutte En projection sur l'axe $(Oz)$ vertical descendant, on obtient:

$$
 \frac{\mathrm{d} v_z}{\mathrm{d}t}=-\frac{3}{8}\frac{\rho_a}{\rho_e} \frac{C_x}{R} v_z^2+g\,.
$$

On propose de résoudre numériquement cette équation différentielle.

### Modélisation du coefficient de traînée

L'expression retenue pour le coefficient de traînée $C_x$ en fonction du nombre de Reynolds est celle qui figure dans la référence suivante: R. Clift, J.R.  Grace, M.E. Weber. *Bubbles, Drops and Particles*. Academic Press, 1978.


```python
## Importation des bibliothèques utiles
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint
```


```python
# Definition des fonctions
def Cx(Re):
   """ Coefficient de trainee Cx d'une sphere, en fonction du nombre de Reynolds """
   # Conversion de l'argument en tableau numpy np.array 
   # parce que np.piecewise ne fonctionne pas avec un argument scalaire.
   if not isinstance(Re, np.ndarray):
      Re = np.array([Re])
   # Selon le modèle de Clift, Grace & Weber
   fa = lambda x: 3./16 + 24./x
   fb = lambda x: 24./x * (1.+0.1315*x**(0.82-0.05*np.log10(x)))
   fc = lambda x: 24./x * (1+0.1935*x**0.6305)
   fd = lambda x: 10.**(np.polyval([0.1558, -1.1242, 1.6435],np.log10(x)))
   fe = lambda x: 10.**(np.polyval([0.1049, -0.9295, 2.5558, -2.4571],np.log10(x)))
   ff = lambda x: 10.**(np.polyval([-0.0636, 0.6370, -1.9181],np.log10(x)))
   fg = lambda x: 10.**(np.polyval([-0.1546, 1.5809, -4.3390],np.log10(x)))
   fh = lambda x: 29.78 -5.3*np.log10(x)
   fi = lambda x: 0.1*np.log10(x) -0.49
   fj = lambda x: 0.19-8.e4/x
   Cx = np.piecewise(Re, [Re<0.01, (0.01 <= Re)*(Re <20), (20 <= Re)*(Re <260), \
                          (260 <= Re)*(Re < 1500), (1500 <= Re)*(Re < 1.2e4), \
                          (1.2e4 <= Re)*(Re < 4.4e4), (4.4e4 <= Re)*(Re < 3.38e5), \
                          (3.38e5 <= Re)*(Re < 4.e5), (4.e5 <= Re)*(Re < 1.e6), \
                          1.e6 <= Re], \
                         [fa, fb, fc, fd, fe, ff, fg, fh, fi, fj])
   return Cx
```


```python
    Re = np.logspace(-2,8,100)
    plt.loglog(Re, Cx(Re),'*-')
    plt.xlabel('Re')
    plt.ylabel('C_x')
    plt.title('Coefficient de traînée en fonction du nombre de Reynolds')
    plt.show()
```


    
![png](mvt_bille_fluidenewtonien_files/mvt_bille_fluidenewtonien_7_0.png)
    


### Mise en forme des données

On dispose tout d'abord d'un ensemble de données numériques constantes


```python
  # Donnees physiques
  rhoa = 1.2 # masse volumique de l’air @ 25 °C (kg/m^3)
  rhoe = 1.0e3 # masse volumique de l'eau (kg/m^3)
  nu = 1.6e-5 # viscosite cinématique de l’air @ 25 °C (m^2/s)
  g = 9.8 # accélération de la pesanteur (m/s^2)
  R = 7.5e-5 # rayon de la goutte d'eau (m)
```

Afin de représenter la trajectoire, il faut disposer des valeurs de la coordonnée verticale $z(t)$ qu'on déduit de la valeur de la composante verticale $v_z(t)$ du vecteur vitesse.
On choisit de représenter ces deux quantités dans une liste:
$$
Z = [z,vz]
$$
Cette liste va être nécessaire pour l'appel à la fonction ```odeint```.

### Résolution numérique de l'équation du mouvement


```python
def chutelibre1D(Z, t, rhoa, rhoe, nu, g, R):
    """ système différentiel pour la chute libre 1D avec frottements """
    vz = Z[1]
    vzdot = 0.0
    if np.abs(vz)>0.:
      Re = np.abs(vz*2*R/nu)
      vzdot += g-3./8.*(rhoa/rhoe)*Cx(Re)*(vz**2)/R
    # on renvoie les valeurs des dérivées de z et vz
    return [vz,vzdot]
```

### Détermination de la vitesse terminale de chute


```python
## conditions initiales
Z0 = 0.
VZ0 = 0.00001 # on est contraint de donner une valeur initiale strictement positive pour que la solution s'amorce.
 
```


```python
t = np.linspace(0.,1.,1000) # intervalle temporel
init_cond = [Z0,VZ0] # conditions initiales
sol = scint.odeint(chutelibre1D, init_cond, t, args=(rhoa,rhoe, nu, g, R))
Z,VZ, = sol.transpose()
```


```python
# on calcule le nombre de Reynolds à tout instant au cours de la chute
Re = VZ*2*R/nu
```


```python
# représentation graphique
figure = plt.figure(figsize=(14, 10))
# On trace un ensemble de graphes rangés selon 1 lignes et 2 colonnes
plt.subplot(1,2,1)
plt.plot(t,VZ)
plt.xlabel('t (s)')
plt.ylabel('$v_z$ (m/s)')

plt.subplot(1,2,2)
plt.plot(t,Re)
plt.xlabel('t (s)')
plt.ylabel('Re')

plt.show()
```


    
![png](mvt_bille_fluidenewtonien_files/mvt_bille_fluidenewtonien_17_0.png)
    



```python
# Extraction de la valeur terminale de chute de la goutte
vterm = np.max(VZ)
print("Valeur de la vitesse terminale = %.2e m/s" % vterm)
```

    Valeur de la vitesse terminale = 4.52e-01 m/s


Il est intéressant de constater que le nombre de Reynolds évolue au cours du mouvement. Il devient assez vite supérieur à l'unité. On en conclut que la modélisation de l'action mécanique de l'air par la force de Stokes $\vec{f}=-6\pi\eta R\vec{v}$, où $\eta$ est la viscosité dynamique du fluide, acceptable seulement pour $\mathrm{Re}<1$, n'est pas cohérente. Bien que très couramment utilisée pour résoudre le problème posé, elle livre une valeur de la vitesse terminale (0,61 m/s) éloignée de celle qui vient d'être calculée.

## Modélisation du mouvement d'une balle de golf dans le champ de pesanteur terrestre

### Mise en équation

Une situation bidimensionnelle, plus complexe, est maintenant proposée.

On étudie le mouvement d'une balle de golf, modélisée par une boule de rayon $R$ et de masse $m$, soumise à l'action du champ de pesanteur terrestre, supposé uniforme et noté $\vec{g}$, et à l'action de l'air.
L'action mécanique de l'air sur la balle de golf est modélisée, comme précédemment, par une force de traînée $\vec{f}=-\frac{1}{2}\rho S C_x v\vec{v}$, où $v$ est la norme de la vitesse $\vec{v}$ de la balle (par rapport au fluide, l'air, supposé immobile dans le référentiel terrestre galiléen), $\rho$ la masse volumique de l'air, $S$ la "surface projetée" de la balle (autrement appelée *maître-couple*) et $C_x$ le coefficient de traînée qui est fonction du nombre de Reynolds $Re = \frac{v(2R)}{\nu}$ où $\nu$ est la viscosité cinématique de l'air.

L'équation différentielle qui régit le mouvement de la balle est donnée par la seconde loi de Newton:

$$
m\frac{\mathrm{d}\vec{v}}{\mathrm{d}t}=-\frac{1}{2}\rho S C_x v\vec{v}-m\vec{g}\,.
$$

### Résolution numérique de l'équation du mouvement

On commence par projeter l'équation du mouvement sur l'axe horizontal ($Ox$) et vertical ($Oz$):

$$
m\frac{\mathrm{d}v_x}{\mathrm{d} t}=-\frac{1}{2}\rho S C_x vv_x\,,
$$
$$
m\frac{\mathrm{d}v_z}{\mathrm{d} t}=-\frac{1}{2}\rho S C_x vv_z-mg\,.
$$

#### Mise en forme des données

On dispose tout d'abord d'un ensemble de données numériques constantes


```python
  # Donnees physiques
  rho = 1.2 # masse volumique de l’air @ 25 °C (kg/m^3)
  nu = 1.6e-5 # viscosite cinématique de l’air @ 25 °C (m^2/s)
  g = 9.8 # accélération de la pesanteur (m/s^2)
  R = 0.0225 # rayon de la balle de golf (m)
  m = 0.0450 # masse d'une balle de golf (kg)
  S = np.pi*R**2 # maître-couple (m^2)
```

Afin de représenter la trajectoire, il faut disposer des valeurs des coordonnées horizontale $x(t)$ et verticale $z(t)$ qu'on déduit des valeurs des composantes horizontale $v_x(t)$ et verticale $v_z(t)$ du vecteur vitesse.
On choisit de représenter ces quatre quantités dans une liste:
$$
X = [x,vx,z,vz]
$$

On a tout d'abord besoin d'une fonction qui calcule la norme de la vitesse à partir des composantes $v_x$ et $v_z$


```python
def norme(vx,vz):
  return(np.sqrt(vx**2+vz**2))
```

On doit maintenant définir une fonction qui renvoit les valeurs instantanées de $\dot{x}(t)$, $\dot{v_x(t)}$, $\dot{z}(t)$ et $\dot{v_z(t)}$. Cette fonction définit le système différentiel qui sera résolu par ```odeint```.

 


```python
def chutelibre(X, t, rho, nu, g, R, m, S):
    """ système différentiel pour la chute libre 2D avec frottements """
    vx = X[1]
    z = X[2]
    vz = X[3]
    v = norme(vx,vz)
    # Si la vitesse verticale est orientée vers le bas et si l'altitude est proche de 0 : le projectile a atteint le sol et doit s'immobiliser
    if (vz <=0) and (z<=0.001):
      vxdot = 0.0
      vzdot = 0.0   
    elif np.abs(v) >0:
        # Reynolds
        Re = np.abs(v*2*R/nu)
        vxdot = -1./2.*rho*S*Cx(Re)*v*vx /m
        vzdot = -g-1./2.*rho*S*Cx(Re)*v*vz /m
    # on renvoie les valeurs des dérivées de x, vx, z et vz
    return [vx,vxdot,vz,vzdot]
```

#### Calcul numérique de la trajectoire


```python
## Conditions initiales
V0 = 40.
angle = 40.
```


```python
  t = np.linspace(0.,10.,1000) # intervalle temporel
  init_cond = [0.,V0*np.cos(angle/180*np.pi),0.,V0*np.sin(angle/180*np.pi)] # conditions initiales
  sol = scint.odeint(chutelibre, init_cond, t, args=(rho, nu, g, R, m, S))
  X,VX,Z,VZ, = sol.transpose()
```


```python
# On retire les points d'altitude négative en utilisant un masque
# le masque est un tableau de même taille que Z dont chaque élément
# vaut True ou False selon que l'élément correspondant de Z vérifie ou pas la condition
masque = Z >= 0
# On utilise ce masque pour ne garder dans les tableaux X, Z, VX, VZ et t que les éléments qui correspondent à la valeur
# True de l'élément correspondant du masque
X = X[masque]
Z = Z[masque]
VX = VX[masque]
VZ = VZ[masque]
t = t[masque]
# on calcule le nombre de Reynolds en chaque point de la trajectoire
Reynolds = norme(VX,VZ)*2*R/nu
```


```python
# on calcule la trajectoire ballistique sans frottements (Galilée)
abscisse = np.linspace(0.,V0**2/g*np.sin(2*angle/180*np.pi),200)
altitude = -0.5*g*(abscisse/(V0*np.cos(angle/180*np.pi)))**2+abscisse*np.tan(angle/180*np.pi)
```


```python
# Représentations graphiques
fig = plt.figure(figsize=(14, 10))
# On trace un ensemble de graphes rangés selon 2 lignes et 2 colonnes
plt.subplot(2,2,1)
plt.plot(X,Z,label='avec frottements')
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.plot(abscisse,altitude,'r-',label='tir balistique sans frottements')
# on ne prend qu'un intervalle limité en abscisse pour visualiser le début des trajectoires
plt.axis([0.,80.,0.,35.])
plt.legend()

plt.subplot(2,2,2)
plt.plot(X,Z,label='avec frottements')
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.plot(abscisse,altitude,'r-',label='tir balistique sans frottements')
plt.legend()

plt.subplot(2,2,3)
plt.semilogy(t,Reynolds)
plt.xlabel('t (s)')
plt.ylabel('Nombre de Reynolds')

fig.tight_layout()
plt.show()
```


    
![png](mvt_bille_fluidenewtonien_files/mvt_bille_fluidenewtonien_35_0.png)
    


Il est instructif de comparer cette trajectoire à la trajectoire parabolique telle qu'elle fut établie par Galilée. On constate qu'au début du mouvement, les deux trajectoires sont proches. La trajectoire avec frottements n'est pas symétrique par rapport au sommet, comme c'est le cas pour la trajectoire parabolique caractéristique d'un tir balistique sans frottements. L'action mécanique de l'air conduit à une diminution progressive de la vitesse horizontale: il en résulte que le mouvement se rapproche de plus en plus d'une chute verticale. La trajectoire modélisée n'est pas sans rappeler les trajectoires imaginées par le mathématicien italien Niccolo Tartaglia qui esquisse en 1537 dans son manuscrit *La nova Scientia* les trajectoires
des boulets de canon (voir ci-après).



![Tartaglia.jpeg](data:image/jpeg;base64,/9j/2wBDAAQDAwQDAwQEAwQFBAQFBgoHBgYGBg0JCggKDw0QEA8NDw4RExgUERIXEg4PFRwVFxkZGxsbEBQdHx0aHxgaGxr/2wBDAQQFBQYFBgwHBwwaEQ8RGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhoaGhr/wAARCADLAQADASIAAhEBAxEB/8QAHAAAAQUBAQEAAAAAAAAAAAAAAwIEBQYHAQAI/8QAQxAAAgECBQIFAQUFBgUDBQEAAQIDBBEABQYSITFBBxMiUWFxFDKBkaEVI0Kx0RZSYsHh8AgkMzTxJUNTF2NygoOi/8QAGgEAAwEBAQEAAAAAAAAAAAAAAQIDAAQFBv/EACoRAAICAQMEAQMFAQEAAAAAAAABAhEhEjFBAyJRYRMyUoFxkaGx8ELR/9oADAMBAAIRAxEAPwD7KYMWF2Fhftxj0od438sXYD0E25/HtjzOVYFjtFrduvxgM1N5oVpHYsnq9LEAH3t3x84j3hMkrOCImIIvYA8BgOhwpFfbuQMyN0UOB398NarUFLSzvSs08tTHT+eI44i5ZORwQLXJB4vc2PtgT57S7wH8yNnaNULRkAlxdbfBsRf3GKUyVofKHUsAJGN+NxHt8e2FeVIwUtGCwHUtcW+nviN/bkCTRRhJJSZ/JctE1o3N7buOAbcfUYO+oKOnmlhZ2UxzCKQlHUKxJsL24+vQ++CkwWvI4k883MZDFR2sdv0ws71ULYD1deOmArndOZLPMuwOELOjA3IPx8G5NgMekziljmeOZ/3iSCN1EZuNx4vwfcXPzjUwJpHZlkLgKHve9uOfqMJAqXZC0ShLg23cj34thDZxABN9jKvOvmWic+WWKdbE4K1dAyTbZoisK7pDvFkHU3P05wKY2Hyc8uWTefUt+VKt3Hb9MFMUnpCmTcet7c8/pgYrqOJVUVUe4ruBaUWK/wB76cH8sJgzJmjk/aCJSSQqrsqz+YpVrhTe3e2DT3BgUzAMFaZS4NlBO3njjCQKtN5a7uALqliDz79OnOOtVwxly00VxwqfPfn26Y9JXRxGQiSMCNtpUvY347e3OAE8kNR5p8xiydGAH48/ywOQTWkLtsC8Bv5YWtRKtRURyPTiLaHp9053kWu+5doAAPSxbg9sJSppWCTRyR+UUDAhvSQRcMPb8cYNCSssqXhYFebMF5/I4A89VGT1VQAefvW4J4GJEVULME8+Lc52hL2NyOgOBw1EU9FG6qqbkJ2NZWTm3PJANxh03RNjE1dQLrzcr0I4/rhKTzBVADCxF9y24/P2w9kaBygD9Tzttx7/AO/jAyAygh1IPC27cX4GFoJH1eayRJ5FP5P2uS/kpNKFDbSNxPPQA3tiPkzyrMctT5kDRNDcRix8uRGIYkjtb+WJcrEXKTxxyD1KBtubWBPUd8AFJSlDekg8sxkWEINxbpa36YoqrKEd3uMlzKetikgqISKaYPCzI/K3sFt0PIIP5Y41bVmmMKoUaKNFuGBYML9Rf/D+uHZgpmIYwwLtKvG23pwB37gAC39MEkgooFikaJFQWt+7PG826AHrfvhrT4NTGCZzWGsp4RAfJkdV8zzLH1RlgbHrYgiwIwiTPq4GoWGktPHI0UX7wWl2qGJPsLE2+mJBqOjYLKlNCXjCtE4AJsAdpB97E/POHLUkDFpDEjOBY3XqCP6Y3b4BT8kcmo6sUlXMKZ3MaXRFYetgobb+R6/GDx5zPmMctIYWhaWB/LdWH3rcfPcHBTS0xVT9mjVQuxfRcbeliOmAxrSo3/Lxqk0QS6op6W2j/wDzxgPT4D3A6jPZJKamYmqHk+RMVjcIXU2R1JvzZupOJibO2hhimenma9V9mljUAtGb/e68jpyPcYiZaDLhE6PSR2UCOwcrdSQbEj598eE2T1Ua0qoksMExcqZGDK/PU9T3+MZ6WZWiWqc0kE7LDSNIhdAsikEurc3t9bi3W+I9s5nvZ6OojBEnEgUG6KDttfra59uMLSDLvNjMdNeVQEB89rW3bgT7+r+eHNRFl9XumroN7MwaysRa6bfjthe1cBySlrOGbcLcHjjHJGWMFyFVR1IPAwpfVcc/HNj/AK4HJ7iIOOhDN1HvbHOdJHVtPSyTRTSwuZGAp2ZG4ANzZxccC/B63PzhpSrk9XtpaZGmMaKm1XZtvlsVAuehBuDzhzWZdBVyS3EsL7kZX3eYl1IJ9B4HIthUOVCmINLWtGn7zYpiG5N5v1PsenGKp+yTWdj1sn8ymVpUV2kTyW+0G7vGDYXHWwBvfsOcEqFoaxj9pY3qIxAWimNpFBJ2kj2uevY4A+VxSSmRagxReYrhFp0AX0sGBPzuJv8AHfDKgo6COhppoKtqiASLttAFZr7Ba3YkqD9CeLHDKvIrvwPDlWWERpD9oYzl2VzLcKeSw3fIYi3Nx9MLmgpwKypkWqvKvqWBizfeFyoF+th9bYDFlkbCIVC0sKLUlvsypcG4a8YJtc83uP8ALCsvyoUVTTvHUrvSMowEYBI6fQjkH6gnvjN+zJPwGeCCSKStlnqhTu/2iwsfSEBt03Lx2/y4weGnp0JeN5nL0iqt7X2dja3X64ZfZo4UpqdqmIzUoJ3dxETcjbf7pAsf0wOKhMtK0VS0bFHD0k6gkWsLg3N+x46WPGBeNwpZ2CNR0aAUu+YOEk2ybR0K7WANrWtz7Xvgn2cUe2QSs1OYvKkJXcet1Ycc/e6fOEpQMsZVnhiieRyu1W9Ibdfqfbt0xwUMhUGlmjQSRkGxYjrdQOeg4/1wb9gr0Bjho6iFhHLUNFMioXMQK2twbEf4fwtjs1KkcTmkU1E7jy23KFv6gQxHcqOnuOMDOW1SW8swpuZtoDNYdT/O3HTrjH/EHWOoc61NUaF8IpwmaUcnmZ7nobdT5PC3RQwH/XuG9NiR91eblXScnhiNqK2J/wASPGvTnh5NS5RWGqzbVG5BBk2Xqry3bhBI5uIt3Ho9TG/3T1xVqLM/HHVlOkmT6XyTRFC4AR89k8ydlIN7RbSQBfoUHQdcX3QPg5kfhuDLlTnMs7qJXkqc9rbPVys6+ogkEoCSeFNz1Ysb4uUcTy07wUs0iSxTi5WX8Tzbpyeo5/DD6oxxEXTJ5ZjI8O/FfM5ZpK3xioqSqJPlwUGUsYrKwBuBstYgjgHoPfHarwv8SFimkpPHKeslIdZVmyPcrXP3Noc8E7gRbGztl08kcknojZ9zXSbabkDoSP7y4JQ0lZBPIauQtHKCTaQNtchRcEDjm5/HG+R74/gC6aMPqqHx50zUQ1NDJpfXcBVt8b0Ao5FBNwCW2Ek3P8Vut+uAP435to5aSl8adA5ppGOSTzYcwoL1dLyLWZQTYi5NlckcenG6U8OYNDD9peSOVAEdZJkcEWb1FgB6gbfUY6aasdFp5A6Usg2y/vFO4WI59+1/fB1rlI2jwyJyHNcn1BlwzPIK+nzTK5Y1ngq4AWW5PRh1Vj3DWPxh7HSpEHkErPscOH8tvSpJuevU+/xjINTeB9XSZq+rPBWuOjdSM5M9NHMEoa64szFCCsb3JuCrIT1VT6sF8LvF+fVdfJpbWELZBryk8yB6NkCx1DxjcdgP3JeSfLJIK+pCwPDONq4sVSp1I1x4lFRVEPGl23g+oFRYWIsfdTf3GHYh3RUyxnbKHBvuYKTc3+v0/lgRjmMCyl7TIxJT0+sDgc8263/THaxJvMV/NeKPzNr2RfULjrz0tfkYkVESUshgpRTqInhnJALkDaN23pe45HB/ywOWicIbJ6yjICrv1JBFj+eHFX58BcxmWcJtIGwE9Rdb9TcYS81YWtJdhGb2MYuw5v06djfGMMpopBEZKhpnkKm6wb+GDrZh0sLCzficGMJq6aJI3k5j2OCWXcQTfn2PPq+MLimq3RgoZUO/kRgkC/DWv0Nv1GGOfaqotN5XLmOoayDJ6Bdgkmq1Plxk7gASBfkgdcHIMCxls1SIamKdzuVD/wBRvZevbkqfz+uCTUG9jBMGlZ90UbMu8ji4Ldh0a1/xxD0Wuslz6fKpcurY6qorIZp8sSSCWmaqiQBZGRJACVBZfVa3PHGJ2LMqmJpWEAVjHvB2tYgXFwL9enA55PXBepAVMDWmSjTL1oIom82TySJJCjLxxztNz6fzw0r6vMqKCdnp1ZEK32OS3JtcArzyffFjqamNKOeaSRFWOMtvXkjg3Px3464gKuSasypftVDJD50Ubs5qF5uBYlCbjnthY5GeC2KoPLWJte9rce2ETJuQq19rixHQ/TC2BvzwL+3Q4azLK6OIJVRypA3DcB7Ht+XGORHUMKzLquV/MhkV4o5VaGNAyMOADuYEhuhNrDDamoK5YgzOpkjk9B85juUMQL/htPPX64aT0BOYVQqKeZI5yjSPDu8uT0FLix9gtwe4w+E84oo5YY5d/nbHUwn1Jt9r3625GLqyDq8jqDL5o8wjdKsz0+3ayySEkizdunUj63PtgEUFRRzR+d5kk49YCOZPOCMwHNuG2N0PW1scNZXGGXy42BvuBeC1xe9hz1sRx3sceFZXyJINjnypNvleQbyi5Fwb8X63+n1wcgdBDFWzVE8kDM1M8qywkSbWVRa42kXW/qGOxUFWskahp41QMoYzbzt3Hn68g/hglbTTSPGKcgRMSk28G4PY3vcd8CFVVU8JZxHMFWRCdrAl19x8j2GBkOEDmhljYyViPFGI1BmO0kclBfjj0tfuMLgSYbYbmOUlZJY1K2UNwze3UE8e+HCTsaiKMKqq6ixJIYkfJ/L8cNjWtPUxRu/lyQzmNtjsAw+7c3Fj1HH9MHPgGFmxztqZIkjYXffIHvtO5SfST8W7fOALDVR+SvlyrFHYAAqCBde30vjrVzpJJCiWKPbkEcAjrcWI5vineJ3ijR+Gekpc9qKb7XW1Eopcrpl3WmqWDbVe3IVdtzt68AcsMZJtpJBbSVtlX8UNb6lmzWi8OPD+SVNYZvGZKqsCj/0uhJsZzbo+29rWIBFvUy4veg9BUHh7p2iyLTtL5NLTuv2icQgS1kp+9UO3diee9gbDpzV/CDSVVorJqnNdTlK/XWo5Erc8qpZLyKzMdlOD0CpxwON3HQLjS0r3e9o0YKBJYS3svv8AGHnjtSJwzmTBn7R5MHokkmBKyblBsBft82HOCSmrSxgWRyrEqu3g8Ejp/vthJqbOY1KxrE9mCuLWv903HbcCe/TBFzEMsYaEK4vu2SXBNmvb36YlT8FbXkTJEtOZIR5ipKQygesqWYjv/Df545w0aSshWRwsm2JWR18s2LLt9Q7ni/Av0w/WsT7UoWGPdMNrESqT6QeD8ex+uA02Z+ZHDMY32udpLSLuVuwPPfmxwafgV15G0xqJWX7RESF2SCykFWV+l+hBF8KFRNcySwOrBdvRrcEHp0uLnn/xh3R1m67XMyyIz+UxF1II9IPS1iMEjrdyl3D7GC7VBBaLqCPnkYP4B+RsWlZKd/LJd7q4AI9QNrD+fOMn8XvCRPETKv2jl0UdJrKgh3UFdC3lNUFCSIJG4P8ACNjk3jYi3pJGNk+3oHdI1Z0WwNu/qtfr9PzwP7WFk9TgxXZrnjatgRc37C+DGTi7QJJSw2ZR4M+JlTr/AE5XUGoy1Nq3IJBFmkLR+VI4DWEjL2NwUcdmB7EY0mWpRpNrFD+9CSqWuNpawPT3tjHPESaPws8Rcp8T6SL7NklbIMo1dGkZYvFIbw1NvcFRcjrtXqWONqlqolAsGeIjcsi9LHkEe45B/EYea/6S3Fg+G9gD1ZjrvXK2xSY2S4HWxB6Xvza2FVFd5VQQ0cTFLrbfbjaSL/PHIwqOsDs6mG6Km/zNw2gA2JPtz1/njOMw8S63VtbV5H4MUNPneYQN5VVqKsJGU0L/AMW1rXqJFF/QvF+pOFScuBm0uSx57rTJNF0E+Yaqzany6lu6UyzSBpJ5AeVhQDc5NxYAccXtioVKZ540acz+kzTLZdF6UzSmRKFqzbJmU8ayKzO1Ow8uBWCkAli3Q2scSukdA0eTZoM81TmFRrHWEg8mTOq+Ij7OOT5dPF92GMWNio3Hkkjpi8vmUG4PIxF7kkofn+mGb0/SsipNrLKJo7RGXaOzWpr6StzLN8zroZBUVmbTxyTyKCGCqQAEHqb7vDdT0xe2rdrosG4BkVwxI5uL2/JeuPSyQh4yNiIwAJKnhCO3Hxa3scIkqAtSyuI2jSNTu3eqxuCRweOALfXCSdux4qlQF6l4GaZpZZKWYjyniS5hb+6V7j5wHMKmGejWNZ2eWMAWaM7m79bfyxIiVpYnMA2gKpQi6XH07Wt16YKzPK5bcRa20q9rc9fywNSXAaJNgwNrEEDmx6YG72V1YfFuh5wr1KdwuQ1iObkYS5HltY34JFjf574gdBAVWZzGSaiEa0UjwCSOYyqTe9rFe3T5BBw7XMrTwRyxQI7U4mSRZV27jxtAPqA5HPPXC6nMIIXMkyR70T723e5FxztAJtc84RHmVOE8ySOOMU8W4Ex7jtJsNvpuORewxVZWxF2nuIizZXjWbyI7FFkRkkHPUH9Rb8b4O2YMI53ipdxRo5Y0SVdxRwCSeeLG/HfDZsyp5qSrGWRwTzQw+dFFIpSNr3YC9r2JXsDzhwlTROtVK/kCBQqSeg/dAuN3p6Wa9uQL41LwC35PU9c0kgcyPIjysVYbLBCAVU8/JF/jC4szjmCAQyLcc7gPQQL82PX+hwg1VFDKw20aPHYm6WIvwCbCwHTn5wpHSojWHyIoJ6d0LCMblPpJsDYdbkXwaXKNbXJ16o1NXShEewlkRyygFXA9Nz7XBHzfCv2gzvH6WCAurIUsyuCLE89LHvhMBoTCKtljpidrFmsCjk9Ljvc4HM+XRSFpIYnlkBkUbrMwFrsBf3tjY8Gth4650krFqZEQQ3dWNtvlBdxJYm1wLm/AsMYZpjzPGDxHHiRXJMujdPSPSaQgliG2rnU2qKxk6AA32ni5VP7hwbxlzX+1NZlvhToeWng1BqSAPmlcgYLlmUCzOWPYuqhQp5INv4xjYcqyLJslymh09k0EcOX5bTLHDT7rlEUW3EjnceST3LH3xVVBXyyTuTrhEm1QJqpoY1vLGwB82OykEA3v/vnDSato65UULIYpJdm7yigV0P3XBHv78HCZ6OhqKxXMCmSePcJVlJX0cLwDboewwuOKkmMqiKR5C3mOFZyPvbue3JvxiVIrbOCuXyFfLxHKAOWmBDMBtJvxyQp/S2HFZUCFJFhETVCR+YimO1xe5P6Hp3wBKSjpoDFt8lVB3DzTdQQVvzz0PX4+MFWmp5Z0upZ/LEQ2yG+0dj84zoCsOGpKaTYiokyoZIwIh6lPJIsOp5vbAYKmhsDHBEsbBXH7oAG97duvB+mIXPdU6eyKKOq1Jn9Bp8wXRErsyjgZgrWtsLXPK9he2KsfGbwrFLd/EPIWhCAFhmO7jduHO3g34/G3XDKDfkVzSNCH2QSsybIxTqrXSOwZX4545BsMcT7K5mQ08K+Q33SA10tuDfTkH4xU8k8UvDrNiYcj1zp+tmkiWMRftiIOw7GzkG4viy01LGI46yikaqh2gLIJRLGy9CQUuDwPe18Zqtwp3sImkpKimnSIxxytCXVlAFlPN7//AK/pgay0E8c4nhRIzIYmLgC/HW9+hHI+mAzy0AnkcSyBqa4YMnHue3s1uPfCZaWKGmklqapkp4VDTSTyIkcSKBcsxAAFl7kd8GvYLI/VeT5ZrHTme6er6aOrgzOilp5TuBKuY7owueGB2sCO4vik6T8RqCj8GtD5xn9FWZhm1flsdNTZZl8Jnqq+ojUwv5apwqnZcs1lUHnnjDin1hmOuaynpfCzZT5azpGdWZlSk0l1JUrRw2U1L9fWxWIW6tjPMlyGop9D6WqdJUmbVc0+qsxjzevpaYylaGKoqozHURwyRMULFW8qIqu7cbbrXsoqqbJOTu0XdNDZ/wCItSo8YHXKMmuTTaRyertE3Q2rahTunNrExiyc40WGiy7IqCjpclyyOjoqVGjp6eiVYo4SFJsB0uQLe/c4iNJ5THl+nMpajSvp1iVGWlzGVnlRQzjYXku+2zekP6gu0Mbg2mkeOSd9lUqkzFinlLe6nn/z84STt1eB4qldZHapFIzSIAxDLIbP902vY88e/tzgLfZXk2JaRiu8Ddfct7db9L2/PAYKfyJmla73Xa3pUBha3qw2MkKM9I1VGkhF7bQBYngWva3p6YSvY1jxxHZAIySuyMBXtZb8X5w1oysgaWnjNjLLJEwZthAkYW4PP+uOeSzyArLGJFUrbywSfUCLm/Ntv6XwiihpaqKCpFSj04VgCsfVT0PxYgfXG0+w36H8MuwRxU8Yjj+4nU/NvzuMBTMJpFBY7RyLobh7Gxt/T+WCmBTT08SyKrI1y6xgBj727Hob++ACjjnjkLTR2uxsIuVUg9r9bnn3wKRrZbBazFhwR79D+GEk7RckqotY7b4VY/wmzL0B5/1wizqLbt1xyoFrY5zpI2ooKKpV438pW8oAhLLIFvf7/UC9yMIjSjgqImieaOVFJQCQ+om7FbE2PU8YBmGRUtZmEVVIEiqac2Evl7y0JBvE3PKkkn4Ivji5aMrpKQqxrJIHjiukYu0d+OeSBycVW25FvOw6EcNM8UQVws6NTho72AA3AHnjobH4tggoqeGmlghlkKSrscmVgwsoH3h07YAMtamlkkWqiQGYTMiQm5Nx3v164ark8iGoEM9PI00hmULDsDHzCQSQevq5t3ANsFL2C/Q/bLIAnkr5ysA6j996gGtfk9egIwcUewlRNKqMFLNHId5ZWvYEcEG5BwzOWMIapGalaOR0aC28BSLm7WP3r88cY5JQzytHsenVFa5O5l9RUC5+pF/ocbPk34HdPRJDFUrFUVDGU8kqAV68D8++IDWWocs8PtKV2oc/rZDT5XH5yRPtDTyAARQJZeC77Rx7nsDaRmy8RzxAxqziUmIJvYBLgkHngk9BzfnGXHKU8VfEOJJGjrtE6Aq2Dh2cw12d7Su0Ek7o6ddoP+NiOcPFW7bwJKVKksj3wb8PMwyLI8z1JraWVNa6qYVWa7trfZIuGipgCeNg7DpcD+AY1BKZqWrhlG519QkLKotxbdx06AWF+mG/2Cu3H1Jd1Utd2Ys20Am9u5B5+b45DDVLVzpOGWKSK28SHcGJ+8L8Aj49sLObbtsMIpKkgsGWpSpAFrWMcRb92IlUEMp4t+I/LGXeNGmM+zCm0rU6JGePX0+dRXjyiqmhVKR0InM3lum4AqhU7g/JCnk40Ssmjyhaisrqg0lNTL5k9bVOBEkC8s8klwF2jsfw6jGWUeoNZ+NUAqtI1Fdo3QkkLRpnEJENfmrjvShwfIh6/vSNzc2tikbT1XgSVNaayPItXTZRR5fpXTuX5jrHWlDSLBmFNW5glqFDutJmNcNyIxBuIwXkPA+cSMHhdqHU6GTxC1vXTBjubJdPb8toY2AsFaVbVEw46llue3bFoyDS8Ok8miyXTOXw5ZlMSNtgjnBaSUhryOzAl5CQpLtcm/XjEqi1khJiEqhmLlWmD+naLjn2IYfjjOVfSZR+4q2U+DGicqFsr01kKhHN5JctjqJLEDdeSXcxJI5uefg4tlDpmno0kgSCi+xyQGJqdKYLGCeoC22heOluMEjp8xFmdmJjlAKgrsZCfVxfpbp0t7HA2bMFeoELSmON2eA7YwJF4svP1Nj8WOE1TfI+mPggM08OtMaiQUmdZJp7MVh4MVRlsUm0C2zgi46E+3JtinU3/DbpHL6s12kVzHS9Y4DK+R5xU0qo1911j3MhFx0sQfbGgZzqbLdIUj55rOtpssy2KpaP7VOFUy7rhUVVu0jEAWVQSfbFPU6516oSIV2gtGtFtp4Y0WPOq1exZ+lGjDoovLYdVJ4qnOt8E2oXsQud5/W6b1Oum8t1L/b7O6r9x+ycwSnpZhJ5bWRq+Py4VkIIPlMjyMOQotuwDJtFy+J8cU+v8+y7PYqHbNHpKgm/5CkJ6SVLElqxlZWXcT5W5WABscTup/C85tQaSyLKao6ZyHI6010GX0FKAzVC/ccybuo3PcsGLFmJ9QBww0l4bZtk+lKPIMxqKbLKjK6qoqNN5vkdI8M+XrI7SmJ1kLCRLuU8oko6j1DcAcPa04eSdSvJpVHlstO8XkLCiQSWihThVUMpQKvAW1iLDpjO/CCkrW0RVFNkWzVWd+X5kjAhBXSghhax57YsOldVZ2M4GRaqy+my/UQLVbR0ys1PmkAChqimJ5Uhrb4Wu8ZbutmMF4PV9aMk1BSxOkopNa55DLaEjber38j/APoTx74V2kx7TaLuuVTrVQOs0TRxzXBadtwQobjkcnd29sKWlqzXpNG8JptzcBySQVAJt73Fv1waOWvCmNhIDtZU3U3qLgn1dQDxbjvhzvqUoEniR2k32bdCVIS1+n4jnr+WJZKYIwZNUt5oDRxjzS9OfNJLXZfS3HQi9r9MKainihdWIcJNviD825PFyL2scOHqq9SI2TcrSMABEzHaQtl+LXbnvbBaSeSYL9rSRUdfWzxbeSvHW1u4t+uGyDA0kymqlM6IUgili+8GJZJSQOABe2BplE3kJFvWOSKQvbzONpv3tY8np9MO6KZo61YmpZUEyi7GJwqkC3fj+EcfOAiWreREqbgiXY1oz3vY39xxz3BHTAzsaluNny+oMqyzDaJFKMqzbl29Q1h35H5DAyKpY1IndvShKFl9JB9V7jkW5wZZHmWJKpAZkj3RSKJEs/q62sCPT3wuaadpZBEAIpYw6vsZ2J23Jve1uLW6iwwTFusVsV2hvre+BsxBZSPwJIv+OPI1x6fUw9jbHmF1YFrAjvzb3OONHYCWELu6eY/DBTcL/T/PClhEYCK8hU29IYH/AC+MC8uW7fvSQeg/8e/zhSxSMZN0g2AngCx6devOCARanadqeQjznjLgbeHQGxN+h57XuMFSEJInVlAJBKiwFh3/AAxz94kKK87PsYG5HX69r4E1Wsc60swG+Q2UBDdja/T2+e344YUMsJjZdq7mX1ArbnAahZaKiaanp5swlhQFaanCCSQ3HALEC9uxPbBFfzVYb25PQCw64iNWatyzRGmMw1FnTWyvLKczTbSA8jdEjX3Z2IUfJvjJNuhZNUUPxQ1bnUL0OhNHVR/tnqR2pqaYxANRUP8A72YNb7oVbqhNrtyL7cWzTWm6TSOl8syHIKFY8ooqWPyIWp9sjm5LmQdTIzetiepYnFd8INL5nHFX6+11HDHrfVqxyzRKhC5fQhR5NGlzcWWzN7ta/Ixp6FuAAxuOSOv64vKSj2ohGLfcyJkWdV+yrGk0Iurjyz8+x+Vwyz7UdNpjJJ86zuqgyrKaWj86qqpwwWNhfi3vcABByxYAc4kM9z7LtJZTWZ1qCoFHQ04XzZVjvJI17IiqOXkYkKqAEsSAMU/KdH5hqvOKfVXiRSq01Md+RadlYSU+V36TTD7stYe7WKxfdS5G4hU1bC206juVyLJM28U6unz/AMR6GTLdG0DrV5RpSaMiSrcG61GYKAbnlSkHIF7tzcHRi8lNHOKcM4CMqs8bN90tYBLWAAsLD346Yn2kld2sbtcG4Yjk374S8vlgM/mG3N+lvocCU08VgeMKzYxhqJ5colqHMcco3Db5bqvBtz/FyPbHlneSoZ1BQCMqm+5RlsrbiAOf4gDh8Jix3Lf1Dgkc/wBcU3V3iLlOja+LKZ56vOtS1Cg0mnsoT7TXSi33ig4iS3WSQqoHPOFWXSRnUd2WOtqmqKeOKGKoSKppy5YGxQ+/Ti3U/GMwfxQqM6qZ8r8HqGm1TVpMyz5zWFxlNC/F1Mosahh6v3cV+12Aw5Oj9WeInlv4rtDQ5JI4MWksmq2eJ+4NdU8NPb/4k2x3Fzu6YvmY5rp/QGSwjNswyfSmU06bYYqmWKjhRRztSPgfgo5xRaY4StiO3l4RVNL6Qo6PMqfUmpJ5tU6mZNgzSusDSLwSlJCB5dMnN/T6zblzi+RQyivdAssiyepSFuFsBcdLm/NzjM5P+ITwtrquGnp9Z/tCenkEg/ZuXVdUpNrbSY4muOeg74o+s9aaR1B4iaTz/wDa0qaey2GaPPI679q5cZwReA/ZzCI5BGwZiCQWvY8Lh9EpPKEU4xWGbW9QtXVUkxsnl7rqpBBYG/Nx2UMCDh+uZxJaEoiOzFXVZRYex472N/wxC6Y8T9Ca1R4tN6syPN3c7TSpWqshJ4t5TlW/TnFoipBSqsUcQiCWusiAt7c37274m6WGiid7Mr+osqo9V5ZHTZgfsOZUlT9oy+oR1EtHVJcCeJuzLyGU8Mu5SCGxQvAvP6d8p1TS1dTRNmVPrDNWzGOnnFoJJJdwYIfUEfaxHJsOCeONay+mNNOwDmq8yoMoLRoNhNrgWHv/ADxhWjlGktZafzZI3Sg1NnGf6bzGcBdpq0zConoHa4uD/wBxFf5UHth41KLQku2SZuDVscshWKMMolX1FlK7DYXH6EfBwBM2iIPnW2ubElrbRbd72J+Pg4fqm52doBJdVBJQXFunTrbCJFi22lhhIJB2hVvweD7dz884j2+Cv5GBzQTUlXLTRqGglRWHmDkG3foCL8g+2B0NcomKyyqxlVWgi3BbtYrtBvybqeuHgpCKueoMs6x1KKhp2jTZcX59z174BJHKZ4jSwIQjs4LAKqnvYW55v+eD2mzuN6/MFSlgljeUhpLMw2MCUI3LcHqRfp7YWuZxQzyky+Ygg3qystnXuBz1uMLOXHynikplMNxIU2KAXve4FrX+fnBUoYkBMdKhmIvtAX2N+3AuenTA7TZ4ApVIbNL6YmLkHardGFxwew5t7HCqD7PHCI4qeSONQ0scahbv3IFiQD0+uHcdIhRlnp0khLG8AsVsQR0A5uO2EGJIG8ymo4VdQVU+kCxJuOnzzgtxo1Oxwytcm5BJHO29rfGF+ZZrISO7A82P+WBLIyqFWPoSpB7fS+OxhRtsArH+Hvf3/njnOkRUSSNZ0EtiSthbgf3r44k0u2zSKZCAWGwAC+D7iL+fu5tt56n8P9cJiZ3csLXtezG9h+GGQrPI7n+6C3AtwT9cdheUs+/yzIFHN7FQev8AL46YTGT5hTYApblfcjuD/vvhyQjD07Sbi4B6ge/xjAWwCR5JAAvIK2sLr36364xbNcvl8avE+OjkeGo0BoWqRq+DaSmZZ3tJ8onoyQgrc9Llhzu4tvi/rnMdIZJl+W6QjSfW+pqj9m5DABcJKbb6hh2SJTuPa5XtfFj0TpKh8P8ASeW6dyxpJ4aGM+dVOtnqp2JMszkdWdyTz2ti0eyOr9iMu+WnhblhLyM15G3FmO5+nPNziE1RqnKtH5LUZzqGoNLQQlY1ZIt8k8rG0cMSD1SOx4VQOfpc4VqPUuWaRyKpznP5ZKekgZUEcSGSWokc2jhij6ySO3CoOST7c4qemdJZpn2dQ6z8R4BS5vErDIskZg8WRxN/E23h6t1+/J0QHYvFyQo41S2DKVPTHcJkeS5jn+cUurtfQrS11J6sjyF2DplG4WaWQjiWsKmxb7sQJRP4mNuapneSUuJPL3hbhLkCwNyB9cSMccZUspG6wuSvNvphSUu4lYk2Fh/CbWHv8YWUnJjRioIbtu8uyhWW5UgX/Dm/XEZqjUuRaPyls11jmtFlFAqemWqksZLD7safedufuqCb/XFNq/EXN9VV0mT+CmXUedtCTFWanri37Ko5AbFIyo3Vcot92M7R3bFeo8r0npHV8tXPUZr4u+LkUYWRgqTVFHccKF4psujsb3chrE9SbGken9xKXU4iTC5prPxRKJpqHMvDzSkoW+aVVMFzmuS/SmgNxTIR/wC7Jd7G6qMByDMdEeHn2rTXhhktVqvURJaup8o21NRJMLktXZjIRGhv1Ej3F+E6DE7VaO1FrhGHiVmZy/J5xZ9N5DUOI5Ae1XXWWWYG9ikYjTsd4xdsqynLtP5ZTZdkdDS5ZldOoWGkpYViijHwqgfn1OC5RSoEYSk7KG+mNbaxpwdVamXR2XyJ6so0o/7/ANwJcxkXcT2IijQf4iMOsi8EtBafL1EGk6DMsxf1PX5uDmVVKT3Ms+4g/Swxf2QEAvFvI+5zax9r49HHZmK7lB6gNbn6YTXLjBT4485AIhpIhHR7aNFFhHTqIlsOgG0D+mCioqmP/cycHuzC/uDguyxb7wN/foPywMgi6ksFPQg837YWx6XgrWpPD7SOry39qtM5NnbkcT1GXxvIoHAtJbcCLm1jxinx+Gup9FzSTeF2sZjlgYbdNapL1tEBtA2x1VzUQni4N3AJNwRxjWGANwRe/e4GBiNdxtJISewtYDrh11JLAj6cWU/TGvYM9zR8hzvL6rS2sIIxPNklbMrtJDf/AK1PMvoqYvdl5W1mVTjPsyy6rk8INdzUJcZnk+qs3zuiQghjLR5gapV47MqMvH97Grap0jlOr6Kmp89il8yknFRQ1dNM0FRRTDpLDKvqjbsbcMOGBHGKx4X5nDPmOptN5kKn7Z/aLN54vtEO1K6laqKSPEQSGCMxRxwVLA7QrLiiaq0RkmsMu1LIM0gp6+gikejqokqYiik3SRQw5F+gYdMdkEgsYwFDE3uev4dr4+Usxy0ZjkOmKNKiKpzej0bWZDBHDn32eopMxgqxFTTiGJ/NbyhE92WNrggc34+jYsy1N+xsthyrJxmFetIkdRmWd1BoEeVQEaUwqrzDeQX27V+9yQcCfS08hjO8UTAUEbZN0jAHlxa45HX+uHKUs0MIcq0caqrPJKoAt7FjYDFVn05rXN4gMz1vHkiFQHh05k6Rsff99UmV/wAQq9MNG8GNHVEkVTqKkrdWVqNf7RqDMZ68seLehmEdgRwAgAucJUVux7lwgWrPGjSmmcpzioynNct1HnNBQz1MeVZfOKiQtGheztHuEaDaSzMRYA97YHoPVOoa3U+bZJqvN9P54gyWhzfLqrKKcQeiZnWRXTex2hguwn7y8+4F7pssocvy85fl9DQ0mXOhjNLBSpHEUIsVKKALG5FiLc4iNMaL07omKam0hkOXZDBUyeZMtDAsW8+7Hv8AHPHYYOqCi0kbTO02yYdnfeqKBbnlCdw6YQW4kV12soBHp4+gPfphwNy7jLZQDxY9frxxhtOURwdwuVte5/K3TES1BHDOouAAbgFrH+eEBtnAAt7ADccGZWkKhQy+xZfT+PPXHlRlLkndY9rjg4kVBeqxsfTYHlhwfY+/4Y8w2sAWBueARfjChEgdd7jjkB/e2ESuIbhirXtxcji9upwwDsgcI7AooJ+9yBx2+cNXmMdO1RUzJTwxIZJpnbZHGiglixvwAAScLmmuVWw3BgLWLfyxlHiVUTa/1PR+FOTTVENHLEtfrGsS96bLr3SkD8Wec2v3CfU4pCOp0TnLShx4ZQt4i6qzPxWzJJEoKqF8r0jSzLs8rLVb11TKed87hiP8HHQjGmZ5n1DpXKJs0zqd4qWEhAsSmSWaRjZIYoxy8jN6VQck/GA5xnOWaNyR67NJIsvyiijjhiWKHd7JFDEgBLux2oqLck2AxEZBlOY53mtNqrWtKKGsgVv2Nk5YSDKUcWaSRhw9W6mzOPTGpMafxs1Pqep7ElcFpW4DTWms0zTN49Y+IMCJnMe5cnygSCSLIoWuD6hcPVuP+pKL7fuJYAlrwlMIE8uBBGtzYXJHW5xDam1fkmjcvjrdSZglDFK3l06MrSz1L/8AxwxKC8jfCg/NsZnqrVud54kFJm1XV6CyyvQtSZVQRmq1Nmif/biS60qHuw3Ed2TmxUZdR3wByj01XJeNYeI2UaTrYsqVZ891ROt6XIcrtLWS36M4vaGL3kkKgfPTGc6mqa/UtXFlXihO++VTJB4f6YlknnqkPT7dIpVpE9wxhh9y2JjTHhxmEmWmjjpv/pzkk8glqqTK6r7RnGYHqTV5gb7CbC4jLt1HmDGjZNkGU6WoPsenKCLK6YvvkWEEmZz/ABSObtI/+Jyxw9w6e2WJU+pvsUyDTGsdR5dBR5vVwaB0/TKEiyPTk4FRJGOiS1aACFOl0pwDb+M4teQ6YTS2WLlemoKHKcsjJcQU9OFVnPVmI5du5diWPUnEsagrzs59jx/4wRZOQWBFr8Ht8DEnNsqunEAKmtguJ6dakHndGxsR9DjlFndNO3luk1HMG2+VOux7/HYj2sThwJd/3fSp6/XA5qWnroyk8fnpfpa5U3B+txhU1yNT4Y+DLITf1m3NyBjoZQO4HJv1xUtQZ5T6GpYanOcyocvob2D5nXJTsf8ACN5FzbtiqzeN+RZvS7dA02davqQ3rGVZRI8QG0kgzyeXD+O4j9MMunJ7C/JFbmqq5U9yB74UWViL+/HBucZDRZ/4t57RKaLTulNKxy2K1WbZk9fOqk8MKanGy9uxk64kf7E6rzN5H1j4n53NT8/8rp6hhyWLkd5B5kzcf4xg6K3YH1G9kXTU2p8i0bQGv1hnFDkVIASklbULCZLddqn1OenCgn88Uim8cslz6YQeHmQan1y9htnyzLPJpLn+9U1BRFt3PPxh9lvhZobKpmrqfS+XV2YEhvt2axvmNS3yZqgu1/yGLs2YPPEt7tHa4BBC/S3QfhjXBewd8vRlMPiXqzOtXV+QmhybSVJllbR5fXVr1ozCUVdTHvjp41YRoXClSbBhdrC9jioZ9QZ7S0ktBlMBziGPNK+smr56V5Pss4znZM6xQxFZmkglPmU/o9Acj2xslRo3TeZagjz2s0/QSZwHhdKySG8m+K4ia97FkDHa3JF7XGI/wnzdc1os7qo1kjvrPN4yHFj6Kspz+AHzimtJWkTcZXlg/CPM6afSXlUkNNQZhQ1tXS5tT0qhBT1H2h3Kg2BaMqytG1rMhU4vUR5CqblRe248fXGNaUq209/YzUm5pqDPJZ9PZxJf7jmtnFBO3FjtkDQH2FQvNhjZPOKIQw4tY+n+eJ9Rd1lem7Q5RX/jYWv0Xn/XHBcufvEKOOOcNTOUXdc2UcHbyPyx41DWGwB1PQ4mVHXA6kC3HI7/ANMN5HtbZtIW5It2tgbyMTckoTwfT+uAmUvbzBtNrn1bv1/zwDWK3MxAe3ve3GErdgOAL/l+N/zwj1MTe5LHjkfrhJG1yyt6mA3G3W3TGMPrJ22b+L7ksRjz24UFhc/3rDCw4QXDbABwN1gfnCGJsfXcnk8dsIUAuioRctY9AZB0+pGEnb3VvSegPIv3wzmq3V5ACodb7SpsCPbnj8cdimmnnjjhljYlwoDRm5/G/wBMGgEJr/W0Hh9pSuzuopzV1K2p8tojw1dXSemGBQDc7m6+yhjiG0hlVN4PaGmrtZZjJXZ/mtYKrOauEefNmOZy8CnhjA3SEW8uONewJ4FyKxHqHLNaanGvs9ilq9Hacqmy/R1LTL5s2b5qW2S1UMYt5huvlRXJWyySHaASD/20y7JMzqdT67qMvrdWUTGlio4KsfYdNxSdKY1JBU1TgDzTGrzOfSkewC/UoNKv3/8ADklO5X+xd8tyeaorRq/xBNNR1tCjyUFBJUp9myOIixdnPoaoK3DzdFBKJxdnrue+LArIEm0jPT0OSzjy4dR5tCzQ1Uht6MvpABLWvz94Wj+XsRiAzDTusfEiup6nOIaeLLI3EsUmeUrClgAvZqbKSd00vQietYWPKxL0xadNaMo49R1Gax09dm2YDck2cZi5qK2oseQr2Aij7COJUQe3fDVHeTEblssEFpbQmocwrmzhquoyatqVIlz/ADvZWZ3NGTfZDCR9noo+SQoU24/dg40nTOkMm0lHU/sKkVKirO6sr55Xnqqs+8s73d+e1wo7AYVqDV+QaVRTqjPcoyHcOEr62OBz8BGO49ugxAxeJuX5hsbSeV6h1UXYWmyzKZI4LG4uaio8qK3HUMf1wjc57bDpdOHOS8EOANyMy9nE3T4xwITwL2J9W5A9/wAQb/jilSZlrfNECUeW6e0wPV/zOa5gcymXnj9xThE9+sp/HAv7JZ5mCiLUniDnNQhc74sipYcnVubhS6K81uO0gPycT0rllNbauKLhmmYUOQ061mosxy/KaS9vPr6laVCfguQDikS+M2lq6KU6Sjz3W7xNt2abyiesjY3t/wBwyrCB877YkMt8M9H5JWnMqbT1PJmYcMK+vjNfVbuzedUM7g89rdTi2vUTtGv/AHO3dtA3WC2+Bxx8DB7F7A9fmjP1zXxLz8b8o0bkGkox9yXUWbNWVA/xGmpRt6dml6/r2Tw91HmSp/bXxGz6oje4al07BFktPc/4498xHUepxi/iRmQXb3DF4wRf5sBgYdQRLFHvYj78ZJ/QHjpja62VG+NPd2U/KPCbRORyiXL9J5bLWX3NW1cArqliQOTNPvZibe+JTMcslYCWlckLcGB4xtIt0t0H0AGJxpAUAVU545NiD8jHkkaVVG0HjbZSR16cWwFOV5YzgqwiPyuqgdWCRU9PKo2SR3YW54t7jD14il3LkoP/AIgb3/zxHV+SVHnrWZdT1KygekRpuDD2It0PzhzQ3lQTTRrR1KIEmSRitu/Qn689cNJXlCR8Mc8jlVkbm4JUk39uePyx0Fb/APMLLuUewA/MX5xE5lqXKcvivmed5VSfw7psyjiX6G7DEFVeJeiaK8k+ttN06XswTNonIPS3pY3wlN8D6lyy6x1cUJWySSgMGYKhNhf3P+eM88IVqcsynP4JkRpI9aZ07spJHNWWF7jrZh+WHkev9N5ioeizH7aoswaGkqJPMFiRs2x+olQTxc8Xtim6P1rlOkcu1HPnRrII31bm06Sim2iOOSVHTdJMUCkqwIU8gEducUSeloRyVpkxpzTy554R5vpi9nmqs7pacsNrRzLmFQ8D3HAZZBGwPwMSsHizlsVLps6hWopKvOslTNmqNiGnQBE88blbduWR1UIFLMXUKG5tFaK1JmlLlbRZFpXPszX9r5hUGRpqGngZpat5QA7TsSQJBewIve18Vqg05X6wpMkr8405JmiZIK+kpIV1NFSxJMKp0b7kG9ZEKqFlRxt8sFdtzelJt6ieVTRs1DmP7QpYayJKmGCUEpFUUr0so5td0kG5L2uAQDaxtzh19oP94O3uPX+uM403DrfI6aek/YMucReYvlzZ5rGOomVQLEBhSXIB/vk3JvfFlSt1vKCJNJabhufvS6jlawv3C0l+lyLHEpRzgqm2s7lgkm2g2CxHsAjE/liDm1TlVOc08/PsrT9lMgzFHqIlajLkKnmXJMdyQAD1Jt3w4SXV5Uj9mabhdk9KnMauUB+L3YQi4tfsDjOvEjwcznXWpK/N4qnK8uqGyikpKKrgWZJBJFO0kizLa0kEisF2liylEI6c6Ki33OgNySwjWI2klRWB9LgFd6st79PSQDg6iRAbL8+oMARb4/lh+XkqJ5GtKd79TdQPpgdtwswH/wCIf1HEyqRwWe/xbi/N/nCbklrXtxzfnp74Vb1sLAXtfjCSyAC9jbvb5xIqRmbZhSZbTGrzirp6CnWWOLz53EaCSRwiKGPdmZVA9yMZX416u/YYoNK/bKmiOcqz5tUwnbPBl6na0cJ4CzzsDEjHhQJH6LcPfHTwp1F4r5fl2VZJm1DleUUsc1TIJpJN0tbdRBcKpvGq77nhgWBF7YZ0PgVBqQrm3jEUzvVgiipzmOU5xWQJJHEu1GKekJJyb7eCSWsCTjqgoJKUmcs3NvSkUZM7lmmoHnqaHT1GkS0dKCXBo6W+37LSQQ/8wyEBQTGEea3qmjUeVjRdH5NTZQaat0xoPVGf5lTKUp8xzulhyqGlU3J8iOYr9ljJPIih3HjcWPOHGU/8PegsiqKiqyelzqjnqf8ArtDqaujaUf3XKSKXH1Jw3zbwu8PIatYK/QC515g8xppppqncxPIZ5ZtzMNvTnjFdcZYVknCcVY7z7UeqKSCc5xqfw+0QhViIxV/tCqAva5aZ4Yx1uTsYc4qL554ZVuXml1d4rVWqY1j3Nl1JnLRQMOlvJoUXf9GZsXV/D7QtIuULp/Q+mKOKomaVmXJ6fcCF4uShPU26/wAsaZE5gp1hpysMSKAkcPoVQLAABbWFh+GFc1FY/wB/YVCUnkxrSevvCXJKVKrQuQ1qdvPy3Q9bLIxBtzMYSzHjqXP1xKN425VO8kVBo3xEzd79E0pUpwe+6ZlHX3P6DGoNUzsdzszsvI/eHbe/bAiZGLcNMpNxyD+WJOcXx/JZQkua/BmLa/z+qilWh8JNaTdT/wA1Pl9Ivt1aoPt7fOER6o8QN8RpvCYQHqRXatpEsByT+7D40SWNWZrDzCtrgcke4574SqoiKZT5YVgRIV2m3PI7X98T1r7f7KaH9xR4NSeKVayij8O9J0AkX/q1msWl2/URU9zx7YWlP4zVUTHzPDTJ32qR+6zGtu1+VuWjC29xe+NBjgAUGXbe+677uLj9MOihJvc3vf8Aw/S3XDqfpE3085bM4TTvizUk/b9f6Wy5Lgg5dpJpDaxuv72f6W68dcek0V4gSInneLM19h3yQ6RoAvmXPIB3EDaQLEnp15xo3kXF/RxxypJH0wrydqgoSSp+6Etxg/I/8gfFEoUuia+RlWp8SNbydNwiqKOn6HsI6Ye/5DAz4aUjyMtVqnXlcdiqRJquojU2J5PlbPxxobQXNyC3yOt/x5vgTU8QcFYiGYEekWI7/rga5B+OF3RnkXg9o+R/tNZluZV0hWwavz+vqxa/s8/zhtmHgjoFonmp9D5FPOCHtUUAnDlb9QxPufr+GNMMXrUyXDL1PXHgGIU7bG3Vb9fgnrgrqST3DLpxa2KPlGjNEzQM+W6R09RTxel1jyGnDKelvuXsbde4xZKbK6GiZVoaGnpth6wU0SW+bAcfXjAMzytklFdliA1QBLRNZRIO4Pzh9llYmYUsctMCxuVkRybxtbof64MratMEaWKMd8VfCvP9carfMKCfJKzLKjK4qGlmzKpqIqvIahXdmrqMoCC53A2upLIgJCjmWzrTNVrjNKgZfmcNHDp/Ws1ZK7M4qJGShijCpKu7ZxM25irdABbGrO8m1bN2Pqva5+mM88Mklb+3nmzvM7a5zWxEjFuPKAX1ccAAALxYDvfDrqS0/oTcFqS8hPCvKcvyrJcxyzJ4Fpcuy/UeYUVNCt5CkcMqxoLt6r2Uc3wnwtg83TFbJsXbNqDOnuHNm/8AUJhfk35t+mD+Fk7VlHqO6KqprXOY9ptzaq6+/c9v0scK8L42l0fFJIsbNJmOaSGwsQDmFR9e3fCz5G6fBcIooyLEPG5H8MpFuPf2w7VVIuim46WYlr/N+uG0YCoBIB3P3hcc/wC+MHaNZFeOVQ0bdVJ6j8P64gdFC1VAzXj2G5JAQXvjoEfBUekj2648oPAdVsLXF+h/zxwuFYhhZu/HFsazCy67lLNvC8fd79sI9LKFbaF+8AOO2Blhe/ueQG6e3FsJaUk2ttIJFwb4xhQ9IaxYWPtxf64Srhja6gAfw9RgzX5+BciwN8DccEn0nndfgYUYC0rxsAgZVbkWs24ntx0wdWAWzOQLX5PXDWQMpYxsAWFyGUkY8Y0kl5I8sjlSoJv9cEUOVWQswJIt2IJ/LEPmGUw1tclVJU1VHIkRhIidCkgJNyQQbNbi45tfEqREibtxCr1YR24748UYM21huABt2b6/l+uGUtLwBx1bkBGsNJnGWUkC2WKnkKBpOVv8d+nXE99oLs6H1GO24Dk++IZYjLqd38w2hplUrtBHN78/iDiUMIChVksAtuR6Rhp8CQ5HVwx9X7wfBv8A+MAZ1A3F2seoHG359hgO07Vjci+8MF2XHXpjzRWSXapl3joBYkgWH4DE2VCkxlSGBI3DaQeo6i3544kDyBbFWgIPRPvL7Ent/rhm1FKwBh/cqegKA2bp17j8MOYKHcYvMZmZRwCbD8vi3vhaCPEBTmRgiW+6D2t/LjDjeSwChvnbzxho9LvIkMgQWIsBceq1uPw/U4P5DuxZyDftbn4+mHQrFWcMFLysvJI4P4YUrKxaPy2AHTkWP+Y/HHEQuCjlOTew629rf0x5oipDlgFAItvIFvp0v0wQULXahNlcD3J/UY4kqum+zIGseSQfxHUH4x4oShKhUPNxxc3x4qAd1wOhPSxxjHQdouAQPcci39ceJZeVCk/4mK3wlrXBQXXubXH44SgPAAYnuwX+uAEWxIk3X4NhtBvb5xC1p8iokzPLQryRArURgW3BevHuMTCx+pmIXrax6kXwkwAWFxwSTxYnjr/rh06Easb0uZwVcEVVTkskhQekXNyQNthzfn/Yxl/gZVRz5VrmeGX7TFUa8zqQM1wZF81VB9u3b2xf2yo5fmCVWWo7iVgZYlNg549V/f6YofgNTebkOc1USosVTrHO5FsSTY1dgL9P4fnFKWh16J33qyx+FMZ8nUTuRIJda5u6ncGNhVWsbdLbf5YZeCzfa/DXK5DzeuzNfXbbxmFR27/jgfgcwqtN1dWWMxq9X51MGMm4EftCQCx9vTj3gf5Y8Lci8gBU+0ZhYDhb/bqgG3A49j7dcaez/UXp7o0YoBw5Vjz1HX+mPXBKi24Wte97fhhNxe/A+jHCH7eqwJ5AviB1BDIqsbWYkX6/798CkYAc2IH8PtgbMQwXoD9bWwEPxckDngE4BhRsCpjVbC5IPHB74QApbdsQ26MBfA5WYW2+oi/XAr3F1BI+tiP6YICZ32HPH4YG3NyObgm5PIwoJcAmwA46WB9scZVAIPIHv0woxGyR1O5443Kr2fqb9uDjtpImLPZoxySD0/3zjssj+YwjtZHHYAAdeThZkZS5ju27nb1H0+PxwQILLKdp9aiK3pLNa/1+mPBjKVkXa6kD+MGw98B9TFItqrGLlrj7vtYHBgCirFEhVeNoRRtPHt26YASEpIaibU2ZuzuBGFC+odwtz8i3T5xLiKdAfNYsLNccAEdjiPyly+bZ06QrGxkW/HJsoF/pxiXmMjIfJKrIy2B2/dP44rPclDYbyb40FpLDgc8k4bSOwuxHA5HPb5/rh1LIzPbao9QAJvfntiPqDKh/dgEIbHsemEHOMZ6jckiuYi1+H73v/O/XD2CGQRBRxcDv056W+n8sRazNK62JZVF3Ifqfa3t8/GJUNJsQAqAeeB19xgUGxyqEsTsPN1+8Bdf9OcOGRr3DEN7huSMNxUlPLOxnLG33eTg0UrEEWG3qbf7+cEB68oXkMARwLA2OF3kKKSQLW4txj2+7uCAVBseb3/DBAyqfSLW4tbjGMC/eEsdqnm/A6/r1xx1JYWKglr8WP+yMKmkZWVFVm3mxst9vHXHWcoQqjm9rDvjWajm2TuPqbcfXHVia1yfWv1/LCVZlFyvP8IXv+mFxgkXJ6jjcvI+pxjHkVyDtN7Na5x5oSt73HxjryWZbvtsCfj88JLgIdzHdb8bfTGMJgEiVdOqn1GRbi5t15xkX/D9TVR0JQ5jTy/aElz/NXdSoDEGvk4YDo9hc35tbF61zQz5vorUWXUObJp6rrsunpYMxlcItPJINisSSLcm3BuL3HNsY5QeHOo6ih1W+msvyzQMFTpSnyOmoqHM1kpqisiffLPJLCTtBXdEJGtKVkJYcDHR00nB5ObqXqwi+/wDD8YG0HlUNBPBUrDm2YhvIkUlCa+dgGA5BKENzbrhHgTLC/hbkkFLWQ1n2KWsp5dkyvsYVcxCvYmzFSp55NwcUbMtJajzZtY6jy3TGVabSuy/LqEacjzKGSPMFp6kzTpNJT2RFljJhBN2233WUkYsfhdkebnWOsta1tFSaVy3PEp6WmyCmmhl2+TYLUzmFiiSkXTat+CfjDziqk7EhJ6o4NZaUC5L+WCbsbj8LXwhjJttIiseQbkEnCWdiS/t0IwF5y6EqWHAKtjkOwUzMLAqT0tfm31wN3ax6AHngck+18IM/J2kr3v74QSrsvPW1jbp/pjAsSZQZvJjNpggcix9IJIBv0HIPHXjHTCZPSUJlPsL3Ht1wb1ldqb2FunQg/TvhzFzw4L97nj2xgj34Nl9iO+FbN/pc77cD6f54ZPPIsiqG4YkHC4ZGN7m/q9vjCrIx6VrnawWw/ha3TDXz0EjjYWc2DHscHqwLB7AsTySL4RUxIr7FUBbjgfj/AEGGEs6Z1twSQFNgDa2FJVoA3I9JsTuwyY3ck8lib/rgkMayBS6hiR3wQWN8qlabOc5k4XbKiL2tZR/ridMiEHc25rX98V2gY/tTN+TxNF391F8TUn/TjtxucKbe2DLcEHgS1QjOtgVtbnoetsM5aqF3VTUHleLcn8fph8sSMSWUG7EG/wAH/THKilhSZZRGpkQAKx523HNr9OmFHYzgQBy0TL6jfdySeO3bBxODsubXPN064U6LHEzRqFIAtYYbyDar7ePUMGhW3Yfz02rxbud1gB9fxwVZx6TtIB6f7/DDSVQUdSLgcgH88F++/qJ6gdexvjGscmoKrfzTY9z0+gwTzib7efcX6fGG+0WPHRuMKCgEWA7/AMsYI4ZgAbgEA2F74Y171U1NOuUyRQ15UiGSoiZ41J6XCkEj4BGC3PqXsBxjjcvzz/sYWgh0N+LE8A8NycLjlcyc7gPki9scWFAu4LZmHJxw+hRt6364KNYvzh90jgE98DMhG4fet0OBTOSJGJ9SGyn2x6VQl9gtZiB9LYNC2VzxIpK7NdDZ7l2TZPQ55mFZStTxUeYTrFAS/pLszKw9F94FuSoFx1xmPhx4f6n0bkWrRWZNlcs+ZZHR0iZeM3So+11kMDwGVpfKRIojGyDYVY3ViSb87TMx+zv9MCqVCwkrwbYquo4x0oSXTUnZkXgzonUWgpq5s8yPLqWjXT2X0Rgps1StlramlDjgsiLFGyuRsa43Ne55xRfCfRWt/D/UWRVSafp8goGpHg1I0lfS1CVMIleSFKdIVDrIpfl3J+tgBj6MLHzpBfgdPyGESQx+UW2DdyL27Yf5Xm+SXxrHo9FVrKFkiIKMOCLkg9746ak2v6voVvz7WxDwqKfPZoIbpCFuEBNgbA/zxPKLIHH3ve+JyjQ6nYhGLG9ioPW6WwePafXtYt0AHX8hh3Two6lmXcRusT9MOREg2WUdv8sTZSORpEgAAZWBHe3UfXDoR2NuSpa/IvggiS7nbyFJH4YIqL6RbhlBPP0wBz//2Q==)

Niccolo Tartaglia. *Nova Scientia*, 1537.

La portée obtenue dans cette modélisation a un ordre de grandeur tout à fait compatible avec ce que l'on peut observer dans la réalité. Par ailleurs, la modélisation numérique montre que le nombre de Reynolds varie sensiblement au cours du mouvement. L'approche numérique apporte ainsi une véritable plus-value à cette étude.

### Optimisation du tir

On propose de calculer la portée du tir en fonction de l'angle afin d'estimer la valeur de l'angle qui correspond à une portée maximale, pour une vitesse donnée. Seule une approche numérique permet de réaliser cette étude.


```python
portee = []
V0 = 40. # ordre de grandeur courant de la vitesse initiale d'une balle de golf
```


```python
# on définit un intervalle d'angles de tir
angle = np.linspace(1.,89.,200)
# pour chaque valeur de l'angle de tir, on recherche la valeur de la portée et on la stocke dans la liste portee
for alpha in angle:
  # intervalle temporel
  t = np.linspace(0,10,1000)
  # conditions initiales
  init_cond = [0.,V0*np.cos(alpha/180*np.pi),0.,V0*np.sin(alpha/180*np.pi)] 
  # on appelle la fonction odeint pour résoudre le système différentiel
  sol = scint.odeint(chutelibre, init_cond, t, args=(rho, nu, g, R, m, S))
  # on récupère les valeurs numériques
  X,VX,Z,VZ, = sol.transpose()
  # On retire les points d'altitude négative en utilisant un masque
  # le masque est un tableau de même taille que Z dont chaque élément
  # vaut True ou False selon que l'élément correspondant de Z vérifie ou pas la condition
  masque = Z >= 0.0
  # On utilise ce masque pour ne garder dans les tableaux X et Z que les éléments qui correspondent à la valeur
  # True de l'élément correspondant du masque
  X = X[masque]
  Z = Z[masque]
  # On ajoute à la liste portee la valeur maximale de l'abscisse X
  portee.append(np.max(X))
```


```python
# portée du tir balistique sans frottements (Galilée)
agl = np.linspace(0,90,1000)
portee_balistique = V0**2/g*np.sin(2*agl/180*np.pi)
# représentation graphique
plt.figure(figsize=(9, 7))
plt.plot(angle,portee,label='avec frottements')
plt.plot(agl,portee_balistique,'r:',label='tir balistique sans frottements')
plt.xlabel('Angle de tir (°)')
plt.ylabel('Portée (m)')
plt.legend()
plt.show()
```


    
![png](mvt_bille_fluidenewtonien_files/mvt_bille_fluidenewtonien_42_0.png)
    


Dans le cas d'un tir balistique sans frottements correspondant à une vitesse initiale de norme $V_0=40\,\mathrm{m}\cdot\mathrm{s}^{-1}$, la portée maximale est obtenue pour un angle de tir de 45° et vaut $\frac{V_0^2}{g}=163,3\,\mathrm{m}$.
La modélisation présentée ici montre que la prise en compte des frottements limite la portée de façon très sensible. On constate aussi qu'en présence de frottements [la portée maximale est obtenue pour un angle inférieur à 40°.](https://www.refletsdelaphysique.fr/articles/refdp/pdf/2012/01/refdp201228p10.pdf)

### La "mur" aérodynamique

Pour un angle de tir de 37°, on cherche maintenant à savoir comment évolue la portée du tir en fonction de la vitesse de tir.


```python
portee=[]
angle = 37
Vmax = 100.
```


```python
# on définit un intervalle de vitesses de tir
vitesse = np.linspace(0.,Vmax,100)
# pour chaque valeur de la vitesse de tir, on recherche la valeur de la portée et on la stocke dans la liste portee
for Vzero in vitesse:
  # intervalle temporel
  t = np.linspace(0,10,1000)
  # conditions initiales
  init_cond = [0.,Vzero*np.cos(angle/180*np.pi),0.,Vzero*np.sin(angle/180*np.pi)] 
  # on appelle la fonction odeint pour résoudre le système différentiel
  sol = scint.odeint(chutelibre, init_cond, t, args=(rho, nu, g, R, m, S))
  # on récupère les valeurs numériques
  X,VX,Z,VZ, = sol.transpose()
  # On retire les points d'altitude négative en utilisant un masque
  # le masque est un tableau de même taille que Z dont chaque élément
  # vaut True ou False selon que l'élément correspondant de Z vérifie ou pas la condition
  masque = Z >= 0.0
  # On utilise ce masque pour ne garder dans les tableaux X et Z que les éléments qui correspondent à la valeur
  # True de l'élément correspondant du masque
  X = X[masque]
  Z = Z[masque]
  # On ajoute à la liste portee la valeur maximale de l'abscisse X
  portee.append(np.max(X))
```


```python
# portée du tir balistique sans frottements (Galilée)
speed = np.linspace(0,Vmax,100)
portee_balistique = speed**2/g*np.sin(2*angle/180*np.pi)
# représentation graphique
plt.figure(figsize=(9, 7))
plt.plot(vitesse,portee,label='avec frottements')
plt.plot(speed,portee_balistique,'r:',label='tir balistique sans frottements')
plt.xlabel('Vitesse de tir (m/s)')
plt.ylabel('Portée (m)')
plt.legend()
plt.show()
```


    
![png](mvt_bille_fluidenewtonien_files/mvt_bille_fluidenewtonien_48_0.png)
    



```python
print("La portée du tir pour V= %2.1f m/s est égale à %2.1f m"%(speed[9],portee[9]))
print("La portée du tir pour V= %2.1f m/s est égale à %2.1f m"%(speed[39],portee[39]))
```

    La portée du tir pour V= 9.1 m/s est égale à 7.6 m
    La portée du tir pour V= 39.4 m/s est égale à 76.2 m


La portée maximale croît avec la vitesse de tir, mais nettement moins rapidement qu'en l'absence de frottements. L'appellation ["mur aérodynamique"](https://hal.archives-ouvertes.fr/tel-01083862/document) traduit cette réduction de la portée du tir: les valeurs numériques ci-dessus montrent bien qu'un facteur 4 sur la vitesse de tir ne se traduit pas par un facteur 16 sur la portée comme le prédit le modèle sans frottement. Cet effet est bien connu des joueurs de badminton qui savent qu'il est vain de frapper plus fort le volant dans le but d'augmenter sensiblement la portée de sa trajectoire.
