## Importation des bibliothèques

import numpy as np
import matplotlib.pyplot as plt


## Génération d'un signal sinusoïdal bruité échantillonné

A = 1 # Amplitude du signal
f = 30 # Fréquence (en Hertz) du signal sinusoïdal
Ab = 0.6 # Amplitude du bruit
fb = 1990 # Fréquence (en Herts) du bruit #  On peut choisir 1990 pour illustrer la limitation due à l'échantillonnage

duree_signal = 0.2 # Durée du signal en seconde
fe = 2000 # Fréquence d'échantillonnage
Te = 1 / fe # Période d'échantillonnage
Ne = 1 + int(duree_signal / Te) # Nombres d'échantillons

temps = np.array([i * Te for i in range(Ne)])
signal_entree = np.array([A*np.sin(2*np.pi*f*Te*i) + Ab*np.sin(2*np.pi*fb*Te*i) for i in range(Ne)]) # Génération d'un signal bruité


## Vérification du critère de Shannon

print("La fréquence du signal sinusoïdal est ", f," Hz")
print("La fréquence du bruit sinusoïdal est ", fb," Hz")
print("La fréquence d'échantillonnage est ", fe," Hz")

if fe > 2*f and fe > 2*fb :
    print("Le critère de Shannon est bien vérifié")
else :
    print("Le critère de Shannon n'est pas vérifié")


## Affichage du signal d'entrée échantillonnée et de son spectre
plt.figure()
plt.subplot((211))
plt.plot(temps, signal_entree,'b.')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude du signal en entrée')

FFT_signal_entree = np.fft.fft(signal_entree)
coef_fourier_entree  = np.concatenate((FFT_signal_entree[0] / Ne, FFT_signal_entree[1:] * 2 / Ne), axis = None) # Permet d'obtenir les bonnes amplitudes sur le spectre
module_coef_fourier_entree = np.absolute(coef_fourier_entree)

freq = np.array([i/duree_signal for i in range(Ne)]) # L'écart fréquentiel entre deux valeurs du spectre est 1/duree_signal

plt.subplot((212))
plt.plot(freq, module_coef_fourier_entree,'b')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Module des coefficients de Fourier du signal en entrée')


## Filtre numérique passe bas d'ordre 1

fc = 40 # Fréquence de coupure

signal_sortie = np.zeros(len(signal_entree))

for i in range(Ne - 1):
    signal_sortie[i+1] = signal_sortie[i] * (1 - 2*np.pi*fc/fe) +  signal_entree[i] * (2*np.pi*fc/fe) # Schéma Euler explicite
    #signal_sortie[i+1] = signal_sortie[i] / (1 + 2*np.pi*fc/fe) +  signal_entree[i-1] * ((2*np.pi*fc/fe) / (1 + 2*np.pi*fc/fe)) # Schéma Euler implicite


## Affichage du signal de sortie échantillonnée et de son spectre

plt.figure()
plt.subplot((211))
plt.plot(temps, signal_sortie,'r.')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude du signal en sortie')

FFT_signal_sortie = np.fft.fft(signal_sortie)
coef_fourier_sortie  = np.concatenate((FFT_signal_sortie[0] / Ne, FFT_signal_sortie[1:] * 2 / Ne), axis = None) # Permet d'obtenir les bonnes amplitudes sur le spectre
module_coef_fourier_sortie = np.absolute(coef_fourier_sortie)

freq = np.array([i/duree_signal for i in range(Ne)])

plt.subplot((212))
plt.plot(freq, module_coef_fourier_sortie,'r')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Module des coefficients de Fourier du signal en sortie')


## Affichage des courbes

plt.show()