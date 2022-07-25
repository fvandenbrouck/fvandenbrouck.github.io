## Importation des bibliothèques

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt


## Affichage du signal audio échantillonné et de son  spectre

fe_origine, signal_entree_origine = wav.read("notes_piano.wav") # fe est la fréquence d'échantillonnage # Attention à spécifier le chemin d'accès du fichier "notes_piano.wav"

fe = 2450 # Nouvelle fréquence d'échantillonnnage qui doit être inférieure à fe_origine (44100 dans le cas de cette ressource)

signal_entree = signal_entree_origine[::int(fe_origine/fe)]

Te = 1 / fe

Ne = len(signal_entree) # Ne est le nombre d'échantillon
duree_signal = (Ne - 1) * Te

temps = np.array([i * Te for i in range(Ne)])

plt.subplot(221)
plt.plot(temps, signal_entree,'b')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude du signal en entrée')

FFT_signal_entree = np.fft.fft(signal_entree)
coef_fourier_entree  = np.concatenate((FFT_signal_entree[0] / Ne, FFT_signal_entree[1:] * 2 / Ne), axis = None) # Permet d'obtenir les bonnes amplitudes sur le spectre
module_coef_fourier_entree = np.absolute(coef_fourier_entree)

freq = np.array([i/duree_signal for i in range(Ne)]) # L'écart fréquentiel entre deux valeurs du spectre est 1/Durée_signal

plt.subplot(222)
plt.plot(freq, module_coef_fourier_entree,'b')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Module des coefficients de Fourier du signal en entrée')


## Filtrage numérique passe bas d'ordre 1

fc = 720 # Fréquence de coupure

signal_sortie = np.zeros_like(signal_entree)

for i in range(Ne - 1):
    signal_sortie[i + 1] = signal_sortie[i] * (1 - 2*np.pi*fc*Te) +  signal_entree[i] * (2*np.pi*fc*Te) # Schéma Euler explicite
    signal_sortie[i + 1] = signal_sortie[i] / (1 + 2*np.pi*fc*Te) +  signal_entree[i + 1] * ((2*np.pi*fc*Te) / (1 + 2*np.pi*fc/fe)) # Schéma Euler implicite


## Affichage du signal sortie du filtre et de son spectre

plt.subplot(223)
plt.plot(temps, signal_sortie,'r')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude du signal en sortie')

FFT_signal_sortie = np.fft.fft(signal_sortie)
coef_fourier_sortie  = np.concatenate((FFT_signal_sortie[0] / Ne, FFT_signal_sortie[1:] * 2 / Ne), axis = None) # Permet d'obtenir les bonnes amplitudes sur le spectre
module_coef_fourier_sortie = np.absolute(coef_fourier_sortie)

freq = np.array([i/duree_signal for i in range(Ne)])

plt.subplot(224)
plt.plot(freq, module_coef_fourier_sortie,'r')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Module des coefficients de Fourier du signal en entrée')


## Enregistrement du signal de sortie

wav.write("notes_piano_re_ech.wav",fe,signal_entree)
wav.write("notes_piano_re_ech_filtre.wav",fe,signal_sortie)


## Affichage des courbes

plt.show()