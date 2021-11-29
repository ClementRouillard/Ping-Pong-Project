# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 12:42:44 2021

@author: François BATTUT & Clément ROUILLARD
"""
import numpy as np
import matplotlib.pyplot as plt


                            ## Partie 1
##Question 0


# A modifier en fonction du répertoire où sont stockées les données
motionU = "./Data/motionU.txt"
u=np.loadtxt(motionU)
motionT = "./Data/motionT.txt"
t=np.loadtxt(motionT)

plt.plot(t,u,'k.-')
plt.title('Position de la balle en fonction du temps')
plt.xlabel('Temps (s)')
plt.ylabel('Position de la balle en m')

##Question 1

v=np.zeros(121)

for I in range(1,121,1):
    v[I]=np.abs(u[I]-u[I-1])/(t[I]-t[I-1])
v[0]=v[1]

plt.figure();
plt.plot(t,v,'k.-')
plt.title('Vitesse de la balle en fonction du temps')
plt.xlabel('Temps (s)')
plt.ylabel('Vitesse (m/s)')

##Question 2

#Calcul du Travail
Somme = 0
m= 0.0027
g=9.81
for I in range(1,56,1):
    Somme = Somme + m*g*(u[I]-u[I-1])
print('')
print ("Le travail est de " , Somme, "J")

#Calcul de l'énergie cinétique
Ec = 0
m= 0.0027
g=9.81
for I in range(1,56,1):
    Ec = Ec + ( 0.5*m*(v[I]*v[I]) - 0.5*m*(v[I-1]*v[I-1]))
print ("L'énegie cinétique est de ",  Ec , "J")

##Question 3
#Calcul de Ec en tout point
ec = 0.5*m*(v*v)
#Calcul de Ep en tout point
Ep = g*m*u
#Calcul de Em en tout point
Em = ec+Ep

plt.figure();
plt.plot(t,Em,'b.-')
plt.plot(t,ec,'r.-')
plt.plot(t,Ep,'g.-')
plt.title('Énergie cinétique, de pesanteur et mécanique en fonction du temps')
plt.xlabel('Temps (s)')
plt.ylabel('Energie (J)')
plt.legend(("Em","Ec","Ep"))

print("La différence d'énegie mécanique est de " , Em[57]-Em[55] ,"J")
                                    #%% Partie 2
                                    
# La question 4 est uniquement théorique 
# Question 5
                                    
import numpy as np
import matplotlib.pyplot as plt


# Nous avons coupé les données en deux par rapport au point d'impact
# Il faudra modifier les lignes suivantes en fonction du répertoire où seront stockées les données


Tdebut = np.loadtxt("./Data/motionT(debut).txt")
Ydebut = np.loadtxt("./Data/motionU(debut).txt") 

Tfin = np.loadtxt("./Data/motionT(fin).txt")
Yfin = np.loadtxt("./Data/motionU(fin).txt")



# Première phase Méthode des moindres carrés

T=Tdebut
Y=Ydebut


Mt = np.array([T**2, T , (T+1)/(T+1)])
M = np.transpose(Mt)

A = np.dot(Mt, M)
B = np.dot(Mt, Y)


X = np.linalg.solve(A,B) # stocke les coéficients de la première partie
print('')
print("Les coefficients pour la position sont")
print(X)

#deuxième phase Méthode des moindres carrés

T2=Tfin
Y2=Yfin


Mt = np.array([T2**2, T2 , (T2+1)/(T2+1)])
M = np.transpose(Mt)

A = np.dot(Mt, M)
B = np.dot(Mt, Y2)


X2 = np.linalg.solve(A,B)
print(X2)

a1 = X[0]
b1 = X[1]
c1 = X[2]

a2 = X2[0]
b2 = X2[1]
c2 = X2[2]

# Ces lignes servent surtout pour la lisibilité, pour avoir une fonction qui ressemble à f(x) = ax²+bx+c

def f1(x) :
    return a1*x**2 +b1*x + c1
def f2(x) : 
    return a2*x**2 +b2*x + c2

# Question 6
print('')    
print ( "L'acceleration de pesenteur semble etre : " , (a1 + a2) , " m/s²")
print ("L'erreur est de : " , (a1 + a2)+9,81 ,  " m/s²")

# Cette fois-ci on corrige la fonction sachant que g = -9,81 m.s^-2

#Question 7

def f3(x) :
    return (-9.81/2)*x**2 +b1*x + c1
def f4(x) : 
    return (-9.81/2)*x**2 +b2*x + c2

# On affiche les trois courbes en même temps

plt.figure()
plt.plot(Tfin, Yfin, 'b-')
plt.plot(Tfin, f2(Tfin), 'r-')
plt.plot(Tfin, f4(Tfin), 'g-')
plt.plot(Tdebut, f3(Tdebut), 'g-')
plt.plot(Tdebut, Ydebut, 'b-')
plt.plot(Tdebut, f1(Tdebut), 'r-')
plt.legend(('positions mesurees' ,  'positions suivant le modèle' , 'position avec le modele corrige'))
plt.xlabel('temps(s)')
plt.ylabel('position en m')
plt.title("Positions au cours du temps de la balle suivant les mesures experimentales et le modele")



# On va mainenant regarder les erreurs suivant la position pour les fonctions corrigées

# On remarque que notre erreur ne dépasse pas 0,001 m i.e. 1 mm! 

erreur = np.arange(-0.0018 , 0.0018, 0.0002)
Nberreur = np.zeros(len(erreur)-1)


# On classe les erreurs en fonction de leurs amplitude

for i in range(len(Tdebut)):
    for j in range(len(erreur -1)):
        if(f3(Tdebut[i])-Ydebut[i] > erreur[j] and f3(Tdebut[i])-Ydebut[i] < erreur[j+1]):
            Nberreur[j] = Nberreur[j] +1
            
for i in range(len(Tfin)):
    for j in range(len(erreur)-1):
        if(f4(Tfin[i])-Yfin[i] > erreur[j] and f4(Tfin[i])-Yfin[i] < erreur[j+1]):
            Nberreur[j] = Nberreur[j] +1
plt.figure()
plt.title('''Frequence d'apparition des erreurs pour la position en fonction de leur amplitude pour les fonctions corrigees''')
plt.ylabel('''Frequence d'appartion des erreurs''')
plt.xlabel(''' Amplitude des erreurs''')

plt.xticks(erreur)

#Pour construire notre histogramme, nous avons défini une fonction constante par classe d'erreur

for i in range(len(Nberreur)):
    X = np.linspace(erreur[i], erreur[i+1], 100)
    Y = np.linspace(Nberreur[i], Nberreur[i], 100)
    plt.plot(X, Y, 'r-')
    plt.fill_between(X, Y, color='#539ecd')
    
#Question 8

##vitesse chute

# On est obligé de recalculer v parce que maintenant on regarde que la chute de la balle et pas le rebond
Z=len(Tdebut)

v=np.zeros(Z)

for I in range(1,Z,1):
    v[I]=np.abs(Ydebut[I]-Ydebut[I-1])/(Tdebut[I]-Tdebut[I-1])
v[0]=v[1]

T=Tdebut
Y=v  # Ce sont les vitesse expérimentalement mesurées


Mt = np.array([ T , (T+1)/(T+1)])
M = np.transpose(Mt)

A = np.dot(Mt, M)
B = np.dot(Mt, Y)


X3 = np.linalg.solve(A,B)

print('')
print("Les coefficients pour la vitesse sont")
print(X3)

a3= X3[1]
b3= X3[0]


def f5(x) :
    return abs((9.83)*x + a3)


##vitesse rebond
W=len(Tfin)
q=np.zeros(W)

for I in range(1,W,1):
    q[I]=np.abs(Yfin[I]-Yfin[I-1])/(Tfin[I]-Tfin[I-1])
q[0]=q[1]




T=Tfin
Y=q


Mt = np.array([ T , (T+1)/(T+1)])
M = np.transpose(Mt)

A = np.dot(Mt, M)
B = np.dot(Mt, Y)


X4 = np.linalg.solve(A,B)
print(X4)


a3= X3[1]
b3= X3[0]
a4=X4[1]
b4=X4[0]

def f5(x) :
    return abs((9.81)*x + a3)
def f6(x) :
    return abs((-9.81)*x + a4)
plt.figure()
plt.plot(Tdebut, (f5(Tdebut)))
plt.plot(Tfin, (f6(Tfin)))
plt.title('vitesse absolue en fonction du temps en m/s')
plt.figure()

# Affichage de l'erreur en fonction de T 

# plt.plot(Tdebut, (f5(Tdebut)-v))
# plt.plot(Tfin, (f6(Tfin)-q))

# Affichage de l'histogramme sachant que l'erreur va de -0.00006 à 0.00006



erreur = np.arange(-0.018 , 0.07, 0.006)
Nberreur = np.zeros(len(erreur)-1)

for i in range(len(Tdebut)):
    for j in range(len(erreur)-1):
        if(f5(Tdebut[i])-v[i] > erreur[j] and f5(Tdebut[i])-v[i] < erreur[j+1]):
            Nberreur[j] = Nberreur[j] +1
            #On répertorie les erreurs sur Tdebut
            
            
            
for i in range(len(Tfin)):
    for j in range(len(erreur)-1):
        if(f6(Tfin[i])-q[i] > erreur[j] and f6(Tfin[i])-q[i] < erreur[j+1]):
            Nberreur[j] = Nberreur[j] +1
            #On répertorie les erreurs sur Tfin

plt.title('''Frequence d'apparition des erreurs sur la vitesse en fonction de leur amplitude avec g connu''')
plt.ylabel('''Frequence d'appartion des erreurs''')
plt.xlabel(''' Amplitude des erreurs''')

plt.xticks(erreur)

for i in range(len(Nberreur)):
    X = np.linspace(erreur[i], erreur[i+1], 100)
    Y = np.linspace(Nberreur[i], Nberreur[i], 100) # On crée un fonction constante au nombre d'erreur par intervalle
    plt.plot(X, Y, 'r-')
    plt.fill_between(X, Y, color='#539ecd') #On colorie en dessous
        
# L'annalyse des graphes précédents nous donne une erreur systhématique d'environ 0,012m/s
#On peut donc chercher à la corriger en rajoutant cette constante dans l'équation
# On a alors : 
    
def f5(x) :
    return abs((9.81)*x + a3) - 0.012
def f6(x) :
    return abs((-9.81)*x + a4) - 0.012

#Si l'on reproduit lhistogramme précédent avec les nouvelles fonctions, celui-ci sera centré



erreur = np.arange(-0.07 , 0.07, 0.006)
Nberreur = np.zeros(len(erreur)-1)

for i in range(len(Tdebut)):
    for j in range(len(erreur)-1):
        if(f5(Tdebut[i])-v[i] > erreur[j] and f5(Tdebut[i])-v[i] < erreur[j+1]):
            Nberreur[j] = Nberreur[j] +1
            #On répertorie les erreurs sur Tdebut
            
            
            
for i in range(len(Tfin)):
    for j in range(len(erreur)-1):
        if(f6(Tfin[i])-q[i] > erreur[j] and f6(Tfin[i])-q[i] < erreur[j+1]):
            Nberreur[j] = Nberreur[j] +1
            #On répertorie les erreurs sur Tfin
plt.figure()
plt.title('''Frequence d'apparition des erreurs sur la vitesse en fonction de leur amplitude avec g connu et l'ajout d'une constante correctrice''')
plt.ylabel('''Frequence d'appartion des erreurs''')
plt.xlabel(''' Amplitude des erreurs''')

plt.xticks(erreur)

for i in range(len(Nberreur)):
    X = np.linspace(erreur[i], erreur[i+1], 100)
    Y = np.linspace(Nberreur[i], Nberreur[i], 100) # On crée un fonction constante au nombre d'erreur par intervalle
    plt.plot(X, Y, 'r-')
    plt.fill_between(X, Y, color='#539ecd') #On colorie en dessous


# On remarque qu'on est plus près de la mesure, ce qui signifie surement une erreur systhématique par rapport à l'appareil de mesure.

# Partie 3
 
# Question 9


#On rappelle que f3 et f4 sont les fonction de la trajectoire repectivement pre et post impact.
#Le croisement de celles-ci correspond donc théoriquement au point d'impact. 
#Pour le trouver, on cherche donc l'annulation de leur différence. 

def f8 (x) : 
    return abs(f4(x)-f3(x))


Au =  0.5*(np.sqrt(5)-1) # Au = or
tolerance = 10**-2

a = 0.04 # Ca a été déterminé  grossièrement graphiquement
b = 0.08

# Programme de la section dorée

while abs(b-a) > tolerance:
    c = Au*a + (1-Au)*b
    d = a+b-c
    if(f8(c)<f8(d)):
        b=d
    else:
        a=c

print(' ')
print(' ')
print('''L'instant de l'impact est ''', c*10**3 , ' ms')

# fin 
