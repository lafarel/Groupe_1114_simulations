"""
11.14 Simulation des oscillation de la plateforme

Hypothèses d'application du modèle:
    - L'angle d'inclinaison est faible:
        - La position de la charge ne change pas (très peu) avec l'inclinaison
        - La position du centre de gravité du système ne change pas avec l'inclinaison
    - La plateforme n'est ni submergée ni soulevée (les simulations s'arrêtent si ces limites sont atteintes)
    - La masse de la plateforme est uniformément répartie
    - La charge a toujours des coordonnées positives
"""
from math import atan, tan, sin, cos, pi, asin, degrees
import numpy as np
import matplotlib.pyplot as plt


class Charge:
    """
    Classe représentant une charge par sa masse et par la position de son centre de masse par rapport au centre de la
    plateforme au niveau de l'eau.
    """

    def __init__(self, x_pos, z_pos, masse):
        """
        Args:
            x_pos, z_pos: floats, représentent les coordonnées de la charge
        """
        self.x = x_pos  # distance horizontale de la charge par rapport au milieu de la plateforme
        self.z = z_pos  # hauteur de la charge par rapport au niveau de l'eau
        self.masse = masse  # masse de la charge

    def couple(self):
        """ Calcule le couple engendré par la charge par rapport à l'origine """
        return self.masse * 9.81 * self.x

    def moment_inertie(self):
        """ Calcule le moment d'inertie de la charge par rapport à l'origine """
        return self.masse * (self.x ** 2 + self.z ** 2)


class Grue:
    """
    Classe représentant le bras de la grue. Cette classe est spécifique à notre design de grue mais les dimensions et
    masses peuvent être paramétrées
    """

    def __init__(self, l_mat, l_inter, l_hor, masses):
        """
        Args:
            l_mat: longueur du mat vertical de la grue
            l_inter: longueur de la partie intermédiaire de la grue
            l_hor: longueur de la partie finale (horizontale) de la grue
            masses: liste/tuple contenant les masses des 3 partie précitée dans le même ordre
        """
        self.l_mat = l_mat
        self.l_inter = l_inter
        self.l_hor = l_hor
        self.masses = masses
        self.masse_tot = sum(self.masses)  # masse totale de la grue

    def angle(self, hauteur):
        """
        Trouve l'angle que doit prendre la première rotule par rapport à l'horizontale pour que la charge atteigne la
        hauteur voulue au dessus de la plateforme

        Args:
            hauteur: float, hauteur de la charge (par rapport au dessus de la plateforme)
        Returns:
            float, valeur de l'angle [rad] entre la partie intermédiaire de la grue et l'axe horizontal
        """
        if hauteur > self.l_mat + self.l_inter:
            raise ValueError("hauteur hors de portée (trop haut)")
        if hauteur < self.l_mat - self.l_inter:
            raise ValueError("hauteur hors de portée (trop bas)")
        return asin((hauteur - self.l_mat) / self.l_inter)

    def base(self, distance, hauteur):
        """
        Trouve la position en x que doit prendre la base de la grue pour que la charge portée soit à la position voulue

        Args:
            distance: float, distance de la charge par rapport au milieu de la plateforme
            hauteur: float, hauteur de la charge par rapport au dessus de la plateforme
        Returns:
            float, coordonnée x de la base de la grue
        """
        angle = self.angle(hauteur)
        return distance - self.l_hor - (cos(angle) * self.l_inter)

    def centre_masse_mat(self, distance, hauteur):
        """
        Trouve les coordonnées du centre de masse du mat pour une position donnée de la charge

        Args:
            distance: float, distance de la charge par rapport au milieu de la plateforme
            hauteur: float, hauteur de la charge par rapport au dessus de la plateforme
        """
        x = self.base(distance, hauteur)
        z = self.l_mat / 2
        return x, z

    def centre_masse_inter(self, distance, hauteur):
        """
        Trouve les coordonnées du centre de masse de la partie intermédiaire pour une position donnée de la charge

        Args:
            distance: float, distance de la charge par rapport au milieu de la plateforme
            hauteur: float, hauteur de la charge par rapport au dessus de la plateforme
        """
        x = (self.base(distance, hauteur) + (distance - self.l_hor)) / 2
        z = (self.l_mat + hauteur) / 2
        return x, z

    def centre_masse_hor(self, distance, hauteur):
        """
        Trouve les coordonnées du centre de masse de la partie horizontale pour une position donnée de la charge

        Args:
            distance: float, distance de la charge par rapport au milieu de la plateforme
            hauteur: float, hauteur de la charge par rapport au dessus de la plateforme
        """
        x = distance - self.l_hor / 2
        z = hauteur
        return x, z

    def centre_masse(self, distance, hauteur):
        """
        Trouve les coordonnées du centre de masse du bras de la grue pour une position donnée de la charge

        Args:
            distance: float, distance de la charge par rapport au milieu de la plateforme
            hauteur: float, hauteur de la charge par rapport au dessus de la plateforme
        Returns:
            Un tuple contenant les coordonnées x, z du centre de masse du bras de la grue
        """
        centre1 = self.centre_masse_mat(distance, hauteur)
        centre2 = self.centre_masse_inter(distance, hauteur)
        centre3 = self.centre_masse_hor(distance, hauteur)
        coords = []
        for i in range(
                2):  # calcule la moyenne des coordonées (d'abord x puis z) pondérées par les masses associées
            total = centre1[i] * self.masses[0] + centre2[i] * self.masses[1] + centre3[i] * \
                    self.masses[2]
            coords.append(total / self.masse_tot)
        return coords

    def couple(self, distance, hauteur):
        """
        Calcule le couple engendré par la grue par rapport à l'origine

        Args:
            distance: float, distance de la charge par rapport au milieu de la plateforme
            hauteur: float, hauteur de la charge par rapport au dessus de la plateforme
        Returns:
            float, valeur du couple [N/m]
        """
        return self.masse_tot * 9.81 * self.centre_masse(distance, hauteur)[0]


class Plateforme:
    """
    Classe représentant une plateforme carrée par ses dimensions (hauteur et largeur) et sa masse (supposée uniformément
    répartie)
    """

    def __init__(self, hauteur, largeur, masse):
        self.hauteur = hauteur
        self.largeur = largeur
        self.masse = masse

    def moment_inertie(self, enfoncement):
        """
        Calcule le moment d'inertie de la plateforme par rapport à l'origine

        Args:
            enfoncement: float, enfoncement de la plateforme dans l'eau
        Returns:
            float donnant la valeur de l'inertie [kg * m**2]
        """
        # formules basées sur les formules du moment d'inertie d'un rectangle (trouvée sur wikipédia)
        h_c_prim = self.hauteur - enfoncement
        i_1 = self.masse * (h_c_prim / self.hauteur) * (
                (4 * h_c_prim ** 2) + self.largeur ** 2) / 12
        i_2 = self.masse * (enfoncement / self.hauteur) * (
                (4 * enfoncement ** 2) + self.largeur ** 2) / 12
        return i_1 + i_2


def get_enfoncement(masse_totale, plateforme):
    """
    Calcule l'enfoncement de la plateforme en fonction des caractéristiques de base du système.

    Args:
        masse_totale: float, masse total du système (grue + charge + plateforme)
        plateforme: objet Plateforme
    Returns:
        float, valeur de l'enfoncement de la plateforme [m]
    """
    return masse_totale / (1000 * plateforme.largeur ** 2)


def incl_max_sub(plateforme, enfoncement):
    """
    Calcule l'inclinaison maximale que peut prendre la plateforme avant d'être submergée en fonction de ses
    caractéristique de base.

    Args:
        plateforme: objet Plateforme
        enfoncement: float, enfoncement de la plateforme dans l'eau
    Returns:
        float, valeur de l'angle max [rad]
    """
    return atan(((plateforme.hauteur - enfoncement) * 2) / plateforme.largeur)


def incl_max_soul(plateforme, enfoncement):
    """
    Calcule l'inclinaison maximale que peut prendre la plateforme avant de se soulever en fonction de ses
    caractéristique de base. (Si l'angle dépasse cette valeur, la plateforme flotte encore mais les équations que nous
    utilisons deviennent inadaptées)

    Args:
        plateforme: objet Plateforme
        enfoncement: float, enfoncement de la plateforme dans l'eau
    Returns:
        float, valeur de l'angle max [rad]
    """
    return atan((enfoncement * 2) / plateforme.largeur)


def xC(theta, plateforme, enfoncement):
    """
    Calcule la coordonnée horizontale (en x) du centre de poussée du système avant de prendre en compte la rotation de
    la plateforme, en fonction des caractéristiques de base et du déplacement de la charge.

    Args:
        theta: float correspondant à l'inclinaison de la plateforme en radians.
        plateforme: objet Plateforme correspondant à la plateforme
        enfoncement: float correspondant à l'enfoncement de la plateforme [m]
    Returns:
        float, coordonnée x du centre de gravité du système
    """
    return (tan(theta) * plateforme.largeur ** 2) / (12 * enfoncement)


def zC(theta, plateforme, enfoncement):
    """
    Calcule la coordonnée verticale (en z) du centre de poussée du système avant de prendre en compte la rotation de
    la plateforme, en fonction des caractéristiques de base et du déplacement de la charge.

    Args:
        theta: float correspondant à l'inclinaison de la plateforme en radians.
        plateforme: objet Plateforme correspondant à la plateforme
        enfoncement: float correspondant à l'enfoncement de la plateforme [m]
    Returns:
        float, coordonnée z du centre de gravité du système
    """
    s1 = (tan(theta) * plateforme.largeur ** 2) / 2
    s2 = (enfoncement * plateforme.largeur) - s1
    z_c1 = -(tan(theta) * plateforme.largeur) / 6
    z_c2 = - ((tan(theta) * plateforme.largeur / 2) + enfoncement) / 2

    return ((s1 * z_c1) + (s2 * z_c2)) / (enfoncement * plateforme.largeur)


def xC_final(theta, plateforme, enfoncement):
    """
    Calcule la coordonnée horizontale (en x) du centre de poussée du système en tenant compte de l'inclinaison theta de
    la plateforme, en fonction des caractéristiques de base et du déplacement de la charge

    Args:
        theta: float correspondant à l'inclinaison de la plateforme en radians.
        plateforme: objet Plateforme correspondant à la plateforme
        enfoncement: float correspondant à l'enfoncement de la plateforme [m]
    Returns:
        Un float correspondant à la coordonnée x finale du centre de gravité du système.
    """
    x_c, z_c = xC(theta, plateforme, enfoncement), zC(theta, plateforme, enfoncement)
    return cos(theta) * x_c + sin(theta) * z_c


def angle_equilibre(plateforme, charge, precision, grue=None):
    """
    Calcule par essais erreur la valeur de l'angle d'équilibre pour une situation donnée

    Args:
        plateforme: objet Plateforme contenant les caractéristique de la plateforme
        charge: objet Charge contenant les caractéristiques de la charge
        precision: int, nombre de décimales voulue pour l'angle
        grue: objet Grue (optionnel)
    """

    masse_totale = plateforme.masse + charge.masse

    enfoncement = get_enfoncement(masse_totale, plateforme)

    soul = incl_max_soul(plateforme, enfoncement)
    sub = incl_max_sub(plateforme, enfoncement)

    theta = 0  # commence en testant un angle = 0
    theta_max = min(sub, soul)  # détermine l'angle a ne pas dépasser
    angle_neg = True  # indique si l'angle est positif on negatif (negatif par défaut, vérification ar après)
    for decimal in range(precision):
        decimal = 10 ** -decimal
        for i in range(9):
            couple_archi = masse_totale * 9.81 * xC_final((theta + decimal), plateforme,
                                                          enfoncement)
            if grue:  # calcul du couple si la grue est prise en compte
                couple_total = charge.couple() + grue.couple(charge.x, charge.z - plateforme.hauteur + enfoncement) - couple_archi
            else:  # calcul du couple si le poids de la grue est ignoré
                couple_total = charge.couple() - couple_archi
            if couple_total > 0 and theta < theta_max:  # Si l'inclinaison est positive
                theta += decimal
                angle_neg = False
            elif couple_total < 0 and theta > -theta_max and angle_neg:  # si l'inclinaison est négative
                theta -= decimal
            else:
                break
    return theta


def initialisation():
    """
    Initialise la simulation. Retourne les variables utilisées dans la simulation.
    C'est dans cette fonction qu'il faut modifier les paramètres d'une simulation.
    """
    # création de la plateforme
    hauteur1 = 0.19  # hauteur de la plateforme [m]
    largeur = 0.59  # largeur de la plateforme [m]
    masse1 = 27  # masse de la plateforme [kg]
    plateforme = Plateforme(hauteur1, largeur, masse1)

    # création de la charge
    distance = 0.7    # [m]
    hauteur2 = 0.5    # [m]
    masse2 = 0.2    # [kg]
    charge = Charge(distance, hauteur2, masse2)

    # création de la grue
    grue = Grue(0.33, 0.305, 0.225, (0.9, 0.4, 0.5))  # les valeurs sont spécifiques à notre prototype de grue

    # variables dépendantes
    masse_totale = plateforme.masse + charge.masse + grue.masse_tot  # masse du système
    enfoncement = get_enfoncement(masse_totale, plateforme)
    soul = incl_max_soul(plateforme, enfoncement)
    sub = incl_max_sub(plateforme, enfoncement)
    theta_final = angle_equilibre(plateforme, charge, 8, grue)

    coef_amort = 0.4  # coefficient d'amortissement de l'eau

    # paramètres de simulation
    step = 0.0001  # "pas" de la simulation [s], directement lié à la précision
    end = 20  # durée de la simulation [s]
    theta_0 = 0  # angle initial [rad]
    omega_0 = 0  # vitesse angulaire initiale [rad/s]

    # initialisation de la simulation
    t = np.arange(0, end, step)
    theta = np.empty_like(t)
    omega = np.empty_like(t)
    alpha = np.empty_like(t)

    theta[0] = theta_0
    omega[0] = omega_0

    return plateforme, charge, grue, masse_totale, enfoncement, sub, soul, coef_amort, step, end, t, theta, omega, alpha, theta_final


def simulation_charge_fixe():
    """
    Réalise une simulation des oscillations de la plateforme pour une charge fixe

    Retourne un tuple contenant les angles critiques calculé en début de simulation, des array numpy représentant (dans
    l'ordre) le temps, l'inclinaison et la vitesse angulaire et l'angle d'équilibre calculé en début de simulation
    Returns:
        sub, soul: floats, angle de submersion et de soulevement [rad]
        t, theta, omega: array numpy de même taille représentant le temps [s], l'inclinaison [rad] et la vitesse
            angulaire [rad/s]
        theta_final: float, angle d'équilibre [rad] calculé en début de simulation
    """
    # Initalisation de la simulation
    plateforme, charge, grue, masse_totale, enfoncement, sub, soul, coef_amort, step, end, t, theta, omega, alpha, theta_final = initialisation()
    moment_inertie = plateforme.moment_inertie(enfoncement) + charge.moment_inertie()
    for i in range(len(t) - 1):
        if abs(theta[i]) > min(sub,
                               soul):  # vérifie que l'inclinaison ne dépasse pas les valeurs critiques
            theta[i:] = 0
            omega[i:] = 0
            alpha[i:] = 0
            break  # arrête la simulation si un anlge critique est dépassé

        # Calcul du couple total
        couple_archi = masse_totale * 9.81 * xC_final(theta[i], plateforme,
                                                      enfoncement)  # Couple de la force d'Archimède
        couple_amort = - coef_amort * omega[i]
        couple_total = charge.couple() - couple_archi + couple_amort + grue.couple(charge.x, charge.z - plateforme.hauteur + enfoncement)

        alpha[i] = couple_total / moment_inertie
        omega[i + 1] = omega[i] + alpha[i] * step
        theta[i + 1] = theta[i] + omega[i] * step
        alpha[i + 1] = alpha[i]
    return sub, soul, t, theta, omega, theta_final


def simulation_charge_mobile():
    """
    Réalise une simulation des oscillations de la plateforme pour une charge allant en ligne droite d'un point A à un
    point B

    Returns:
        sub, soul: floats, angle de submersion et de soulevement [rad]
        t, theta, omega: array numpy de même taille représentant le temps [s], l'inclinaison [rad] et la vitesse
            angulaire [rad/s]
        fin_deplacement: int, index correpondant au moment de fin du déplacement de la charge
    """
    # Initalisation de la simulation
    plateforme, charge, grue, masse_totale, enfoncement, sub, soul, coef_amort, step, end, t, theta, omega, alpha = initialisation()[:-1]

    # point d'arrivée
    x_B, z_B = charge.x, charge.z

    # point de départ
    x_A, z_A = 0, 0.15
    charge.x, charge.z = x_A, z_A

    # vitesse de déplacement [m/s]
    v = 0.2

    # position de départ équilibrée
    theta[0] = angle_equilibre(plateforme, charge, 8, grue)

    # trajectoire
    try:
        angle_traj = atan((z_B - z_A) / (x_B - x_A))
    except ZeroDivisionError:  # évite les erreurs de division par 0 qui arrivent lorsque le déplacement est vertical
        angle_traj = pi / 2
    dz = sin(
        angle_traj) * v * step  # calcul du déplacement vertical de la charge à chaque instant de durée step
    dx = cos(
        angle_traj) * v * step  # calcul du déplacement horizontal de la charge à chaque instant de durée step

    fin_deplacement = None

    for i in range(len(t) - 1):
        # vérifie si la charge à atteint l'objectif ou si le mouvement doit continuer
        if not (charge.x - dx < x_B < charge.x + dx):
            charge.x = charge.x + dx  # déplace la charge horizontalement
            charge.z = charge.z + dz  # déplace la charge verticalement
            # calcule une nouvelle approximation d l'inertie
            moment_inertie = plateforme.moment_inertie(enfoncement) + charge.moment_inertie()
        else:
            fin_deplacement = i  # enregistre l'instant de fin du déplacement
        if abs(theta[i]) > min(sub,
                               soul):  # vérifie que l'inclinaison ne dépasse pas les valeurs critiques
            break

        # Calcul du couple total
        couple_archi = masse_totale * 9.81 * xC_final(theta[i], plateforme,
                                                      enfoncement)  # Couple de la force d'Archimède
        couple_amort = - coef_amort * omega[i]
        couple_total = charge.couple() - couple_archi + couple_amort + grue.couple(charge.x, charge.z - plateforme.hauteur + enfoncement)

        alpha[i] = couple_total / moment_inertie
        omega[i + 1] = omega[i] + alpha[i] * step
        theta[i + 1] = theta[i] + omega[i] * step
        alpha[i + 1] = alpha[i]

    return sub, soul, t, theta, omega, fin_deplacement


def graphique_theta(sub, soul, t, theta, theta_final=None):
    """
    Affiche un graphique représentant l'inclinaison en fonction du temps ainsi que les angles critiques.

    Args:
        sub: float, angle de submersion
        soul: float, angle de soulèvement
        t: array numpy, représente le temps
        theta: array numpy, contient les inclinaison correspondantes à chaque instant de t
        theta_final: float, angle d'équilibre (optionnel)
    """
    plt.figure(1)
    plt.axhline(y=sub, color='red', linestyle='--', label="angle de submersion")
    plt.axhline(y=-sub, color='red', linestyle='--')
    plt.axhline(y=soul, color='purple', linestyle='--', label="angle de soulèvement")
    plt.axhline(y=-soul, color='purple', linestyle='--')
    if theta_final:
        plt.axhline(y=theta_final, color='black', linestyle='--', label="angle d'équilibre")
    plt.plot(t, theta, label="theta")
    plt.legend()
    plt.xlabel('temps [s]')
    plt.ylabel("inclinaison de la plateforme [rad]")
    plt.title("Graphique de l'oscillation de la barge en fonction du temps")
    plt.show()


def compare_sim_graph(sub, soul, t, theta1, theta2, theta_final):
    """
    Similaire à graphique_theta(), représente une simulation à charge mobile et une simulation à charge fixe sur le même
    graphique.

    Args:
        sub: float, angle de submersion
        soul: float, angle de soulèvement
        t: array numpy, représente le temps
        theta1: array numpy, contient les inclinaison correspondantes à chaque instant de t pour la simulation à charge fixe
        theta2: array numpy, contient les inclinaison correspondantes à chaque instant de t pour la simulation à charge mobile
        theta_final: float, angle d'équilibre (optionnel)
    """
    plt.figure(1)
    plt.axhline(y=sub, color='red', linestyle='--', label="angle de submersion")
    plt.axhline(y=-sub, color='red', linestyle='--')
    plt.axhline(y=soul, color='purple', linestyle='--', label="angle de soulèvement")
    plt.axhline(y=-soul, color='purple', linestyle='--')
    plt.axhline(y=theta_final, color='black', linestyle='--', label="angle d'équilibre")
    plt.plot(t, theta1, label="charge fixe", color='green')
    plt.plot(t, theta2, label="charge mobile", color='blue')
    plt.legend()
    plt.xlabel('temps [s]')
    plt.ylabel("inclinaison de la plateforme [rad]")
    plt.title("Comparaison de l'inclinaison en fonction du temps :\n Charge fixe et charge mobile")
    plt.show()


def phase_graph(theta, omega, fin=None):
    """
    Affiche un diagramme de phase de la vitesse angulaire en fonction de l'angle.

    Args:
        theta: array numpy, contient les inclinaisons successives calculées lors d'une simulation
        omega: array numpy, contient les vitesses angulaires successive calculées lors d'une simulation
        fin: float, moment de la fin du déplacement de la charge pour les simulations avec charge mobile (optionnel)
    """
    plt.figure(1)
    plt.plot(theta, omega, color='blue')
    if fin:
        plt.axvline(x=theta[fin], color="green", linestyle='--', label="fin du déplacement")
    plt.xlabel('inclinaison [rad]')
    plt.ylabel("vitesse angulaire [rad/s]")
    plt.title("Diagramme de phase des oscillation de la plateforme")
    plt.show()


def compare_phase(theta1, omega1, theta2, omega2):
    """
    Affiche un diagramme de phase de la vitesse angulaire en fonction de l'angle.

    Args:
        theta1: array numpy, contient les inclinaisons successives calculées lors d'une simulation à charge fixe
        omega1: array numpy, contient les vitesses angulaires successive calculées lors d'une simulation à charge fixe
        theta2: array numpy, contient les inclinaisons successives calculées lors d'une simulation à charge mobile
        omega2: array numpy, contient les vitesses angulaires successive calculées lors d'une simulation à charge mobile
    """
    plt.figure(1)
    plt.plot(theta1, omega1, color='green', label="charge fixe")
    plt.plot(theta2, omega2, color='blue', label="charge mobile")
    plt.xlabel('inclinaison [rad]')
    plt.ylabel("vitesse angulaire [rad/s]")
    plt.title("Diagrame de phase des oscillation de la plateforme")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    choice = input("1: Angle d'inclinaison\n2: Graphiques\nchoix:")
    while choice not in ("1","2"):
        print("choix non valide")
        choice = input("1: Angle d'inclinaison\n2: Graphiques\nchoix:")
    if choice == "2":
        sim_1 = simulation_charge_fixe()
        sim_2 = simulation_charge_mobile()
        graphique_theta(sim_1[0], sim_1[1], sim_1[2], sim_1[3], sim_1[5])  # inclinaison charge fixe
        graphique_theta(sim_2[0], sim_2[1], sim_2[2], sim_2[3])
        compare_sim_graph(sim_1[0], sim_1[1], sim_1[2], sim_1[3], sim_2[3],
                          sim_1[5])  # inclinaison comparaison fixe/mobile
        phase_graph(sim_1[3], sim_1[4])  # phase charge fixe
        phase_graph(sim_2[3], sim_2[4], sim_2[5])  # phase charge mobile
        compare_phase(sim_1[3], sim_1[4], sim_2[3], sim_2[4])  # phase comparaison fixe/mobile
    else:
        print("Inclinaison:", degrees(initialisation()[-1]))


