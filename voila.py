# ==================== PROCÉDURE GIT ====================
#
# RÉCUPÉRER les modifications des autres (à faire AVANT de travailler) :
#     Terminal : git pull origin main
#     VS Code  : onglet Source Control (Ctrl+Shift+G) -> "Sync Changes"
#
# ENVOYER tes modifications sur GitHub :
#   Dans le Terminal (Ctrl+`) :
#     1. git pull origin main               (récupérer d'abord ce que les autres ont fait)
#     2. git add voila.py                   (ou "git add ." pour tous les fichiers)
#     3. git commit -m "ce que j'ai changé"
#     4. git push origin main
#
#   Ou dans VS Code (onglet Source Control, Ctrl+Shift+G) :
#     1. Passer la souris sur le fichier sous "Changes" -> cliquer sur le "+"
#        (ou clic droit -> "Stage Changes")
#        ATTENTION : ne pas cliquer sur la flèche courbe, elle efface les modifs !
#     2. Écrire un message dans le champ en haut
#     3. Cliquer sur "Commit"
#     4. Cliquer sur "Sync Changes" (sinon rien n'est envoyé sur GitHub)
#
# VÉRIFIER : "git status" doit afficher "nothing to commit, working tree clean"
#            et "Your branch is up to date with 'origin/main'"
# ========================================================


# ==================== PLAN DU PROGRAMME (numéros = matrice de traçabilité, version actuelle) ====================
#   Étape 1  Lire la CAO .stl / .step                                         E.1, E.1.1          <- FAIT
#   Étape 2  Séparer et identifier chaque shaped hole + perçage               E.2, E.2.1, E.2.6, E.2.7
#   Étape 3  Repère pièce, surface d'entrée, point central, axe, section,     E.2.2, E.2.2.1, E.2.2.2, E.2.3,
#            décomposition shaped hole / perçage                              E.2.3.1, E.2.4, E.2.4.1, E.2.5
#   Étape 4  Découpe en sections perpendiculaires à l'axe                     E.3.2, E.3.2.1, E.3.4, E.3.5
#   Étape 5  Écriture des DXF R12 + dossier de sortie                         E.3, E.3.1, E.3.2.2, E.3.3
#   Étape 6  Paliers de descente de la tête laser                             E.8, E.8.1 à E.8.5
#   Étape 7  Temps d'usinage prévisionnel                                     E.4, E.4.1 à E.4.4
#   Plus tard : E.5 (plan / CAO), E.6 (positions), E.7 (collisions)
#
# Installation (une seule fois) :  python -m pip install trimesh cascadio networkx
#     trimesh  : lecture et géométrie des maillages
#     cascadio : permet à trimesh de lire les .step / .stp
#     networkx : utilisé par trimesh pour séparer les corps (étape 2)
# ===============================================================================================================

import sys
from pathlib import Path

import numpy as np
import trimesh


# ======================= ÉTAPE 1 : LIRE LE FICHIER CAO (E.1, E.1.1) =======================
#
# Un STL décrit la SURFACE de la pièce comme une liste de triangles.
# Un STEP décrit la surface EXACTE (cylindres, plans, courbes...) : trimesh la transforme
# en triangles (on dit qu'il la "maille") pour qu'on travaille pareil dans les deux cas.
#
# Le résultat est un objet "Trimesh" qui contient deux tableaux :
#     mesh.vertices : les sommets, forme (nb_sommets, 3)        -> coordonnées x, y, z en mm
#     mesh.faces    : les triangles, forme (nb_triangles, 3)    -> 3 NUMÉROS de sommets par triangle
#
# Pourquoi des numéros et pas directement les coordonnées ?
#   Dans le fichier, chaque triangle répète ses 3 sommets : un même point apparaît
#   dans 5 ou 6 triangles voisins. On FUSIONNE ces doublons.
#   Résultat : deux triangles qui se touchent partagent le même numéro de sommet,
#   donc on sait qui est voisin de qui. C'est indispensable pour l'étape 2
#   (séparer les shaped holes = trouver les groupes de triangles qui se touchent).

# E.1 critère 3 : le format est reconnu automatiquement à partir de l'extension du fichier
FORMATS = {".stl": "STL", ".step": "STEP", ".stp": "STEP"}

# Deux sommets plus proches que 1e-6 mm (= 1 nanomètre) sont considérés comme le même point.
# Nécessaire pour le STEP : les faces maillées séparément ont des bords qui diffèrent de ~1e-7 mm.
DECIMALES_FUSION = 6


class ErreurLecture(Exception):
    """Erreur "prévue" : le fichier fourni par l'utilisateur est inutilisable (E.1.1).
    On la distingue d'un bug du programme pour renvoyer un code de retour différent."""


def lire_cao(chemin):
    """Lit un fichier .stl ou .step. Renvoie (mesh en mm, format, liste d'informations).
    Lève ErreurLecture avec le nom du fichier et la cause si le fichier est inutilisable."""
    chemin = Path(chemin)
    nom = chemin.name
    infos = []

    # --- E.1.1 : vérifications AVANT de lire, pour donner un message compréhensible ---
    if not chemin.exists():
        raise ErreurLecture(f"{nom} : fichier introuvable ({chemin})")
    if not chemin.is_file():
        raise ErreurLecture(f"{nom} : ce chemin est un dossier, pas un fichier")
    fmt = FORMATS.get(chemin.suffix.lower())
    if fmt is None:
        raise ErreurLecture(f"{nom} : extension « {chemin.suffix or '(aucune)'} » non supportée "
                            f"(acceptées : .stl, .step, .stp)")
    try:
        taille = chemin.stat().st_size
        with open(chemin, "rb"):          # on vérifie juste qu'on a le droit de l'ouvrir
            pass
    except OSError as exc:
        raise ErreurLecture(f"{nom} : fichier non lisible ({exc.strerror})") from exc
    if taille == 0:
        raise ErreurLecture(f"{nom} : fichier vide (0 octet)")

    # --- E.1 : lecture ---
    # force="mesh" : on veut UN seul maillage, même si le fichier contient plusieurs objets
    try:
        mesh = trimesh.load_mesh(chemin, force="mesh")
    except Exception as exc:
        # Pour le STEP, l'erreur brute de cascadio parle d'un fichier temporaire interne : peu utile.
        cause = "contenu non conforme au format STEP" if fmt == "STEP" else str(exc)
        raise ErreurLecture(f"{nom} : fichier corrompu ou illisible comme {fmt} ({cause})") from exc

    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ErreurLecture(f"{nom} : aucun triangle lu (fichier corrompu, tronqué ou d'un autre format)")

    # --- E.1 critère 2 : les dimensions doivent correspondre à la CAO d'origine ---
    # STEP : le fichier contient son unité. trimesh/cascadio le renvoie en MÈTRES -> on repasse en mm.
    # STL  : le format n'a PAS d'unité. Par convention on suppose des mm, et on le dit.
    if mesh.units is None:
        infos.append("unité absente du fichier (normal pour un STL) : coordonnées supposées en mm")
    elif mesh.units != "millimeters":
        infos.append(f"unité du fichier convertie : {mesh.units} -> mm")
        mesh.convert_units("millimeters")

    # Fusion des sommets en double (voir explication en haut de l'étape)
    mesh.merge_vertices(digits_vertex=DECIMALES_FUSION)

    if not np.isfinite(mesh.vertices).all():
        raise ErreurLecture(f"{nom} : coordonnées infinies ou non numériques (fichier corrompu)")

    return mesh, fmt, infos


def verifier_maillage(mesh):
    """Signale les défauts du maillage qui pourraient fausser la suite. Ne bloque pas, avertit.
    E.2.7 critère 2 : sera appelée à l'étape 2 sur CHAQUE shaped hole, pour signaler
    le défaut avec l'identifiant du trou concerné."""
    alertes = []
    # Étanche = chaque arête est partagée par exactement 2 triangles (pas de trou dans la surface).
    # Sans ça, une coupe par un plan donne des contours ouverts.
    if not mesh.is_watertight:
        alertes.append("maillage non étanche (faces manquantes)")
    # Normales cohérentes = tous les triangles "regardent" du même côté de la matière.
    if not mesh.is_winding_consistent:
        alertes.append("orientation des triangles incohérente")
    # Triangles dégénérés = d'aire quasi nulle (3 sommets alignés) : inutiles et sources d'erreurs.
    nb_degeneres = int((mesh.area_faces < 1e-12).sum())
    if nb_degeneres:
        alertes.append(f"{nb_degeneres} triangle(s) d'aire nulle")
    return alertes


# ======================= PROGRAMME PRINCIPAL =======================
# Codes de retour (E.1.1 critère 3) :
#   0 = tout va bien
#   2 = fichier d'entrée inutilisable (message explicite, aucun fichier de sortie produit)
#   3 = erreur inattendue (bug du programme) : message, mais pas de plantage brutal

CHEMIN_PAR_DEFAUT = r"C:\Users\duche\OneDrive\Documents\Maxime\ETUDES SUPERIEURES\SUPMECA\Safran\2 shaped holes.stl"


def main():
    # On peut donner le fichier dans le terminal :  python voila.py "mon fichier.step"
    chemin = sys.argv[1] if len(sys.argv) > 1 else CHEMIN_PAR_DEFAUT
    try:
        mesh, fmt, infos = lire_cao(chemin)
    except ErreurLecture as exc:
        print("ERREUR :", exc)
        return 2
    except Exception as exc:  # filet de sécurité : un bug ne doit pas faire planter brutalement
        print(f"ERREUR INATTENDUE ({type(exc).__name__}) :", exc)
        return 3

    print(f"Format reconnu : {fmt} ({Path(chemin).name})")
    for info in infos:
        print("INFO :", info)
    dims = mesh.bounds[1] - mesh.bounds[0]
    print("Triangles :", len(mesh.faces), "| Sommets :", len(mesh.vertices))
    print("Dimensions (mm) : X = %.4f  Y = %.4f  Z = %.4f" % tuple(dims))
    print("Boîte englobante (mm) : min =", mesh.bounds[0].round(4), " max =", mesh.bounds[1].round(4))
    alertes = verifier_maillage(mesh)
    print("Maillage OK" if not alertes else "ATTENTION : " + " ; ".join(alertes))
    return 0


if __name__ == "__main__":   # s'exécute seulement si on lance ce fichier (pas si on l'importe)
    sys.exit(main())
