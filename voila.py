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


# Installation (une seule fois) :  python -m pip install trimesh
from pathlib import Path

import trimesh


# ======================= ÉTAPE 1 : LIRE LE FICHIER CAO (E.1, E.1.1, E.1.2) =======================
#
# Un STL décrit la SURFACE de la pièce comme une liste de triangles.
# trimesh lit le fichier (binaire ou texte, il reconnaît tout seul) et le range dans un objet
# "Trimesh" qui contient deux tableaux :
#     mesh.vertices : les sommets, forme (nb_sommets, 3)        -> coordonnées x, y, z en mm
#     mesh.faces    : les triangles, forme (nb_triangles, 3)    -> 3 NUMÉROS de sommets par triangle
#
# Pourquoi des numéros et pas directement les coordonnées ?
#   Dans le fichier STL, chaque triangle répète ses 3 sommets : un même point apparaît
#   dans 5 ou 6 triangles voisins. trimesh FUSIONNE ces doublons (process=True).
#   Résultat : deux triangles qui se touchent partagent le même numéro de sommet,
#   donc on sait qui est voisin de qui. C'est indispensable pour l'étape 2
#   (séparer les shaped holes = trouver les groupes de triangles qui se touchent).

FORMATS_ACCEPTES = {".stl"}   # on ajoutera .step plus tard (il faut une bibliothèque en plus)


def lire_cao(chemin):
    """Lit le fichier CAO et renvoie un objet trimesh.Trimesh. S'arrête avec un message clair sinon."""
    chemin = Path(chemin)

    # --- E.1.1 : vérifications AVANT de lire, pour donner un message compréhensible ---
    if not chemin.is_file():
        raise ValueError(f"Fichier introuvable : {chemin}")
    if chemin.suffix.lower() not in FORMATS_ACCEPTES:
        raise ValueError(f"{chemin.name} : extension « {chemin.suffix} » non supportée (accepté : .stl)")

    # --- E.1 : lecture ---
    # force="mesh" : on veut UN seul maillage, même si le fichier contient plusieurs objets
    # process=True : fusion des sommets en double (voir explication plus haut)
    try:
        mesh = trimesh.load_mesh(chemin, force="mesh", process=True)
    except Exception as exc:
        raise ValueError(f"{chemin.name} : lecture impossible ({exc})") from exc

    if len(mesh.faces) == 0:
        raise ValueError(f"{chemin.name} : le fichier ne contient aucun triangle")

    return mesh


def verifier_maillage(mesh):
    """E.1.2 : signale les défauts qui pourraient fausser la suite. Ne bloque pas, avertit."""
    alertes = []
    # Étanche = chaque arête est partagée par exactement 2 triangles (pas de trou dans la surface).
    # Sans ça, une coupe par un plan donne des contours ouverts.
    if not mesh.is_watertight:
        alertes.append("le maillage n'est pas étanche (trous dans la surface)")
    # Normales cohérentes = tous les triangles "regardent" vers l'extérieur de la matière.
    if not mesh.is_winding_consistent:
        alertes.append("l'orientation des triangles est incohérente")
    # Triangles dégénérés = d'aire quasi nulle (3 sommets alignés) : inutiles et sources d'erreurs.
    nb_degeneres = int((mesh.area_faces < 1e-12).sum())
    if nb_degeneres:
        alertes.append(f"{nb_degeneres} triangle(s) d'aire nulle")
    return alertes


# ======================= TEST (s'exécute seulement si on lance ce fichier) =======================
if __name__ == "__main__":
    CHEMIN = r"C:\Users\duche\OneDrive\Documents\Maxime\ETUDES SUPERIEURES\SUPMECA\Safran\2 shaped holes.stl"

    mesh = lire_cao(CHEMIN)
    print("Triangles :", len(mesh.faces), "| Sommets (après fusion des doublons) :", len(mesh.vertices))
    print("Boîte englobante (mm) : min =", mesh.bounds[0], " max =", mesh.bounds[1])
    print("Volume (mm³) :", round(mesh.volume, 5))

    alertes = verifier_maillage(mesh)
    print("Maillage OK" if not alertes else "ATTENTION : " + " ; ".join(alertes))
