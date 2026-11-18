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
#   Étape 2  Séparer chaque corps, trouver le perçage (cylindre) et le        E.2, E.2.1, E.2.5,
#            retirer, garder la partie évasée et sa face d'entrée             E.2.6, E.2.7              <- FAIT
#   Étape 3  Repère pièce, point central de l'entrée, vecteur du perçage,     E.2.2, E.2.2.1, E.2.2.2, E.2.3,
#            section du perçage                                               E.2.3.1, E.2.4, E.2.4.1   <- EN COURS
#            (centre du cercle extérieur du perçage + vecteur : FAIT)
#   Étape 4  Découpe en sections perpendiculaires à l'axe                     E.3.2, E.3.2.1, E.3.4, E.3.5
#   Étape 5  Écriture des DXF R12 + dossier de sortie                         E.3, E.3.1, E.3.2.2, E.3.3
#   Étape 6  Paliers de descente de la tête laser                             E.8, E.8.1 à E.8.5
#   Étape 7  Temps d'usinage prévisionnel                                     E.4, E.4.1 à E.4.4
#   Plus tard : E.5 (plan / CAO), E.6 (positions), E.7 (collisions)
#
# Installation (une seule fois) :  python -m pip install trimesh cascadio networkx scipy "pyglet<2"
#     trimesh  : lecture et géométrie des maillages
#     cascadio : permet à trimesh de lire les .step / .stp
#     networkx, scipy : utilisés par trimesh pour séparer les corps et les surfaces (étape 2)
#     pyglet   : fenêtre 3D (AIDE AU PROTOTYPAGE, à retirer à la fin)
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
    try:
        existe = chemin.exists()
    except OSError as exc:            # Windows : caractère interdit dans le nom, chemin trop long...
        raise ErreurLecture(f"{nom} : chemin invalide ({exc.strerror})") from exc
    if not existe:
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
    E.2.7 critère 2 : appelée à l'étape 2 sur CHAQUE corps, pour signaler le défaut
    avec l'identifiant du trou concerné."""
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


# ======================= ÉTAPE 2 : SÉPARER ET IDENTIFIER LES SHAPED HOLES (E.2, E.2.1, E.2.5, E.2.6, E.2.7) =======================
#
# Le fichier contient les CONTRE-FORMES des trous : chaque shaped hole (partie évasée + perçage)
# est un volume fermé, qui ne touche pas les autres.
#
#   1. On sépare le maillage en "corps" = groupes de triangles qui se touchent      -> separer_corps
#   2. Dans chaque corps, on regroupe les triangles en SURFACES : deux triangles voisins sont
#      dans la même surface si l'angle entre eux est faible. Une arête vive = on change de surface.
#                                                                                    -> surfaces_lisses
#   3. On cherche le PERÇAGE = la partie cylindrique (méthode RANSAC, voir plus bas) -> detecter_percage
#      et on lui rattache sa face de sortie (le "bouchon" au bout du cylindre, PLAT OU COURBE).
#   4. On retire le perçage : il reste la partie ÉVASÉE (le shaped hole proprement dit).
#      Sa face d'entrée (PLATE OU COURBE) = celle par où l'axe du perçage ressort    -> trouver_face_entree
#   5. Selon ce qu'on a trouvé, le corps est classé                                   -> identifier_shaped_holes
#         perçage + 1 face d'entrée          -> shaped hole OK
#         perçage sans partie évasée         -> « perçage sans shaped hole », à vérifier (E.2.7)
#         pas de perçage mais forme de trou  -> « shaped hole sans perçage », à vérifier (E.2.7)
#         perçage CREUX (trou dans un solide) -> pas une contre-forme, écarté (ex. : la plaque du STEP)
#         rien de tout ça, maillage sain     -> pas un shaped hole, écarté

ANGLE_ARETE_VIVE_DEG = 30.0   # au-delà, deux triangles voisins appartiennent à deux surfaces différentes
AIRE_MIN_FACE = 0.01          # une face plane de moins de 1 % de l'aire étudiée est ignorée (triangle parasite)
PLANEITE_MIN = 0.8            # 1 = parfaitement plane ; une paroi qui "tourne" autour du trou est vers 0,4-0,5
                              # 0,8 laisse passer une face d'entrée légèrement bombée

# --- Détection du perçage (RANSAC) ---
# À REVOIR : seuils réglés sur un seul fichier ("2 shaped holes.stl"). Point prévu pour fiabiliser
# le RANSAC (seuils liés à la finesse du maillage, campagne de tests sur plusieurs fichiers).
RANSAC_ESSAIS = 300           # nombre de tirages (300 suffit largement : testé sur 20 graines différentes)
RANSAC_GRAINE = 0             # graine FIXE du hasard -> même résultat à chaque exécution (reproductible)
TOL_RAYON = 0.01              # un sommet est "sur le cylindre" s'il est à moins de 1 % du rayon (et au moins 1 µm)
TOL_NORMALE = 0.05            # |normale . axe| < 0,05 : la normale est perpendiculaire à l'axe (à ~3° près)
TOL_RAYON_LARGE = 0.05        # 1er tri, plus large (5 % du rayon, ~6°) : le cylindre proposé à partir de
TOL_NORMALE_LARGE = 0.1       # 2 triangles est imprécis, il est réajusté avant le tri fin (LO-RANSAC)
TROU_ANGULAIRE_MAX_DEG = 45.0 # un perçage fait le TOUR COMPLET de l'axe : aucun "trou" de plus de 45°
                              # dans la répartition des normales autour de l'axe


def separer_corps(mesh):
    """Sépare le maillage en corps (volumes) distincts, rangés dans un ordre reproductible."""
    messages = []
    corps = []
    # only_watertight=False : on garde aussi les corps troués, pour pouvoir les SIGNALER (E.2.7)
    # au lieu de les faire disparaître sans rien dire.
    # repair=False : par défaut, split() bouche tout seul les petits trous du maillage. Ça cacherait
    # le défaut à l'utilisateur -> on désactive, et c'est verifier_maillage qui le signale.
    for c in mesh.split(only_watertight=False, repair=False):
        if abs(c.volume) < 1e-12:
            messages.append(f"morceau ouvert ou plat de {len(c.faces)} triangle(s) ignoré (volume nul)")
            continue
        if c.volume < 0:              # triangles orientés vers l'intérieur : on les retourne
            c.invert()
            c.metadata["retourne"] = True
        corps.append(c)
    # Numérotation reproductible : on trie par position du centre (x, puis y, puis z).
    # Sans ça, l'ordre dépendrait de l'ordre des triangles dans le fichier.
    corps.sort(key=lambda c: tuple(np.round(c.centroid, 6)))
    return corps, messages


def surfaces_lisses(corps):
    """Renvoie, pour chaque triangle, le numéro de la surface à laquelle il appartient."""
    # face_adjacency        : toutes les paires de triangles qui partagent une arête
    # face_adjacency_angles : l'angle entre les deux triangles de chaque paire (en radians)
    paires_lisses = corps.face_adjacency[corps.face_adjacency_angles < np.radians(ANGLE_ARETE_VIVE_DEG)]
    # "Composantes connexes" : on suit les paires lisses de proche en proche, chaque groupe = une surface
    return trimesh.graph.connected_component_labels(paires_lisses, node_count=len(corps.faces))


def faces_planes(corps, surface, garder=None):
    """Renvoie la liste des numéros de surfaces planes (assez grandes pour compter).
    garder : masque des triangles à étudier (par défaut : tous). Sert à ignorer le perçage."""
    if garder is None:
        garder = np.ones(len(corps.faces), bool)
    nb = surface.max() + 1
    aires_tri = corps.area_faces * garder                     # aire 0 pour les triangles ignorés
    aire = np.bincount(surface, weights=aires_tri, minlength=nb)
    # Somme des normales pondérées par l'aire des triangles, surface par surface :
    #   - face plane : toutes les normales sont identiques, elles s'additionnent -> longueur = aire
    #   - paroi qui tourne autour du trou : les normales se compensent          -> longueur << aire
    somme_normales = np.stack([np.bincount(surface, weights=aires_tri * corps.face_normals[:, k],
                                           minlength=nb) for k in range(3)], axis=1)
    planeite = np.linalg.norm(somme_normales, axis=1) / np.maximum(aire, 1e-300)
    # (une surface entièrement ignorée a une planéité de 0 : elle est exclue d'office)
    return [s for s in range(nb) if aire[s] >= AIRE_MIN_FACE * aire.sum() and planeite[s] >= PLANEITE_MIN]


# ----------------------------- Détection du perçage (partie cylindrique) -----------------------------
#
# MÉTHODE : RANSAC ("RANdom SAmple Consensus"), version "points orientés" (Schnabel et al., 2007,
# la méthode utilisée par CloudCompare pour reconnaître plans, cylindres, cônes dans un nuage de points).
#
# Idée : sur un cylindre, toutes les normales sont PERPENDICULAIRES à l'axe et toutes les droites
# portées par les normales passent PAR l'axe. Donc 2 triangles suffisent pour proposer un cylindre :
#     axe    = normale1 ^ normale2 (produit vectoriel : perpendiculaire aux deux normales)
#     centre = point où se croisent les 2 droites "normales", vues dans le plan perpendiculaire à l'axe
#     rayon  = distance entre ce centre et le triangle
# On répète avec 2 triangles tirés au hasard, et on garde le cylindre sur lequel tombent le plus
# de triangles (en surface). Les triangles de l'évasement ne "votent" pas pour le bon cylindre,
# donc ils ne le perturbent pas : c'est ce qui rend la méthode robuste.
#
# Deux règles écartent les faux cylindres :
#   - rayon < moitié de la taille du trou (sinon c'est une paroi presque plane vue comme un cylindre géant)
#   - le cylindre fait le TOUR COMPLET de l'axe (un congé ou un arrondi d'angle n'en fait qu'une partie)

def base_plan(a):
    """Deux vecteurs unitaires (u, v), perpendiculaires à a et entre eux."""
    ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = ref - (ref @ a) * a
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


def ajuster_cylindre(corps, sel):
    """Cylindre qui passe au mieux par les triangles sélectionnés (moindres carrés)."""
    N, A = corps.face_normals[sel], corps.area_faces[sel]
    # axe = direction la plus perpendiculaire à toutes les normales = plus petit vecteur propre de N^T N
    axe = np.linalg.eigh((N * A[:, None]).T @ N)[1][:, 0]
    # centre et rayon : cercle qui passe au mieux par les sommets projetés dans le plan perpendiculaire
    u, v = base_plan(axe)
    q = corps.vertices[np.unique(corps.faces[sel])] @ np.array([u, v]).T
    sol = np.linalg.lstsq(np.c_[q, np.ones(len(q))], -(q ** 2).sum(axis=1), rcond=None)[0]
    c2 = -sol[:2] / 2
    rayon = float(np.sqrt(max(c2 @ c2 - sol[2], 0.0)))
    return axe, c2[0] * u + c2[1] * v, rayon


def sur_le_cylindre(corps, axe, point, rayon, tol_rayon=TOL_RAYON, tol_normale=TOL_NORMALE):
    """Masque des triangles posés sur le cylindre : leurs 3 sommets sont à la bonne distance de l'axe
    (à tol_rayon x rayon près, et au moins 1 µm) ET leur normale est perpendiculaire à l'axe."""
    P = corps.triangles - point                               # (nb_triangles, 3 sommets, 3 coordonnées)
    dist_axe = np.linalg.norm(P - (P @ axe)[..., None] * axe, axis=2)
    ecart = np.abs(dist_axe - rayon).max(axis=1)              # pire des 3 sommets
    cos_axe = np.abs(corps.face_normals @ axe)
    return (ecart < max(tol_rayon * rayon, 1e-3)) & (cos_axe < tol_normale)


def fait_le_tour(corps, sel, axe):
    """Vrai si les normales des triangles sélectionnés couvrent tout le tour de l'axe."""
    u, v = base_plan(axe)
    N = corps.face_normals[sel]
    angles = np.sort(np.degrees(np.arctan2(N @ v, N @ u)))
    return np.diff(np.r_[angles, angles[0] + 360]).max() <= TROU_ANGULAIRE_MAX_DEG


def detecter_percage(corps, surface):
    """Renvoie None si aucun perçage, sinon un dictionnaire :
        masque : True pour les triangles du perçage (paroi cylindrique + face de sortie)
        paroi  : True pour les triangles de la paroi cylindrique seule
        axe, point, rayon : le cylindre (axe unitaire, un point de l'axe, rayon en mm)"""
    rng = np.random.default_rng(RANSAC_GRAINE)
    G, N, A = corps.triangles_center, corps.face_normals, corps.area_faces
    # On ne tire que parmi les triangles des surfaces COURBES (une face plane ne peut pas être un cylindre)
    planes = faces_planes(corps, surface)
    candidats = np.flatnonzero(~np.isin(surface, planes))
    if len(candidats) < 6:
        return None

    meilleur_score, meilleur = 0.0, None
    for _ in range(RANSAC_ESSAIS):
        i, j = rng.choice(candidats, 2, replace=False)
        axe = np.cross(N[i], N[j])
        if np.linalg.norm(axe) < 0.05:          # normales presque parallèles : axe mal défini
            continue
        axe /= np.linalg.norm(axe)
        # Dans le plan perpendiculaire à l'axe : intersection des 2 droites portées par les normales
        u, v = base_plan(axe)
        B = np.array([u, v]).T
        pi, pj, ni, nj = G[i] @ B, G[j] @ B, N[i] @ B, N[j] @ B
        M = np.c_[ni, -nj]
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        t, _ = np.linalg.solve(M, pj - pi)
        centre2d = pi + t * ni
        rayon = np.linalg.norm(pi - centre2d)
        if rayon < 1e-4 or rayon > 0.5 * corps.scale:
            continue
        point = centre2d[0] * u + centre2d[1] * v
        # 1er tri large, puis on réajuste le cylindre sur ces triangles et on re-trie avec la tolérance fine
        sel = sur_le_cylindre(corps, axe, point, rayon, TOL_RAYON_LARGE, TOL_NORMALE_LARGE)
        if sel.sum() < 6:
            continue
        axe, point, rayon = ajuster_cylindre(corps, sel)
        sel = sur_le_cylindre(corps, axe, point, rayon)
        if sel.sum() < 6 or rayon > 0.5 * corps.scale or not fait_le_tour(corps, sel, axe):
            continue
        score = A[sel].sum()
        if score > meilleur_score:
            meilleur_score, meilleur = score, (axe, point, rayon, sel)

    if meilleur is None:
        return None
    axe, point, rayon, sel = meilleur

    # On ne garde que le plus grand morceau d'un seul tenant (des triangles isolés de l'évasement
    # peuvent tomber par hasard sur le cylindre)
    paires = corps.face_adjacency[sel[corps.face_adjacency].all(axis=1)]
    groupe = trimesh.graph.connected_component_labels(paires, node_count=len(corps.faces))
    masque = sel & (groupe == np.bincount(groupe[sel]).argmax())
    paroi = masque.copy()                       # paroi cylindrique seule (sans la face de sortie)

    # Face de sortie (PLANE OU COURBE) : on la rattache au perçage.
    # On coupe le corps le long de la paroi : il reste des MORCEAUX (triangles voisins hors paroi).
    #   - la partie évasée est un morceau PLUS LARGE que le perçage (c'est la définition de l'évasement)
    #   - la face de sortie est un morceau enfermé par la paroi et qui reste DANS le rayon du perçage
    # Ce critère ne dépend ni de la planéité ni des angles : une face de sortie courbe qui se raccorde
    # en douceur au cylindre (arête < 30°, donc même "surface" lisse que la paroi) est bien trouvée.
    # Il rattache aussi les triangles "aiguilles" des exports CAO (ex. : 6 µm x 6 µm x 12 µm sur
    # votre fichier) posés sur la paroi mais dont la normale ne veut rien dire.
    hors = ~paroi
    paires = corps.face_adjacency[hors[corps.face_adjacency].all(axis=1)]
    morceau = trimesh.graph.connected_component_labels(paires, node_count=len(corps.faces))
    P = corps.triangles - point
    dist_axe = np.linalg.norm(P - (P @ axe)[..., None] * axe, axis=2).max(axis=1)   # sommet le plus éloigné
    for m in np.unique(morceau[hors]):
        dans_m = hors & (morceau == m)
        if dist_axe[dans_m].max() <= rayon + max(TOL_RAYON * rayon, 1e-3):
            masque |= dans_m

    return {"masque": masque, "paroi": paroi, "axe": axe, "point": point, "rayon": rayon}


def croisements_axe(corps, point, axe, garder):
    """Triangles (parmi le masque 'garder') traversés par la droite (point, axe).
    Renvoie (numéros des triangles, abscisses s des points de croisement : point + s.axe).
    Méthode de Möller-Trumbore (2 produits vectoriels par triangle, calculés pour tous à la fois).
    Marche sur n'importe quelle surface, plane ou courbe : c'est ce qui remplace "intersection avec un plan"."""
    idx = np.flatnonzero(garder)
    T = corps.triangles[idx]
    e1, e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    h = np.cross(axe, e2)
    det = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(det) > 1e-30                    # det = 0 : droite parallèle au triangle
    f = np.divide(1.0, det, out=np.zeros_like(det), where=ok)
    d0 = point - T[:, 0]
    u = f * np.einsum("ij,ij->i", d0, h)
    q = np.cross(d0, e1)
    v = f * (q @ axe)
    s = f * np.einsum("ij,ij->i", q, e2)
    tol = 1e-9                                  # un croisement pile sur une arête compte pour les 2 triangles
    dedans = ok & (u >= -tol) & (v >= -tol) & (u + v <= 1 + tol)
    return idx[dedans], s[dedans]


def percage_creux(corps, percage):
    """Vrai si la paroi du perçage est CREUSE (normales vers l'axe) : c'est un trou DANS une pièce
    (ex. : la plaque du STEP), pas une contre-forme. Sur une contre-forme, le perçage est un cylindre
    plein et ses normales s'éloignent de l'axe (separer_corps a orienté tous les corps vers l'extérieur)."""
    p = percage["paroi"]
    r = corps.triangles_center[p] - percage["point"]
    r -= np.outer(r @ percage["axe"], percage["axe"])                 # composante radiale
    return (np.einsum("ij,ij->i", corps.face_normals[p], r) * corps.area_faces[p]).sum() < 0


def trouver_face_entree(corps, surface, percage):
    """Face d'entrée = surface par où l'axe du perçage ressort de la partie évasée (plane ou courbe).
    Si l'axe croise plusieurs fois la partie évasée, c'est le croisement le plus éloigné du perçage
    (= là où la droite quitte le corps pour la dernière fois). None si l'axe ne ressort pas par l'évasement."""
    idx, s = croisements_axe(corps, percage["point"], percage["axe"], ~percage["masque"])
    if len(idx) == 0:
        return None
    s_paroi = (corps.triangles_center[percage["paroi"]] - percage["point"]) @ percage["axe"]
    return int(surface[idx[np.argmax(np.abs(s - s_paroi.mean()))]])


def identifier_shaped_holes(mesh):
    """Renvoie (trous, corps_ignorés, messages). Chaque trou est un dictionnaire :
        nom         : "SH_001", "SH_002"... (nom réutilisé pour les fichiers de sortie, E.3.1)
        corps       : son maillage trimesh (complet : partie évasée + perçage)
        surface     : numéro de surface de chaque triangle
        percage     : résultat de detecter_percage (None si pas trouvé)
        evasement   : masque des triangles de la partie évasée (= tout sauf le perçage)
        face_entree : numéro de la surface d'entrée (None si pas trouvée)
        a_verifier  : liste des défauts détectés (E.2.7), vide si tout va bien"""
    corps, messages = separer_corps(mesh)
    trous, ignores = [], []
    for k, c in enumerate(corps, start=1):
        surface = surfaces_lisses(c)
        defauts = verifier_maillage(c)
        percage = detecter_percage(c, surface)

        if percage is not None:
            if percage_creux(c, percage):
                ignores.append(c)                                     # ex. : plaque percée
                messages.append(f"corps n°{k} ignoré : perçage creux, c'est une pièce percée et non une "
                                f"contre-forme de shaped hole")
                continue
            # On retire le perçage, puis on cherche la face d'entrée (plane ou courbe) dans ce qui reste
            evasement = ~percage["masque"]
            face_entree = trouver_face_entree(c, surface, percage)
            if c.area_faces[evasement].sum() < AIRE_MIN_FACE * c.area:
                defauts.append("perçage sans shaped hole (aucune partie évasée)")       # E.2.7 critère 1
            elif face_entree is None:
                defauts.append("face d'entrée non trouvée : l'axe du perçage ne ressort pas par la partie évasée")
            elif percage["paroi"][surface == face_entree].any():
                # Raccords tous en douceur (arêtes < 30°) : la "surface" trouvée englobe aussi le perçage,
                # on ne peut pas isoler la face d'entrée avec le découpage en surfaces lisses
                defauts.append("face d'entrée non isolée : elle se raccorde en douceur au reste du trou")
                face_entree = None
        else:
            # Pas de perçage trouvé : on regarde si le corps a quand même la forme d'un trou (2 faces planes)
            evasement = np.ones(len(c.faces), bool)
            entrees = faces_planes(c, surface)
            face_entree = entrees[0] if len(entrees) == 1 else None
            if len(entrees) in (1, 2):
                defauts.append("shaped hole sans perçage identifié (aucune partie cylindrique complète)")
            elif not defauts:
                ignores.append(c)
                messages.append(f"corps n°{k} ignoré : pas de perçage et {len(entrees)} face(s) plane(s), "
                                f"ce n'est pas un shaped hole")
                continue
            else:
                defauts.append(f"détection incertaine : pas de perçage, {len(entrees)} face(s) plane(s)")

        nom = f"SH_{len(trous) + 1:03d}"
        # E.2.7 critère 2 : défaut signalé AVEC l'identifiant du trou concerné
        # E.2.7 critère 3 : le trou est quand même gardé, marqué « à vérifier »
        if c.metadata.get("retourne"):
            messages.append(f"{nom} : triangles orientés vers l'intérieur, retournés automatiquement")
        for defaut in defauts:
            messages.append(f"{nom} : À VÉRIFIER - {defaut}")
        trous.append({"nom": nom, "corps": c, "surface": surface, "percage": percage, "evasement": evasement,
                      "face_entree": face_entree, "a_verifier": defauts})

    # E.2.6 : l'absence de shaped hole n'est PAS une erreur -> liste vide + message d'information
    if not trous:
        messages.append("aucun shaped hole détecté dans ce modèle (résultat vide, ce n'est pas une erreur)")
    return trous, ignores, messages


# ======================= ÉTAPE 3 (début) : CENTRE DU CERCLE EXTÉRIEUR ET VECTEUR DU PERÇAGE =======================
#
# Le cercle extérieur = le bout du perçage (en JAUNE) qui débouche sur la face de sortie.
# Attention : le perçage est en général INCLINÉ par rapport à cette face, et la face peut être COURBE.
# Le bout du perçage n'est donc pas un cercle mais une courbe fermée sur la face de sortie. Son centre
# est le point où l'AXE du perçage (le trait noir) traverse la face de sortie : c'est ce qu'on calcule.
#
#   1. Face de sortie = les triangles rattachés au perçage qui ne sont pas sur la paroi cylindrique
#   2. Croisement droite / triangles (croisements_axe) : exact sur une face plane comme sur une face courbe
#   3. Contrôle : vue le long de l'axe, la face de sortie recouvre exactement le disque du perçage
#      (rayon r). Le centre de gravité de cette vue (chaque triangle pesé par son aire VUE LE LONG DE
#      L'AXE) doit donc tomber sur l'axe. Un écart signale une face de sortie incomplète ou un axe mal
#      ajusté -> trou « à vérifier » (E.2.7).
#
# Pourquoi le croisement plutôt que le centre de gravité directement ? L'axe est ajusté sur TOUTE la paroi
# du cylindre (des centaines de triangles), le centre de gravité seulement sur le bouchon (faux si le
# bouchon est entamé). On le garde comme contrôle indépendant : deux méthodes d'accord = résultat fiable.
#
# Sens du vecteur : le RANSAC donne l'axe au signe près (a et -a décrivent le même trait noir).
# On l'ORIENTE du cercle extérieur VERS la partie évasée, pour que le résultat ne dépende pas du hasard.

ECART_CENTRE_MAX = 0.05       # écart toléré entre les 2 calculs du centre : 5 % du rayon du perçage


def centre_et_vecteur_percage(trou):
    """Renvoie None si le calcul est impossible (le défaut est alors ajouté à trou["a_verifier"]), sinon :
        centre  : centre du cercle extérieur (bout jaune du perçage, sur la face de sortie), mm, repère pièce
        vecteur : vecteur UNITAIRE de l'axe du perçage (trait noir), orienté du centre vers la partie évasée
        angle_z : angle entre l'axe et Z, en degrés
        ecart   : distance entre les 2 calculs du centre (contrôle), en mm"""
    p = trou["percage"]
    if p is None:
        return None
    c = trou["corps"]
    a, P0 = p["axe"], p["point"]

    # 1. Face de sortie (plane ou courbe) = triangles rattachés au perçage, hors paroi cylindrique
    dans = p["masque"] & ~p["paroi"]

    # 2. Point où l'axe traverse la face de sortie
    idx, s = croisements_axe(c, P0, a, dans)
    if len(idx) == 0:
        trou["a_verifier"].append("centre du perçage non calculé : l'axe ne traverse pas de face de sortie")
        return None
    centre = P0 + np.median(s) * a              # médiane : un croisement pile sur une arête compte 2 fois

    # Orientation : le vecteur pointe vers la partie évasée (son centre de gravité est "devant" le centre)
    ev = trou["evasement"]
    if ev.any():
        g_ev = (c.triangles_center[ev] * c.area_faces[ev, None]).sum(axis=0) / c.area_faces[ev].sum()
        if (g_ev - centre) @ a < 0:
            a = -a

    # 3. Contrôle croisé : centre de gravité g de la face de sortie VUE LE LONG DE L'AXE
    #    (aire projetée = aire x |n.a|), puis distance entre g et l'axe (sans la composante le long de l'axe)
    A = c.area_faces[dans] * np.abs(c.face_normals[dans] @ a)
    g = (c.triangles_center[dans] * A[:, None]).sum(axis=0) / A.sum()
    ecart = float(np.linalg.norm((g - centre) - ((g - centre) @ a) * a))
    if ecart > ECART_CENTRE_MAX * p["rayon"]:
        trou["a_verifier"].append(f"centre du perçage incertain : les 2 calculs diffèrent de {ecart:.4f} mm")

    return {"centre": centre, "vecteur": a, "angle_z": float(np.degrees(np.arccos(min(1.0, abs(a[2]))))),
            "ecart": ecart}


# ############################################################################################################
# ################   AIDE AU PROTOTYPAGE  —  À SUPPRIMER QUAND LE DÉVELOPPEMENT SERA FINI   ################
# ############################################################################################################
# Affiche le résultat en 3D dans une fenêtre. Mettre AFFICHER_3D = False pour la désactiver.
#   Souris : clic gauche = tourner | molette = zoom | Ctrl + clic gauche = déplacer
#   Clavier : W = fil de fer | A = afficher les axes | Z = recentrer | Q = fermer la fenêtre
# Nécessite :  python -m pip install "pyglet<2"
AFFICHER_3D = True

PALETTE = [("rouge", [220, 50, 50, 255]), ("bleu", [50, 100, 220, 255]), ("vert", [40, 170, 70, 255]),
           ("orange", [240, 140, 20, 255]), ("violet", [150, 60, 200, 255]), ("cyan", [20, 180, 200, 255])]
GRIS_IGNORES = [150, 150, 150, 70]          # dernier nombre = opacité (0 transparent, 255 opaque)
JAUNE_PERCAGE = [255, 205, 0, 255]          # même couleur pour tous les perçages (partie à retirer)
MAGENTA_ETAPE3 = [230, 0, 200, 255]         # centre du cercle extérieur + vecteur du perçage (étape 3)


def fleche(depart, direction, longueur, rayon_percage):
    """Flèche pleine (tige + pointe conique) qui part de 'depart' dans la 'direction' (vecteur unitaire).
    Les épaisseurs sont proportionnelles au rayon du perçage, pour rester lisibles quelle que soit l'échelle."""
    r_tige, r_pointe, h_pointe = 0.12 * rayon_percage, 0.35 * rayon_percage, 1.5 * rayon_percage
    bout_tige = depart + (longueur - h_pointe) * direction
    tige = trimesh.creation.cylinder(radius=r_tige, segment=[depart, bout_tige], sections=16)
    # Le cône de trimesh est construit le long de +Z, base à l'origine : on le tourne vers 'direction'
    pointe = trimesh.creation.cone(radius=r_pointe, height=h_pointe, sections=16)
    pointe.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], direction))
    pointe.apply_translation(bout_tige)
    f = trimesh.util.concatenate([tige, pointe])
    f.visual.face_colors = MAGENTA_ETAPE3
    return f


def afficher_3d(trous, ignores):
    scene = trimesh.Scene()
    print("Affichage 3D :")
    print("   partie évasée = couleur du trou | face d'entrée = même couleur en clair")
    print("   perçage (partie cylindrique retirée) = JAUNE, axe du perçage = trait noir")
    print("   centre du cercle extérieur (bout jaune, sur la face de sortie) = boule MAGENTA")
    print("   vecteur du perçage = flèche MAGENTA, part du centre, pointe vers la partie évasée")
    print("   (touche W = fil de fer, pour voir la flèche à l'intérieur du trou)")
    print("   corps ignorés = gris transparent")
    for i, t in enumerate(trous):
        nom_couleur, couleur = PALETTE[i % len(PALETTE)]
        c = t["corps"].copy()                                   # copie : on ne touche pas aux données de calcul
        couleurs = np.tile(couleur, (len(c.faces), 1))
        if t["face_entree"] is not None:
            claire = (0.35 * np.array(couleur) + 0.65 * 255).astype(int)   # même teinte, mélangée avec du blanc
            couleurs[t["surface"] == t["face_entree"]] = claire
        p = t["percage"]
        if p is not None:
            couleurs[p["masque"]] = JAUNE_PERCAGE
            # Axe du perçage : un trait qui dépasse un peu de chaque côté du cylindre
            s = (c.triangles[p["masque"]].reshape(-1, 3) - p["point"]) @ p["axe"]
            marge = 0.3 * (s.max() - s.min())
            bouts = p["point"] + np.outer([s.min() - marge, s.max() + marge], p["axe"])
            trait = trimesh.load_path(bouts[None, :, :])
            trait.colors = [[0, 0, 0, 255]]
            scene.add_geometry(trait, geom_name=t["nom"] + "_axe")
            # Étape 3 : centre du cercle extérieur (boule MAGENTA) + vecteur du perçage (flèche MAGENTA)
            if t.get("sortie") is not None:
                centre, vecteur = t["sortie"]["centre"], t["sortie"]["vecteur"]
                boule = trimesh.creation.icosphere(subdivisions=2, radius=0.3 * p["rayon"])
                boule.apply_translation(centre)
                boule.visual.face_colors = MAGENTA_ETAPE3
                scene.add_geometry(boule, geom_name=t["nom"] + "_centre")
                # La flèche part du centre et traverse tout le trou : sa pointe sort au-dessus de la face d'entrée
                longueur = ((c.vertices - centre) @ vecteur).max() + 2 * p["rayon"]
                scene.add_geometry(fleche(centre, vecteur, longueur, p["rayon"]), geom_name=t["nom"] + "_vecteur")
        c.visual.face_colors = couleurs
        scene.add_geometry(c, geom_name=t["nom"])
        info = f"perçage r = {p['rayon']:.4f} mm" if p is not None else "pas de perçage"
        if t.get("sortie") is not None:
            ctr, vec = t["sortie"]["centre"], t["sortie"]["vecteur"]
            info += (f", centre ({ctr[0]:.4f}, {ctr[1]:.4f}, {ctr[2]:.4f})"
                     f", vecteur ({vec[0]:.5f}, {vec[1]:.5f}, {vec[2]:.5f})")
        print(f"   {t['nom']} : {nom_couleur} ({info})" + ("  À VÉRIFIER" if t["a_verifier"] else ""))
    for c in ignores:
        g = c.copy()
        g.visual.face_colors = GRIS_IGNORES
        scene.add_geometry(g)
    # Caméra centrée sur les shaped holes, vue inclinée (calculée AVANT d'ajouter le repère,
    # sinon la vue engloberait aussi l'origine (0, 0, 0), souvent loin des trous)
    centre, taille = scene.centroid, scene.scale
    scene.set_camera(angles=(np.radians(55), 0, np.radians(30)), distance=1.6 * taille, center=centre)
    # Repère pièce : POINT ROUGE à l'origine (0, 0, 0), axes X rouge, Y vert, Z bleu (E.2.2.1, vérifié à l'étape 3).
    # Il peut être hors de l'écran : touche Z pour recentrer sur tout, ou molette pour dézoomer.
    print("   point rouge = origine (0, 0, 0) du repère pièce | axes X rouge, Y vert, Z bleu")
    scene.add_geometry(trimesh.creation.axis(origin_size=0.04 * taille, origin_color=[255, 0, 0, 255],
                                             axis_length=0.3 * taille), geom_name="repere")
    scene.show(smooth=False, caption="Shaped holes - aide au prototypage")
# ############################################################################################################
# ################################   FIN DE L'AIDE AU PROTOTYPAGE   ##########################################
# ############################################################################################################


# ======================= PROGRAMME PRINCIPAL =======================
# Codes de retour (E.1.1 critère 3) :
#   0 = tout va bien (y compris : aucun shaped hole, E.2.6)
#   1 = au moins un shaped hole « à vérifier » (E.2.7), les autres sont quand même traités
#   2 = fichier d'entrée inutilisable (message explicite, aucun fichier de sortie produit)
#   3 = erreur inattendue (bug du programme) : message, mais pas de plantage brutal

CHEMIN_PAR_DEFAUT = r"C:\Users\duche\OneDrive\Documents\Maxime\ETUDES SUPERIEURES\SUPMECA\Safran\shaped_hole_V1_contre-formes_v2.stl"


def main():
    # On peut donner le fichier dans le terminal :  python voila.py "mon fichier.step"
    chemin = sys.argv[1] if len(sys.argv) > 1 else CHEMIN_PAR_DEFAUT
    try:
        # --- Étape 1 ---
        mesh, fmt, infos = lire_cao(chemin)
        print(f"Format reconnu : {fmt} ({Path(chemin).name})")
        for info in infos:
            print("INFO :", info)
        print("Triangles :", len(mesh.faces), "| Sommets :", len(mesh.vertices))
        print("Dimensions (mm) : X = %.4f  Y = %.4f  Z = %.4f" % tuple(mesh.extents))
        print("Boîte englobante (mm) : min =", mesh.bounds[0].round(4), " max =", mesh.bounds[1].round(4))

        # --- Étape 2 ---
        trous, ignores, messages = identifier_shaped_holes(mesh)
        for m in messages:
            print("INFO :", m)
        print(f"Shaped holes détectés : {len(trous)}")
        for t in trous:
            p = t["percage"]
            if p is not None:
                angle = np.degrees(np.arccos(min(1.0, abs(p["axe"][2]))))
                info = (f"perçage r = {p['rayon']:.4f} mm, axe {np.round(p['axe'], 3)} ({angle:.1f}° / Z), "
                        f"{int(p['masque'].sum())} triangles retirés")
            else:
                info = "pas de perçage"
            print(f"   {t['nom']} : {len(t['corps'].faces)} triangles, {info} -> "
                  f"{'À VÉRIFIER' if t['a_verifier'] else 'OK'}")

        # --- Étape 3 (début) : centre du cercle extérieur + vecteur du perçage ---
        print("Centre du cercle extérieur et vecteur du perçage (repère pièce, mm) :")
        for t in trous:
            nb_defauts = len(t["a_verifier"])
            t["sortie"] = centre_et_vecteur_percage(t)
            for defaut in t["a_verifier"][nb_defauts:]:
                print(f"INFO : {t['nom']} : À VÉRIFIER - {defaut}")
            s = t["sortie"]
            if s is not None:
                print(f"   {t['nom']} : centre = ({s['centre'][0]:.4f}, {s['centre'][1]:.4f}, {s['centre'][2]:.4f})"
                      f" | vecteur = ({s['vecteur'][0]:.5f}, {s['vecteur'][1]:.5f}, {s['vecteur'][2]:.5f})"
                      f" | {s['angle_z']:.2f}° / Z | contrôle : écart {s['ecart'] * 1000:.2f} µm")

    except ErreurLecture as exc:
        print("ERREUR :", exc)
        return 2
    except Exception as exc:  # filet de sécurité : un bug, à n'importe quelle étape, ne fait pas planter brutalement
        print(f"ERREUR INATTENDUE ({type(exc).__name__}) :", exc)
        return 3

    if AFFICHER_3D:   # AIDE AU PROTOTYPAGE - À SUPPRIMER
        afficher_3d(trous, ignores)

    return 1 if any(t["a_verifier"] for t in trous) else 0


if __name__ == "__main__":   # s'exécute seulement si on lance ce fichier (pas si on l'importe)
    sys.exit(main())
