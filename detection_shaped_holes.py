# -*- coding: utf-8 -*-
"""
DÉTECTION DES SHAPED HOLES ET DE LEURS PERÇAGES
===============================================

Entrée
    Un fichier CAO .stl, .step ou .stp qui contient les CONTRE-FORMES des shaped holes :
    chaque trou est un volume fermé, composé d'une partie évasée et d'un perçage cylindrique.

Sorties
    1. Un compte rendu dans le terminal.
    2. Un fichier texte « <nom du fichier CAO>_percages.txt », écrit à côté du fichier CAO.
       Pour chaque perçage : le point central sur la surface de sortie et le vecteur de son axe.
    3. Une fenêtre 3D :
         - chaque shaped hole a sa propre couleur (sa face d'entrée dans la même teinte, plus claire) ;
         - les perçages sont en JAUNE, leur axe est un trait noir ;
         - le point central de sortie (boule) et le sens de l'axe (flèche) sont en MAGENTA.

Utilisation
    python detection_shaped_holes.py "C:\\chemin\\piece.step"
    python detection_shaped_holes.py piece.stl --sans-3d            (pas de fenêtre 3D)
    python detection_shaped_holes.py piece.stl --sortie res.txt     (autre nom pour le fichier texte)
    python detection_shaped_holes.py                                (une fenêtre demande le fichier)

Installation (une seule fois)
    python -m pip install trimesh cascadio networkx scipy "pyglet<2"

Conventions
    - Unités : millimètres. Un STEP est converti en mm. Un STL n'a pas d'unité : il est supposé en mm.
    - Repère : celui du fichier CAO (repère pièce), sans aucune transformation.
    - Vecteur de l'axe : unitaire, orienté du point de sortie VERS la partie évasée.
    - Les faces d'entrée et de sortie peuvent être planes ou courbes.

Codes de retour (utiles pour un enchaînement automatique)
    0 = tout va bien (y compris : aucun shaped hole dans le fichier)
    1 = au moins un shaped hole « À VÉRIFIER » ; les autres sont traités normalement
    2 = fichier d'entrée inutilisable (message explicite, aucun fichier produit)
    3 = fichier texte impossible à écrire, ou erreur inattendue
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import trimesh


# =====================================================================================================
# PARAMÈTRES
# =====================================================================================================

# --- Lecture ---
FORMATS = {".stl": "STL", ".step": "STEP", ".stp": "STEP"}
DECIMALES_FUSION = 6            # deux sommets à moins de 1e-6 mm sont un seul point (bords des faces d'un STEP)

# --- Découpage de chaque trou en surfaces ---
ANGLE_ARETE_VIVE_DEG = 30.0     # au-delà, deux triangles voisins appartiennent à deux surfaces différentes
AIRE_MIN_FACE = 0.01            # une face de moins de 1 % de l'aire étudiée est ignorée (triangle parasite)
PLANEITE_MIN = 0.8              # 1 = parfaitement plane ; une paroi qui tourne autour du trou est vers 0,4-0,5

# --- Détection du perçage (méthode RANSAC, voir la section « Détection du perçage ») ---
RANSAC_ESSAIS = 300             # nombre de cylindres candidats essayés par trou
RANSAC_GRAINE = 0               # graine fixe : même résultat à chaque exécution
TOL_RAYON = 0.01                # un triangle est sur le cylindre à 1 % du rayon près (et au moins 1 µm)
TOL_NORMALE = 0.05              # ... et si sa normale est perpendiculaire à l'axe (|n.axe| < 0,05, ~3°)
TOL_RAYON_LARGE = 0.05          # 1er tri plus large, avant de réajuster le cylindre (5 % du rayon)
TOL_NORMALE_LARGE = 0.1         # ... et ~6° pour la normale
TROU_ANGULAIRE_MAX_DEG = 45.0   # un perçage fait le tour complet de l'axe : aucun manque de plus de 45°

# --- Contrôle du point central de sortie ---
ECART_CENTRE_MAX = 0.05         # écart toléré entre les deux calculs du point : 5 % du rayon du perçage

# --- Affichage 3D (nom affiché dans le terminal, couleur RVB + opacité) ---
PALETTE = [("rouge", [220, 50, 50, 255]), ("bleu", [50, 100, 220, 255]), ("vert", [40, 170, 70, 255]),
           ("orange", [240, 140, 20, 255]), ("violet", [150, 60, 200, 255]), ("cyan", [20, 180, 200, 255]),
           ("marron", [140, 90, 50, 255]), ("bleu marine", [30, 40, 120, 255]),
           ("vert olive", [120, 130, 40, 255]), ("gris foncé", [90, 90, 90, 255])]
JAUNE_PERCAGE = [255, 205, 0, 255]
MAGENTA_SORTIE = [230, 0, 200, 255]
GRIS_IGNORES = [150, 150, 150, 70]          # dernier nombre = opacité (0 transparent, 255 opaque)


# =====================================================================================================
# DONNÉES
# =====================================================================================================

class ErreurLecture(Exception):
    """Le fichier CAO fourni est inutilisable : on l'explique sans planter."""


class ErreurEcriture(Exception):
    """Le fichier texte de résultats n'a pas pu être écrit."""


@dataclass
class Percage:
    """La partie cylindrique d'un shaped hole."""
    axe: np.ndarray             # direction de l'axe (vecteur unitaire, sens quelconque à ce stade)
    point: np.ndarray           # un point de l'axe (mm)
    rayon: float                # rayon du cylindre (mm)
    paroi: np.ndarray           # masque des triangles de la paroi cylindrique
    masque: np.ndarray          # masque des triangles du perçage : paroi + face de sortie


@dataclass
class ShapedHole:
    """Un shaped hole reconnu et tout ce qu'on a calculé sur lui."""
    nom: str                                 # "SH_001", "SH_002"...
    corps: trimesh.Trimesh                   # son maillage complet (partie évasée + perçage)
    surface: np.ndarray                      # numéro de surface de chaque triangle
    percage: Optional[Percage]               # None si aucun perçage n'a été trouvé
    evasement: np.ndarray                    # masque des triangles de la partie évasée
    face_entree: Optional[int]               # numéro de la surface d'entrée (None si non isolée)
    a_verifier: List[str] = field(default_factory=list)   # défauts détectés, vide si tout va bien
    centre_sortie: Optional[np.ndarray] = None   # point central du perçage sur la surface de sortie (mm)
    vecteur: Optional[np.ndarray] = None         # axe du perçage, unitaire, vers la partie évasée
    ecart_controle: Optional[float] = None       # écart entre les deux calculs du point central (mm)

    @property
    def angle_z(self) -> Optional[float]:
        """Angle entre l'axe du perçage et l'axe Z du repère pièce, en degrés."""
        if self.vecteur is None:
            return None
        return float(np.degrees(np.arccos(min(1.0, abs(self.vecteur[2])))))


# =====================================================================================================
# 1. LECTURE DU FICHIER CAO
# =====================================================================================================
# Un STL décrit la surface de la pièce par des triangles. Un STEP décrit la surface exacte (cylindres,
# plans, surfaces courbes) : il est transformé en triangles pour être traité de la même façon.

def lire_cao(chemin):
    """Lit un fichier .stl, .step ou .stp. Renvoie (maillage en mm, format, liste d'informations).
    Lève ErreurLecture, avec le nom du fichier et la cause, si le fichier est inutilisable."""
    chemin = Path(chemin)
    nom = chemin.name
    infos = []

    # Vérifications avant lecture, pour donner un message compréhensible
    try:
        existe = chemin.exists()
    except OSError as exc:                      # caractère interdit dans le nom, chemin trop long...
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
        with open(chemin, "rb"):
            pass
    except OSError as exc:
        raise ErreurLecture(f"{nom} : fichier non lisible ({exc.strerror})") from exc
    if taille == 0:
        raise ErreurLecture(f"{nom} : fichier vide (0 octet)")

    # Lecture : un seul maillage, même si le fichier contient plusieurs objets
    try:
        mesh = trimesh.load_mesh(chemin, force="mesh")
    except Exception as exc:
        cause = "contenu non conforme au format STEP" if fmt == "STEP" else str(exc)
        raise ErreurLecture(f"{nom} : fichier corrompu ou illisible comme {fmt} ({cause})") from exc
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ErreurLecture(f"{nom} : aucun triangle lu (fichier corrompu, tronqué ou d'un autre format)")

    # Unités : le STEP contient la sienne (lue en mètres), le STL n'en a pas (supposé en mm)
    if mesh.units is None:
        infos.append("unité absente du fichier (normal pour un STL) : coordonnées supposées en mm")
    elif mesh.units != "millimeters":
        infos.append(f"unité du fichier convertie : {mesh.units} -> mm")
        mesh.convert_units("millimeters")

    # Fusion des sommets en double : deux triangles qui se touchent partagent alors le même sommet,
    # ce qui permet de savoir quels triangles sont voisins (indispensable pour séparer les trous)
    mesh.merge_vertices(digits_vertex=DECIMALES_FUSION)
    if not np.isfinite(mesh.vertices).all():
        raise ErreurLecture(f"{nom} : coordonnées infinies ou non numériques (fichier corrompu)")
    return mesh, fmt, infos


def verifier_maillage(mesh):
    """Liste les défauts du maillage qui pourraient fausser les calculs (ne bloque pas)."""
    alertes = []
    if not mesh.is_watertight:                  # chaque arête doit être partagée par exactement 2 triangles
        alertes.append("maillage non étanche (faces manquantes)")
    if not mesh.is_winding_consistent:          # tous les triangles doivent "regarder" du même côté
        alertes.append("orientation des triangles incohérente")
    nb_degeneres = int((mesh.area_faces < 1e-12).sum())
    if nb_degeneres:
        alertes.append(f"{nb_degeneres} triangle(s) d'aire nulle")
    return alertes


# =====================================================================================================
# 2. OUTILS GÉOMÉTRIQUES
# =====================================================================================================

def base_plan(a):
    """Deux vecteurs unitaires (u, v), perpendiculaires à a et entre eux."""
    ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = ref - (ref @ a) * a
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


def croisements_axe(corps, point, axe, garder):
    """Triangles (parmi le masque 'garder') traversés par la droite (point, axe).
    Renvoie (numéros des triangles, abscisses s des croisements : point + s.axe).
    Méthode de Möller-Trumbore, calculée pour tous les triangles à la fois.
    Fonctionne sur n'importe quelle surface, plane ou courbe."""
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


# =====================================================================================================
# 3. SÉPARATION DES SHAPED HOLES ET DÉCOUPAGE EN SURFACES
# =====================================================================================================

def separer_corps(mesh):
    """Sépare le maillage en volumes distincts (un par shaped hole), rangés dans un ordre reproductible."""
    messages, corps = [], []
    # only_watertight=False et repair=False : un corps abîmé est gardé tel quel pour être SIGNALÉ,
    # au lieu d'être réparé ou écarté sans rien dire
    for c in mesh.split(only_watertight=False, repair=False):
        if abs(c.volume) < 1e-12:
            messages.append(f"morceau ouvert ou plat de {len(c.faces)} triangle(s) ignoré (volume nul)")
            continue
        if c.volume < 0:                        # triangles orientés vers l'intérieur : on les retourne
            c.invert()
            c.metadata["retourne"] = True
        corps.append(c)
    # Numérotation indépendante de l'ordre des triangles dans le fichier : tri par position du centre
    corps.sort(key=lambda c: tuple(np.round(c.centroid, 6)))
    return corps, messages


def surfaces_lisses(corps):
    """Numéro de surface de chaque triangle : deux triangles voisins sont dans la même surface
    si l'angle entre eux est inférieur à ANGLE_ARETE_VIVE_DEG (une arête vive sépare deux surfaces)."""
    lisses = corps.face_adjacency[corps.face_adjacency_angles < np.radians(ANGLE_ARETE_VIVE_DEG)]
    return trimesh.graph.connected_component_labels(lisses, node_count=len(corps.faces))


def faces_planes(corps, surface, garder=None):
    """Numéros des surfaces planes assez grandes pour compter (parmi les triangles du masque 'garder').
    Une surface est plane si ses normales, pondérées par l'aire, s'additionnent sans se compenser."""
    if garder is None:
        garder = np.ones(len(corps.faces), bool)
    nb = surface.max() + 1
    aires_tri = corps.area_faces * garder
    aire = np.bincount(surface, weights=aires_tri, minlength=nb)
    somme_normales = np.stack([np.bincount(surface, weights=aires_tri * corps.face_normals[:, k],
                                           minlength=nb) for k in range(3)], axis=1)
    planeite = np.linalg.norm(somme_normales, axis=1) / np.maximum(aire, 1e-300)
    return [s for s in range(nb) if aire[s] >= AIRE_MIN_FACE * aire.sum() and planeite[s] >= PLANEITE_MIN]


# =====================================================================================================
# 4. DÉTECTION DU PERÇAGE (PARTIE CYLINDRIQUE)
# =====================================================================================================
# Méthode RANSAC « à points orientés » (Schnabel et al., 2007, utilisée par CloudCompare).
# Sur un cylindre, toutes les normales sont perpendiculaires à l'axe et passent par l'axe. Deux triangles
# suffisent donc pour proposer un cylindre :
#     axe    = normale1 ^ normale2
#     centre = croisement des deux droites portées par les normales (dans le plan perpendiculaire à l'axe)
#     rayon  = distance entre ce centre et le triangle
# On essaie RANSAC_ESSAIS paires tirées au hasard et on garde le cylindre qui porte la plus grande aire
# de triangles. Les triangles de la partie évasée ne votent pas pour le bon cylindre : ils ne le
# perturbent pas. Deux règles écartent les faux cylindres :
#     - rayon inférieur à la moitié de la taille du trou (sinon c'est une paroi presque plane) ;
#     - le cylindre fait le tour complet de l'axe (un congé ou un arrondi n'en fait qu'une partie).

def ajuster_cylindre(corps, sel):
    """Cylindre qui passe au mieux par les triangles sélectionnés (moindres carrés)."""
    N, A = corps.face_normals[sel], corps.area_faces[sel]
    axe = np.linalg.eigh((N * A[:, None]).T @ N)[1][:, 0]     # direction la plus perpendiculaire aux normales
    u, v = base_plan(axe)
    q = corps.vertices[np.unique(corps.faces[sel])] @ np.array([u, v]).T
    sol = np.linalg.lstsq(np.c_[q, np.ones(len(q))], -(q ** 2).sum(axis=1), rcond=None)[0]
    c2 = -sol[:2] / 2
    rayon = float(np.sqrt(max(c2 @ c2 - sol[2], 0.0)))
    return axe, c2[0] * u + c2[1] * v, rayon


def sur_le_cylindre(corps, axe, point, rayon, tol_rayon=TOL_RAYON, tol_normale=TOL_NORMALE):
    """Masque des triangles posés sur le cylindre : leurs 3 sommets à la bonne distance de l'axe
    ET leur normale perpendiculaire à l'axe."""
    P = corps.triangles - point
    dist_axe = np.linalg.norm(P - (P @ axe)[..., None] * axe, axis=2)
    ecart = np.abs(dist_axe - rayon).max(axis=1)
    cos_axe = np.abs(corps.face_normals @ axe)
    return (ecart < max(tol_rayon * rayon, 1e-3)) & (cos_axe < tol_normale)


def fait_le_tour(corps, sel, axe):
    """Vrai si les normales des triangles sélectionnés couvrent tout le tour de l'axe."""
    u, v = base_plan(axe)
    N = corps.face_normals[sel]
    angles = np.sort(np.degrees(np.arctan2(N @ v, N @ u)))
    return np.diff(np.r_[angles, angles[0] + 360]).max() <= TROU_ANGULAIRE_MAX_DEG


def detecter_percage(corps, surface):
    """Cherche le perçage cylindrique du corps et lui rattache sa face de sortie.
    Renvoie un Percage, ou None si aucun cylindre complet n'est trouvé."""
    rng = np.random.default_rng(RANSAC_GRAINE)
    G, N, A = corps.triangles_center, corps.face_normals, corps.area_faces
    # On ne tire que parmi les triangles des surfaces courbes : une face plane ne peut pas être un cylindre
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
        # Dans le plan perpendiculaire à l'axe : croisement des deux droites portées par les normales
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
        # 1er tri large, réajustement du cylindre sur ces triangles, puis tri fin
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

    # Paroi = le plus grand morceau d'un seul tenant (des triangles isolés de la partie évasée
    # peuvent tomber par hasard sur le cylindre)
    paires = corps.face_adjacency[sel[corps.face_adjacency].all(axis=1)]
    groupe = trimesh.graph.connected_component_labels(paires, node_count=len(corps.faces))
    paroi = sel & (groupe == np.bincount(groupe[sel]).argmax())

    # Face de sortie (plane ou courbe). On coupe le corps le long de la paroi : il reste des morceaux.
    #   - la partie évasée est un morceau PLUS LARGE que le perçage (c'est la définition de l'évasement) ;
    #   - la face de sortie est un morceau enfermé par la paroi, qui reste DANS le rayon du perçage.
    # Ce critère ne dépend ni de la planéité ni des angles : une face de sortie courbe qui se raccorde
    # en douceur au cylindre est bien trouvée. Il rattache aussi les triangles « aiguilles » que
    # produisent certains exports CAO, posés sur la paroi mais avec une normale sans signification.
    masque = paroi.copy()
    hors = ~paroi
    paires = corps.face_adjacency[hors[corps.face_adjacency].all(axis=1)]
    morceau = trimesh.graph.connected_component_labels(paires, node_count=len(corps.faces))
    P = corps.triangles - point
    dist_axe = np.linalg.norm(P - (P @ axe)[..., None] * axe, axis=2).max(axis=1)   # sommet le plus éloigné
    for m in np.unique(morceau[hors]):
        dans_m = hors & (morceau == m)
        if dist_axe[dans_m].max() <= rayon + max(TOL_RAYON * rayon, 1e-3):
            masque |= dans_m

    return Percage(axe=axe, point=point, rayon=rayon, paroi=paroi, masque=masque)


def percage_creux(corps, percage):
    """Vrai si la paroi du perçage est creuse (normales tournées vers l'axe) : c'est alors un trou DANS
    une pièce, pas une contre-forme. Sur une contre-forme, le perçage est un cylindre plein et ses
    normales s'éloignent de l'axe (tous les corps ont été orientés vers l'extérieur)."""
    p = percage.paroi
    r = corps.triangles_center[p] - percage.point
    r -= np.outer(r @ percage.axe, percage.axe)                       # composante radiale
    return (np.einsum("ij,ij->i", corps.face_normals[p], r) * corps.area_faces[p]).sum() < 0


def trouver_face_entree(corps, surface, percage):
    """Face d'entrée (plane ou courbe) = surface par où l'axe du perçage ressort de la partie évasée.
    Si l'axe la croise plusieurs fois, on garde le croisement le plus éloigné du perçage.
    Renvoie None si l'axe ne ressort pas par la partie évasée."""
    idx, s = croisements_axe(corps, percage.point, percage.axe, ~percage.masque)
    if len(idx) == 0:
        return None
    s_paroi = (corps.triangles_center[percage.paroi] - percage.point) @ percage.axe
    return int(surface[idx[np.argmax(np.abs(s - s_paroi.mean()))]])


# =====================================================================================================
# 5. IDENTIFICATION DES SHAPED HOLES
# =====================================================================================================
# Chaque corps est classé :
#     perçage + partie évasée                -> shaped hole
#     perçage sans partie évasée             -> gardé, « À VÉRIFIER » (perçage sans shaped hole)
#     pas de perçage mais forme de trou      -> gardé, « À VÉRIFIER » (shaped hole sans perçage)
#     perçage creux (trou dans un solide)    -> écarté : ce n'est pas une contre-forme
#     rien de tout ça, maillage sain         -> écarté : ce n'est pas un shaped hole
# « À VÉRIFIER » : un défaut a été détecté sur ce trou. Il est gardé et traité, et la cause est indiquée.

def identifier_shaped_holes(mesh):
    """Renvoie (shaped holes, corps écartés, messages d'information)."""
    corps, messages = separer_corps(mesh)
    trous, ignores = [], []
    for k, c in enumerate(corps, start=1):
        print(f"   corps {k}/{len(corps)} ({len(c.faces)} triangles)...", flush=True)
        surface = surfaces_lisses(c)
        defauts = verifier_maillage(c)
        percage = detecter_percage(c, surface)

        if percage is not None:
            if percage_creux(c, percage):
                ignores.append(c)
                messages.append(f"corps n°{k} écarté : perçage creux, c'est une pièce percée et non une "
                                f"contre-forme de shaped hole")
                continue
            evasement = ~percage.masque
            face_entree = trouver_face_entree(c, surface, percage)
            if c.area_faces[evasement].sum() < AIRE_MIN_FACE * c.area:
                defauts.append("perçage sans shaped hole (aucune partie évasée)")
            elif face_entree is None:
                defauts.append("face d'entrée non trouvée : l'axe du perçage ne ressort pas par la partie évasée")
            elif percage.paroi[surface == face_entree].any():
                # Tous les raccords sont en douceur (arêtes < 30°) : la surface trouvée englobe aussi le
                # perçage, la face d'entrée ne peut pas être isolée par le découpage en surfaces
                defauts.append("face d'entrée non isolée : elle se raccorde en douceur au reste du trou")
                face_entree = None
        else:
            # Pas de perçage : le corps a-t-il quand même la forme d'un trou (1 ou 2 faces planes) ?
            evasement = np.ones(len(c.faces), bool)
            entrees = faces_planes(c, surface)
            face_entree = entrees[0] if len(entrees) == 1 else None
            if len(entrees) in (1, 2):
                defauts.append("shaped hole sans perçage identifié (aucune partie cylindrique complète)")
            elif not defauts:
                ignores.append(c)
                messages.append(f"corps n°{k} écarté : pas de perçage et {len(entrees)} face(s) plane(s), "
                                f"ce n'est pas un shaped hole")
                continue
            else:
                defauts.append(f"détection incertaine : pas de perçage, {len(entrees)} face(s) plane(s)")

        nom = f"SH_{len(trous) + 1:03d}"
        if c.metadata.get("retourne"):
            messages.append(f"{nom} : triangles orientés vers l'intérieur, retournés automatiquement")
        trous.append(ShapedHole(nom=nom, corps=c, surface=surface, percage=percage, evasement=evasement,
                                face_entree=face_entree, a_verifier=defauts))

    if not trous:
        messages.append("aucun shaped hole détecté dans ce modèle (résultat vide, ce n'est pas une erreur)")
    return trous, ignores, messages


# =====================================================================================================
# 6. POINT CENTRAL DE SORTIE ET AXE DU PERÇAGE
# =====================================================================================================
# Le perçage est en général incliné par rapport à la surface de sortie, et cette surface peut être
# courbe : le bout du perçage n'est donc pas un cercle mais une courbe fermée. Son point central est
# le point où l'AXE du perçage traverse la surface de sortie.
#   1. Surface de sortie = triangles rattachés au perçage, hors paroi cylindrique.
#   2. Croisement de l'axe avec ces triangles : exact sur une surface plane comme courbe.
#   3. Contrôle indépendant : vue le long de l'axe, la surface de sortie recouvre exactement le disque
#      du perçage. Son centre de gravité (chaque triangle pesé par son aire vue le long de l'axe) doit
#      donc tomber sur l'axe. Un écart trop grand signale un résultat douteux : « À VÉRIFIER ».
# Le point retenu est le croisement : l'axe est ajusté sur toute la paroi du cylindre (des milliers de
# triangles), alors que le centre de gravité ne repose que sur la surface de sortie.
# Sens du vecteur : la détection donne l'axe au signe près. Il est orienté du point de sortie vers la
# partie évasée, pour que le résultat ne dépende pas du hasard.

def calculer_point_de_sortie(trou):
    """Remplit trou.centre_sortie, trou.vecteur et trou.ecart_controle.
    Si le calcul est impossible, ils restent à None et la cause est ajoutée à trou.a_verifier."""
    p = trou.percage
    if p is None:
        return
    c = trou.corps
    a, P0 = p.axe, p.point

    # 1. Surface de sortie (plane ou courbe)
    sortie = p.masque & ~p.paroi

    # 2. Point où l'axe traverse la surface de sortie
    idx, s = croisements_axe(c, P0, a, sortie)
    if len(idx) == 0:
        trou.a_verifier.append("point central non calculé : l'axe ne traverse pas de surface de sortie")
        return
    centre = P0 + np.median(s) * a              # médiane : un croisement pile sur une arête compte 2 fois

    # Orientation : le vecteur pointe vers la partie évasée
    ev = trou.evasement
    if ev.any():
        g_ev = (c.triangles_center[ev] * c.area_faces[ev, None]).sum(axis=0) / c.area_faces[ev].sum()
        if (g_ev - centre) @ a < 0:
            a = -a

    # 3. Contrôle : distance entre l'axe et le centre de gravité de la sortie vue le long de l'axe
    A = c.area_faces[sortie] * np.abs(c.face_normals[sortie] @ a)
    g = (c.triangles_center[sortie] * A[:, None]).sum(axis=0) / A.sum()
    ecart = float(np.linalg.norm((g - centre) - ((g - centre) @ a) * a))
    if ecart > ECART_CENTRE_MAX * p.rayon:
        trou.a_verifier.append(f"point central incertain : les 2 calculs diffèrent de {ecart:.4f} mm")

    trou.centre_sortie, trou.vecteur, trou.ecart_controle = centre, a, ecart


# =====================================================================================================
# 7. FICHIER TEXTE DES RÉSULTATS
# =====================================================================================================

def ecrire_fichier_texte(chemin_txt, chemin_cao, fmt, trous):
    """Écrit un tableau lisible : une ligne par shaped hole, colonnes séparées par des espaces.
    Les lignes qui commencent par # sont des commentaires (en-tête et détail des points à vérifier)."""
    colonnes = "{:<8} {:>12} {:>12} {:>12} {:>10} {:>10} {:>10} {:>9} {:>8}  {}"
    lignes = [
        "# Shaped holes : point central de chaque perçage sur la surface de sortie, et axe du perçage",
        f"# Fichier CAO : {Path(chemin_cao).name} ({fmt})",
        f"# Date        : {datetime.now():%Y-%m-%d %H:%M:%S}",
        "# Unités      : mm ; repère = repère du fichier CAO (repère pièce)",
        "# X, Y, Z     : point central du perçage sur la surface de sortie",
        "# Vx, Vy, Vz  : vecteur unitaire de l'axe du perçage, orienté de la sortie vers la partie évasée",
        "# Rayon       : rayon du perçage (mm) ; Angle/Z : angle entre l'axe et Z (degrés)",
        "#",
        colonnes.format("# Nom", "X", "Y", "Z", "Vx", "Vy", "Vz", "Rayon", "Angle/Z", "Statut"),
    ]
    for t in trous:
        statut = "À VÉRIFIER" if t.a_verifier else "OK"
        rayon = f"{t.percage.rayon:.4f}" if t.percage is not None else "-"
        if t.centre_sortie is not None:
            x, y, z = (f"{v:.4f}" for v in t.centre_sortie)
            vx, vy, vz = (f"{v:.6f}" for v in t.vecteur)
            angle = f"{t.angle_z:.3f}"
        else:
            x = y = z = vx = vy = vz = angle = "-"
        lignes.append(colonnes.format(t.nom, x, y, z, vx, vy, vz, rayon, angle, statut))
    if not trous:
        lignes.append("# Aucun shaped hole détecté dans ce fichier.")

    a_verifier = [t for t in trous if t.a_verifier]
    if a_verifier:
        lignes += ["#", "# Détail des shaped holes à vérifier :"]
        for t in a_verifier:
            for defaut in t.a_verifier:
                lignes.append(f"#   {t.nom} : {defaut}")

    try:
        Path(chemin_txt).write_text("\n".join(lignes) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ErreurEcriture(f"{chemin_txt} : {exc.strerror} (le fichier est-il ouvert dans un autre "
                             f"logiciel ?)") from exc


# =====================================================================================================
# 8. AFFICHAGE 3D
# =====================================================================================================
# Souris : clic gauche = tourner | molette = zoom | Ctrl + clic gauche = déplacer
# Clavier : W = fil de fer (pour voir l'intérieur des trous) | Z = recentrer | Q = fermer

def fleche(depart, direction, longueur, rayon_percage):
    """Flèche pleine (tige + pointe) qui part de 'depart' dans la 'direction' (vecteur unitaire).
    Ses épaisseurs suivent le rayon du perçage, pour rester lisibles quelle que soit l'échelle."""
    r_tige, r_pointe, h_pointe = 0.12 * rayon_percage, 0.35 * rayon_percage, 1.5 * rayon_percage
    bout_tige = depart + (longueur - h_pointe) * direction
    tige = trimesh.creation.cylinder(radius=r_tige, segment=[depart, bout_tige], sections=16)
    pointe = trimesh.creation.cone(radius=r_pointe, height=h_pointe, sections=16)   # construit le long de +Z
    pointe.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], direction))
    pointe.apply_translation(bout_tige)
    f = trimesh.util.concatenate([tige, pointe])
    f.visual.face_colors = MAGENTA_SORTIE
    return f


def afficher_3d(trous, ignores, titre):
    scene = trimesh.Scene()
    for i, t in enumerate(trous):
        couleur = PALETTE[i % len(PALETTE)][1]
        c = t.corps.copy()                                  # copie : les données de calcul ne sont pas touchées
        couleurs = np.tile(couleur, (len(c.faces), 1))
        if t.face_entree is not None:
            couleurs[t.surface == t.face_entree] = (0.35 * np.array(couleur) + 0.65 * 255).astype(int)
        p = t.percage
        if p is not None:
            couleurs[p.masque] = JAUNE_PERCAGE
            # Axe du perçage : trait noir qui dépasse un peu de chaque côté du trou
            s = (c.vertices - p.point) @ p.axe
            marge = 0.15 * (s.max() - s.min())
            bouts = p.point + np.outer([s.min() - marge, s.max() + marge], p.axe)
            trait = trimesh.load_path(bouts[None, :, :])
            trait.colors = [[0, 0, 0, 255]]
            scene.add_geometry(trait, geom_name=t.nom + "_axe")
        if t.centre_sortie is not None:
            boule = trimesh.creation.icosphere(subdivisions=2, radius=0.3 * p.rayon)
            boule.apply_translation(t.centre_sortie)
            boule.visual.face_colors = MAGENTA_SORTIE
            scene.add_geometry(boule, geom_name=t.nom + "_sortie")
            # La flèche part du point de sortie et traverse le trou : sa pointe sort par la face d'entrée
            longueur = ((c.vertices - t.centre_sortie) @ t.vecteur).max() + 2 * p.rayon
            scene.add_geometry(fleche(t.centre_sortie, t.vecteur, longueur, p.rayon), geom_name=t.nom + "_vecteur")
        c.visual.face_colors = couleurs
        scene.add_geometry(c, geom_name=t.nom)
    for k, c in enumerate(ignores):
        g = c.copy()
        g.visual.face_colors = GRIS_IGNORES
        scene.add_geometry(g, geom_name=f"ecarte_{k}")
    if not scene.geometry:
        return
    # Vue inclinée centrée sur les trous (calculée avant d'ajouter le repère, souvent loin des trous)
    centre, taille = scene.centroid, scene.scale
    scene.set_camera(angles=(np.radians(55), 0, np.radians(30)), distance=1.6 * taille, center=centre)
    # Repère pièce : origine (0, 0, 0) en rouge, axes X rouge, Y vert, Z bleu (touche Z pour le voir)
    scene.add_geometry(trimesh.creation.axis(origin_size=0.04 * taille, origin_color=[255, 0, 0, 255],
                                             axis_length=0.3 * taille), geom_name="repere")
    scene.show(smooth=False, caption=titre)


# =====================================================================================================
# 9. PROGRAMME PRINCIPAL
# =====================================================================================================

def lire_arguments(argv):
    ap = argparse.ArgumentParser(description="Détection des shaped holes, de leurs perçages, du point "
                                             "central de sortie et de l'axe de chaque perçage.")
    ap.add_argument("fichier", nargs="?", help="fichier CAO .stl, .step ou .stp (sinon, une fenêtre le demande)")
    ap.add_argument("--sortie", help="chemin du fichier texte de résultats "
                                     "(par défaut : <fichier CAO>_percages.txt, à côté du fichier CAO)")
    ap.add_argument("--sans-3d", action="store_true", help="ne pas ouvrir la fenêtre 3D")
    return ap.parse_args(argv)


def demander_fichier():
    """Ouvre une fenêtre de choix de fichier. Renvoie le chemin choisi, ou None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        racine = tk.Tk()
        racine.withdraw()
        chemin = filedialog.askopenfilename(title="Fichier CAO des shaped holes",
                                            filetypes=[("Fichiers CAO", "*.stl *.step *.stp"), ("Tous", "*.*")])
        racine.destroy()
        return chemin or None
    except Exception:                           # pas d'interface graphique disponible
        return None


def afficher_compte_rendu(trous):
    print(f"Shaped holes détectés : {len(trous)}")
    for i, t in enumerate(trous):
        couleur = PALETTE[i % len(PALETTE)][0]
        print(f"   {t.nom} ({couleur}) : {'À VÉRIFIER' if t.a_verifier else 'OK'}")
        if t.percage is None:
            print("      pas de perçage")
        else:
            print(f"      perçage : rayon {t.percage.rayon:.4f} mm")
        if t.centre_sortie is not None:
            x, y, z = t.centre_sortie
            vx, vy, vz = t.vecteur
            print(f"      point central de sortie : ({x:.4f}, {y:.4f}, {z:.4f}) mm")
            print(f"      axe du perçage          : ({vx:.6f}, {vy:.6f}, {vz:.6f}), {t.angle_z:.2f}° par rapport à Z"
                  f" (contrôle : écart {t.ecart_controle * 1000:.2f} µm)")
        for defaut in t.a_verifier:
            print(f"      à vérifier : {defaut}")


def main(argv=None):
    args = lire_arguments(argv)
    chemin = args.fichier or demander_fichier()
    if not chemin:
        print("ERREUR : aucun fichier CAO indiqué. Exemple : python detection_shaped_holes.py piece.step")
        return 2
    try:
        mesh, fmt, infos = lire_cao(chemin)
        print(f"Fichier lu : {Path(chemin).name} ({fmt}), {len(mesh.faces)} triangles")
        for info in infos:
            print("INFO :", info)
        print("Dimensions (mm) : X = %.4f  Y = %.4f  Z = %.4f" % tuple(mesh.extents))

        print("Recherche des shaped holes (quelques minutes sur un maillage très fin) :")
        trous, ignores, messages = identifier_shaped_holes(mesh)
        for t in trous:
            calculer_point_de_sortie(t)
        for m in messages:
            print("INFO :", m)
        afficher_compte_rendu(trous)

        chemin_txt = Path(args.sortie) if args.sortie else Path(chemin).with_name(Path(chemin).stem + "_percages.txt")
        ecrire_fichier_texte(chemin_txt, chemin, fmt, trous)
        print(f"Fichier texte écrit : {chemin_txt}")

    except ErreurLecture as exc:
        print("ERREUR :", exc)
        return 2
    except ErreurEcriture as exc:
        print("ERREUR : fichier texte non écrit :", exc)
        return 3
    except Exception as exc:                    # filet de sécurité : message clair au lieu d'un plantage brutal
        print(f"ERREUR INATTENDUE ({type(exc).__name__}) :", exc)
        return 3

    if not args.sans_3d:
        print("Fenêtre 3D : chaque shaped hole a sa couleur, perçages en JAUNE, axe en noir,")
        print("             point central de sortie et sens de l'axe en MAGENTA, corps écartés en gris.")
        print("             W = fil de fer, Z = recentrer, Q = fermer.")
        afficher_3d(trous, ignores, f"Shaped holes - {Path(chemin).name}")

    return 1 if any(t.a_verifier for t in trous) else 0


if __name__ == "__main__":
    sys.exit(main())
