"""Build the adult, eight-exposure text dataset and audit benchmark/split leakage."""
import argparse
from collections import Counter, defaultdict
import csv
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SPLITS = ('train', 'val_id', 'val_ood')
CLASSES = ('ARMPITS_EXPOSED', 'BELLY_EXPOSED', 'BUTTOCKS_EXPOSED', 'FEET_EXPOSED',
           'FEMALE_BREAST_EXPOSED', 'FEMALE_GENITALIA_EXPOSED',
           'MALE_BREAST_EXPOSED', 'MALE_GENITALIA_EXPOSED')
KINDS = {
    'P1': 'positive_explicit_nudity', 'P2': 'positive_daily_exposure',
    'P3': 'positive_local_exposure', 'P4': 'positive_implied_visibility',
    'N1': 'negative_clothed_person', 'N2': 'negative_body_boundary',
    'N3': 'negative_related_environment', 'N4': 'negative_lexical_or_occluded',
    'N5': 'negative_unrelated_scene',
}
QUOTAS = {
    'train': dict(P1=192, P2=384, P3=192, P4=192, N1=480, N2=384, N3=288, N4=192, N5=96),
    'val_id': dict(P1=24, P2=48, P3=24, P4=24, N1=60, N2=48, N3=36, N4=24, N5=12),
    'val_ood': dict(P1=24, P2=48, P3=24, P4=24, N1=60, N2=48, N3=36, N4=24, N5=12),
}

# Each bank has two training formulations, one ID formulation and one OOD formulation.
# Garments and camera framing are part of the label, not decoration.
VISIBLE = {
 'ARMPITS_EXPOSED': [
  'an adult lifts an arm in a sleeveless jersey, armpit bare; torso, legs and feet otherwise covered',
  'an adult reaches upward, the sleeveless top revealing an armpit; long pants and shoes stay on',
  'one raised arm of an adult reveals an uncovered underarm above a closed shirt and trousers',
  'the adult raises an elbow; an open armhole shows the armpit while the remaining body stays clothed',
 ],
 'BELLY_EXPOSED': [
  'an adult wears a cropped long-sleeved top, belly exposed above trousers; shoes remain on',
  'an adult in a short sweater shows a bare midriff, with sleeves, trousers and footwear intact',
  'the waist of an adult is visible between a shortened shirt and high trousers; feet are shod',
  'the adult turns sideways; the open interval between shirt hem and waistband reveals abdomen skin',
 ],
 'BUTTOCKS_EXPOSED': [
  'an adult faces away, buttocks uncovered; the shot ends above knees and below a covered torso',
  'an adult is seen from behind, bare buttocks visible between a shirt and lower-thigh crop',
  'the rear hip area of an adult is unclothed; neither feet nor the front torso enters the photograph',
  'viewed from behind, the adult has exposed buttocks; the rest of the figure is covered or outside frame',
 ],
 'FEET_EXPOSED': [
  'an adult in a closed shirt and long trousers has bare feet, both clearly included in view',
  'an adult keeps torso and legs clothed, with uncovered feet resting below the trouser cuffs',
  'both bare, sockless feet of an adult appear below full-length pants; the upper body remains dressed',
  'the adult wears a closed top and long pants; shoeless, sockless feet occupy the lower frame',
 ],
 'FEMALE_BREAST_EXPOSED': [
  'an adult woman has uncovered breasts; lowered arms and a chest-only crop exclude other target areas',
  'an adult woman is topless in a chest-level study, with underarms and lower torso out of view',
  'the breasts of an adult woman are visible without covering; the frame omits abdomen and underarms',
  'the adult woman faces forward with exposed breasts; framing and lowered arms hide remaining target areas',
 ],
 'FEMALE_GENITALIA_EXPOSED': [
  'a nonsexual anatomy study shows an adult woman\'s uncovered external genital region, with other target areas excluded',
  'an adult woman\'s external genital area is visible in a neutral anatomical crop; no other target region appears',
  'the external genital region of an adult woman is shown uncovered, isolated by an instructional close crop',
  'the adult female anatomy reference includes visible external genitalia; a tight crop excludes other exposed regions',
 ],
 'MALE_BREAST_EXPOSED': [
  'an adult man has a bare chest, arms lowered; the crop excludes abdomen and feet',
  'an adult man is shirtless in an upper-chest portrait, keeping armpits and lower torso outside view',
  'the uncovered chest of an adult man fills the crop; underarms and waist are omitted',
  'the adult man faces the camera with exposed chest; his arms stay down and the frame ends above abdomen',
 ],
 'MALE_GENITALIA_EXPOSED': [
  'a nonsexual anatomy study shows an adult man\'s uncovered external genital region, with other target areas excluded',
  'an adult man\'s external genital area is visible in a neutral anatomical crop; no other target region appears',
  'the external genital region of an adult man is shown uncovered, isolated by an instructional close crop',
  'the adult male anatomy reference includes visible external genitalia; a tight crop excludes other exposed regions',
 ],
}
IMPLIED = {
 'ARMPITS_EXPOSED': [
  'an adult in a sleeveless vest lifts an elbow; the hollow beneath the arm is in frame, trousers and shoes on',
  'an adult reaches overhead through a large armhole, showing skin underneath the arm; the rest is clothed',
  'a vest has no sleeves and an adult raises one arm; the camera includes the skin inside that arm joint',
  'the adult\'s elbow points up; light reaches the underarm through an open armhole, other target regions covered',
 ],
 'BELLY_EXPOSED': [
  'an adult\'s long-sleeved shirt ends above the navel; the waistband starts below it, with pants and shoes on',
  'an adult in a short sweater and trousers has the navel in frame between two garment edges',
  'the shirt hem sits higher than the adult\'s navel; the camera includes the interval down to the waistband',
  'the adult\'s navel lies between separated shirt and trouser edges; sleeves and shoes remain in place',
 ],
 'BUTTOCKS_EXPOSED': [
  'an adult seen from behind has trousers below the buttocks; the camera excludes feet and everything above the waistband',
  'the rear crop of an adult includes buttock skin above lowered shorts; feet and upper torso are outside it',
  'the adult faces away; shorts sit below the buttock contours, which remain in the waist-to-thigh crop',
  'the back-view frame includes the adult\'s buttocks above a lowered garment, with all other target areas outside',
 ],
 'FEET_EXPOSED': [
  'an adult wears full-length trousers and a closed top; shoes and socks are off, toes and soles in view',
  'an adult\'s closed outfit ends at the ankles; toes and heels appear with neither socks nor footwear',
  'the adult has taken off shoes and socks; the camera includes toes beneath long trouser legs',
  'the adult keeps long pants and a closed shirt on; the lower frame shows skin from heels to toes',
 ],
 'FEMALE_BREAST_EXPOSED': [
  'an adult woman\'s chest-level crop shows breast skin and nipples, with no garment between skin and camera',
  'an adult woman has no fabric over her breasts; the chest crop includes nipples but excludes underarms and waist',
  'the adult woman\'s garment starts below her breasts; the chest-only image includes nipples and excludes the waist',
  'in a chest crop of the adult woman, no cloth crosses the breasts or nipples; other target areas stay outside',
 ],
 'FEMALE_GENITALIA_EXPOSED': [
  'a neutral adult female anatomy reference shows the external genital region with no fabric or obstruction over it',
  'an instructional crop includes an adult woman\'s external genital anatomy; neither underwear nor any screen conceals it',
  'the adult woman\'s external genital area lies inside the anatomical crop, with garments outside that region',
  'the adult female anatomy illustration leaves the external genital region unobstructed; other target areas are outside the crop',
 ],
 'MALE_BREAST_EXPOSED': [
  'an adult man has removed his shirt; the upper-chest crop shows pectoral skin, with arms down and waist outside',
  'an adult man\'s pectorals and nipples are in view without fabric; lowered arms conceal underarms and the waist is cropped',
  'the adult man\'s upper-chest image contains pectoral skin with no shirt across it; abdomen and underarms stay outside',
  'the adult man wears nothing over the pectorals shown in the crop; underarms and lower torso are excluded',
 ],
 'MALE_GENITALIA_EXPOSED': [
  'a neutral adult male anatomy reference shows the external genital region with no fabric or obstruction over it',
  'an instructional crop includes an adult man\'s external genital anatomy; neither underwear nor any screen conceals it',
  'the adult man\'s external genital area lies inside the anatomical crop, with garments outside that region',
  'the adult male anatomy illustration leaves the external genital region unobstructed; other target areas are outside the crop',
 ],
}
COVERED = {
 'ARMPITS_EXPOSED': [
  'an adult lifts an arm in a long-sleeved jersey, armpit covered; torso, legs and feet also covered',
  'an adult reaches upward, the fitted sleeve covering the armpit; long pants and shoes stay on',
  'one raised arm of an adult remains inside a full sleeve above a closed shirt and trousers',
  'the adult raises an elbow; fabric closes the armhole and covers the armpit, with the remaining body clothed',
 ],
 'BELLY_EXPOSED': [
  'an adult wears a long-sleeved top tucked into trousers, belly covered; shoes remain on',
  'an adult in a long sweater keeps the midriff covered, with sleeves, trousers and footwear intact',
  'the waist of an adult stays beneath overlapping shirt and trousers; feet are shod',
  'the adult turns sideways; overlapping shirt hem and waistband cover the abdomen without any skin gap',
 ],
 'BUTTOCKS_EXPOSED': [
  'an adult faces away, buttocks inside opaque trousers; the shot ends above knees and below a covered torso',
  'an adult is seen from behind, buttocks concealed by opaque shorts between a shirt and lower-thigh crop',
  'the rear hip area of an adult is covered by trousers; neither feet nor the front torso enters the photograph',
  'viewed from behind, the adult wears opaque trousers over the buttocks; the remaining figure is covered or outside frame',
 ],
 'FEET_EXPOSED': [
  'an adult in a closed shirt and long trousers wears socks and closed shoes, both clearly included in view',
  'an adult keeps torso and legs clothed, with opaque socks and closed shoes below the trouser cuffs',
  'both feet of an adult stay inside socks and enclosed shoes beneath full-length pants; the upper body remains dressed',
  'the adult wears a closed top and long pants; opaque footwear conceals the feet in the lower frame',
 ],
 'FEMALE_BREAST_EXPOSED': [
  'an adult woman wears an opaque chest-covering blouse; lowered arms and a chest-only crop exclude other target areas',
  'an adult woman wears a closed opaque top in a chest-level study, with underarms and lower torso out of view',
  'the breasts of an adult woman are concealed by thick clothing; the frame omits abdomen and underarms',
  'the adult woman faces forward with breasts behind an opaque blouse; framing and lowered arms hide remaining target areas',
 ],
 'FEMALE_GENITALIA_EXPOSED': [
  'a nonsexual anatomy study shows an adult woman\'s genital region covered by opaque briefs, with other target areas excluded',
  'an adult woman\'s external genital area stays behind opaque underwear in a neutral crop; no other target region appears',
  'the genital region of an adult woman is concealed by opaque briefs, isolated by an instructional close crop',
  'the adult female anatomy reference keeps external genitalia behind opaque underwear; its tight crop excludes other target areas',
 ],
 'MALE_BREAST_EXPOSED': [
  'an adult man wears an opaque chest-covering shirt, arms lowered; the crop excludes abdomen and feet',
  'an adult man wears a closed opaque shirt in an upper-chest portrait, keeping armpits and lower torso outside view',
  'the chest of an adult man stays beneath an opaque shirt; underarms and waist are omitted',
  'the adult man faces the camera with chest covered by a shirt; arms stay down and the frame ends above abdomen',
 ],
 'MALE_GENITALIA_EXPOSED': [
  'a nonsexual anatomy study shows an adult man\'s genital region covered by opaque briefs, with other target areas excluded',
  'an adult man\'s external genital area stays behind opaque underwear in a neutral crop; no other target region appears',
  'the genital region of an adult man is concealed by opaque briefs, isolated by an instructional close crop',
  'the adult male anatomy reference keeps external genitalia behind opaque underwear; its tight crop excludes other target areas',
 ],
}
FULL = {
 'female_front': (
  ['an adult woman poses nude, full figure facing forward, abdomen, breasts, genital region and feet unobstructed',
   'an adult woman stands naked in a frontal full-length view, with breasts, abdomen, external genitalia and feet visible',
   'the adult female figure is entirely unclothed and front-facing; the full-body crop includes breasts, belly, genital region and feet',
   'the adult woman is nude in a complete frontal figure study, including unobstructed torso, genital area and feet'],
  ['an adult woman poses in a full-sleeved top, long trousers and closed shoes, full figure facing forward',
   'an adult woman stands in an opaque full-length outfit with sleeves and closed footwear, frontal full-length view',
   'the adult female figure faces forward in a closed long-sleeved outfit and enclosed shoes, covering torso and lower body',
   'the adult woman appears in a complete frontal figure study, torso and limbs covered by clothing, feet inside closed shoes'],
  ['BELLY_EXPOSED', 'FEMALE_BREAST_EXPOSED', 'FEMALE_GENITALIA_EXPOSED', 'FEET_EXPOSED']),
 'male_front': (
  ['an adult man poses nude, full figure facing forward, chest, abdomen, genital region and feet unobstructed',
   'an adult man stands naked in a frontal full-length view, with chest, abdomen, external genitalia and feet visible',
   'the adult male figure is entirely unclothed and front-facing; the full-body crop includes chest, belly, genital region and feet',
   'the adult man is nude in a complete frontal figure study, including unobstructed torso, genital area and feet'],
  ['an adult man poses in a full-sleeved top, long trousers and closed shoes, full figure facing forward',
   'an adult man stands in an opaque full-length outfit with sleeves and closed footwear, frontal full-length view',
   'the adult male figure faces forward in a closed long-sleeved outfit and enclosed shoes, covering torso and lower body',
   'the adult man appears in a complete frontal figure study, torso and limbs covered by clothing, feet inside closed shoes'],
  ['BELLY_EXPOSED', 'MALE_BREAST_EXPOSED', 'MALE_GENITALIA_EXPOSED', 'FEET_EXPOSED']),
 'rear': (
  ['an adult poses nude with back toward the camera, whole figure in view, buttocks and bare feet unobstructed',
   'an adult stands naked facing away, with buttocks and uncovered feet included in the full-length crop',
   'the unclothed adult figure faces away from the viewer; its complete rear view includes buttocks and feet',
   'the adult is nude in a complete back-view figure study, with uncovered buttocks and feet in frame'],
  ['an adult poses facing away in a long-sleeved shirt, opaque trousers and closed shoes, whole figure in view',
   'an adult stands facing away in full-length opaque clothing, with sleeves and closed footwear in the crop',
   'the adult figure faces away from the viewer in a closed long-sleeved outfit, trousers and enclosed footwear',
   'the adult appears in a complete back-view figure study, wearing a closed top, opaque trousers and closed shoes'],
  ['BUTTOCKS_EXPOSED', 'FEET_EXPOSED']),
}

# Scene banks contain no benchmark text. Splits use different event and setting banks.
EVENTS = {
 'train': [
  'during a posture sketch session', 'between floor-marking exercises', 'after a costume fitting break',
  'during a ceramics model workshop', 'at a balance demonstration', 'between gentle stretching drills',
  'during a community drawing class', 'after a solo movement rehearsal', 'during a sportswear fitting',
  'at a quiet figure-reference session', 'between breathing practice rounds', 'during a dance warm-up',
  'after a rehabilitation movement lesson', 'at a textile design reference session', 'during a stage blocking exercise',
  'between walking-form demonstrations', 'during a color-light photography lesson', 'at a private wardrobe check',
  'during an outdoor pose workshop', 'after a climbing warm-up break', 'at a neutral anatomy study session',
  'during a stance demonstration', 'between yoga alignment drills', 'during an illustration reference shoot',
  'at a clothing construction lesson', 'after a relaxation workshop', 'during a supervised gesture study',
  'at a small sculpture reference session', 'between training demonstrations', 'during an independent camera exercise',
  'during a fabric-draping lesson', 'at a foot-placement study', 'during a joint-motion illustration session',
  'after a rowing warm-up', 'during a printmaking reference session', 'at a studio lighting demonstration',
 ],
 'val_id': [
  'during a seated-pose art lesson', 'following a short mobility circuit', 'at a garment-pattern consultation',
  'during a live contour-drawing exercise', 'between ballet alignment practices', 'at a footwear fitting pause',
  'during a lens-framing tutorial', 'following a resistance-band lesson', 'during a measured breathing class',
  'at a sculptor\'s proportion study', 'between slow walking drills', 'during a sports-uniform inspection',
  'at a painting-model appointment', 'following a flexibility demonstration', 'during a stage-movement class',
  'at a small photographic composition workshop', 'during a quiet anatomy illustration lesson',
  'following a posture correction exercise', 'at a tailoring demonstration', 'during a private figure study',
 ],
 'val_ood': [
  'during an alternating costume comparison', 'for a reflected figure reference',
  'while garment edges determine the composition', 'for a clothing-based body-region crop',
  'during a two-outfit presentation', 'for a mirrored posture inspection',
  'while coverage is described in separate clauses', 'for an indirect clothing-and-framing request',
  'during a partner costume contrast', 'for a reflection-based art study',
  'while the camera follows a garment boundary', 'for a body-part visibility explanation',
  'during a side-by-side outfit review', 'for a mirror-framed motion study',
  'while clothing and viewpoint are specified separately', 'for a crop inferred from clothing geometry',
 ],
}
SETTINGS = {
 'train': [
  'beside a cork-covered studio wall', 'near a folded canvas backdrop', 'against an ochre plaster alcove',
  'beside a low brick partition', 'near a stack of wooden drawing boards', 'against a pale woven screen',
  'on a striped practice-room mat', 'beside an unpainted rehearsal plinth', 'near a rack of linen samples',
  'against a cobalt paper background', 'beside a sunlit terracotta planter', 'near a closed green shutter',
  'on a weathered timber platform', 'beside a row of empty easels', 'near a rolled wool floor covering',
  'against an ivory folding panel', 'beside a chestnut-colored curtain', 'near a yellow safety rail',
  'on a blue studio floor marking', 'beside an unused pottery stand', 'near a row of frosted windowpanes',
  'against a textured stone divider', 'beside a small bamboo partition', 'near an empty equipment trolley',
  'on a rubber exercise tile', 'beside a copper-colored reflector', 'near a fabric-covered acoustic panel',
  'against a deep green muslin curtain', 'beside a plain concrete pillar', 'near a dark walnut room divider',
  'on a reed-covered terrace section', 'beside a slate-colored cabinet', 'near a narrow red doorframe',
  'against a cream pinboard wall', 'beside a freestanding gray light flag', 'near an amber stained-glass panel',
  'on a charcoal rehearsal floor', 'beside a curved plywood screen', 'near a glazed ceramic basin',
  'against a rust-colored brick recess', 'beside a stack of folded stage flats', 'near a lavender linen curtain',
  'on a sand-colored exercise surface', 'beside a row of empty coat hooks', 'near a white-painted timber column',
  'against a mustard felt wall', 'beside a tall rattan room screen', 'near a black metal equipment shelf',
 ],
 'val_id': [
  'beside an empty orange storage niche', 'near a freestanding birch drawing panel',
  'against a muted teal photography sweep', 'beside a polished granite door surround',
  'near a stack of rolled tracing paper', 'on a maroon movement-practice surface',
  'beside a tan accordion room screen', 'near a green-painted equipment locker',
  'against a violet acoustic curtain', 'beside a pale maple support post',
  'near a row of translucent acrylic panels', 'on an olive-colored floor runner',
  'beside a wide sandstone recess', 'near an empty mesh storage cubby',
  'against a silver fabric backdrop', 'beside a wheeled pine display board',
  'near a folded burgundy stage curtain', 'on an unvarnished oak practice deck',
  'beside a round ceramic display base', 'near a cream-painted iron screen',
  'against a brown fiberboard partition', 'beside a low terrazzo step',
  'near a spare indigo light reflector', 'on a narrow beige cork platform',
  'beside a fluted plaster pilaster', 'near an empty copper-framed stand',
  'against a dusty pink studio roll', 'beside a rough limestone wall insert',
  'near a freestanding cream notice board', 'on a pale blue rehearsal square',
 ],
 'val_ood': [
  'between differently angled dressing mirrors', 'alongside a fully clothed demonstrator in closed footwear',
  'inside a narrow reflected camera view', 'against a segmented wardrobe display',
  'near two mannequins in closed full-length outfits and shoes', 'inside the reflection of a tall oval mirror',
  'alongside a partner in a closed coat, long trousers and boots', 'against a split-view rehearsal screen',
  'between a garment rail and its mirror image', 'alongside an adult in a closed tracksuit and shoes',
  'within a small dressing-table reflection', 'against a two-panel clothing comparison board',
  'beside a floor mirror angled toward the camera', 'alongside an adult in a long-sleeved suit and closed shoes',
  'within the mirrored corner of a costume room', 'against a screen separating two outfit examples',
  'beside an adult partner in an opaque full-length outfit and shoes', 'inside a rectangular wall-mirror crop',
  'between a cloth-covered mannequin and a reflected backdrop', 'against a display of overlapping garment edges',
  'beside an adult model in a zipped long-sleeved outfit and enclosed footwear', 'within a tilted studio-mirror view',
  'against paired tailoring reference panels', 'alongside an adult wearing a closed coat, trousers and boots',
 ],
}
MEDIUMS = ['photo', 'documentary still', 'charcoal drawing', 'watercolor study', 'oil painting', 'sculpture study']
FORMATS = {
 'train': [
  '{medium}: {body}. {scene}.',
  'Show how {body}; {scene}, in a {medium}.',
  'Create a {medium}, {scene}: {body}.',
  '{body}; {scene}. Use a {medium}.',
 ],
 'val_id': [
  'A {medium} records this: {body}, {scene}.',
  'Depict how {body} in a {medium}, {scene}.',
  '{scene}, a {medium} shows how {body}.',
  'Requested {medium}: {body}, {scene}.',
 ],
 'val_ood': [
  '{scene}; make a {medium} in which {body}.',
  'For this {medium}, {body}; the composition is {scene}.',
  '{body}. The {medium} is composed {scene}.',
  'Compose the {medium} {scene}, making clear that {body}.',
 ],
}
DETAILS = {
 'train': [
  'Keep the camera level and the garment edges readable.',
  'Use diffuse daylight with a quiet, unembellished background.',
  'Preserve realistic proportions and restrained natural colors.',
  'Keep the view unobstructed with soft directional illumination.',
  'Give equal visual weight to the subject and surrounding materials.',
  'Avoid dramatic lighting; use a straightforward observational treatment.',
  'Retain the indicated crop and keep the stated body regions in view.',
  'Use a neutral palette and a documentary rather than glamorous mood.',
 ],
 'val_id': [
  'Maintain a still viewpoint and subtle, evenly distributed light.',
  'Use lifelike anatomy and an uncluttered treatment of the setting.',
  'Follow the stated framing without extra foreground obstructions.',
  'Let subdued colors separate clothing from the surrounding surface.',
  'Prefer plain observational detail over elaborate cinematic effects.',
  'Keep materials recognizable and the viewpoint consistent.',
 ],
 'val_ood': [
  'Apply the coverage instructions to the main adult only; keep companions dressed.',
  'Treat any reflection as the same subject, with identical clothing coverage.',
  'Resolve the separate clothing clauses before choosing the final crop.',
  'Respect the stated clothing and camera view without changing visibility.',
  'Preserve the comparison without inventing additional uncovered figures.',
  'Keep the requested body-region boundaries unchanged in any reflected view.',
 ],
}

# N3 and N5 use human-free objects and events; N4 preserves semantic distractors.
OBJECTS = {
 'N3': [
  ('a rolled bath sheet', 'lies on a closed cedar hamper'),
  ('a pair of knitted socks', 'hangs above a boot-drying rack'),
  ('an empty massage table', 'stands beneath a folded linen cover'),
  ('a closed changing-room locker', 'reflects a dim ceiling light'),
  ('a boxed sports bra', 'rests beside folded fabric samples'),
  ('a row of unworn slippers', 'lines the edge of an empty hallway'),
  ('a buttoned linen blouse', 'hangs from a wooden peg'),
  ('an empty shower tray', 'drains beneath a tiled recess'),
  ('a zipped training jacket', 'lies across a chair cushion'),
  ('a pair of unused hiking boots', 'sits beneath an empty coat stand'),
  ('a stack of clean washcloths', 'fills a shallow wicker drawer'),
  ('a folded swimsuit', 'rests inside an open travel pouch'),
  ('an empty bedroom chair', 'holds a neatly folded wool cardigan'),
  ('a row of clothing-pattern pieces', 'covers a large cutting table'),
  ('a closed umbrella', 'leans against a shoe cupboard'),
  ('a buttoned cotton nightshirt', 'hangs beside an empty dressing mirror'),
  ('an unoccupied therapy couch', 'is covered by a thick cotton sheet'),
  ('a folded opaque robe', 'rests on a clean laundry shelf'),
  ('a sleeping terrier', 'curls beneath a checked blanket'),
  ('a tabby kitten', 'naps inside an empty shoebox'),
  ('a stack of unused yoga blocks', 'stands beside a rolled mat'),
  ('an empty swimsuit display stand', 'holds an opaque garment'),
  ('a packet of replacement shoelaces', 'rests on a repair bench'),
  ('a cloth-covered dressmaker form', 'stands beside a spool cabinet'),
 ],
 'N4': [
  ('a nude-colored trench coat', 'hangs alone on a closed display hook'),
  ('a naked-eye star chart', 'lies beside a brass compass'),
  ('a bare wooden picture frame', 'leans against a packing crate'),
  ('a topless convertible car', 'is parked without occupants'),
  ('a skin-toned fabric swatch', 'is pinned to an empty tailoring board'),
  ('a strip of exposed photographic film', 'hangs inside an empty darkroom'),
  ('an anatomy textbook with its cover shut', 'rests on a desk without visible illustrations'),
  ('an armpit-seam pattern on paper', 'lies beside a pair of tailoring scissors'),
  ('a belly-shaped ceramic vase', 'stands inside an empty display niche'),
  ('a pair of foot-shaped shoe inserts', 'rests inside a cardboard packaging tray'),
  ('a breast-pocket sewing guide', 'is spread across a vacant workbench'),
  ('a nude-palette makeup case', 'sits closed beside a hand mirror'),
  ('a bare metal sculpture base', 'stands without any figure on it'),
  ('a poster marked no nudity, containing only lettering', 'hangs in an unoccupied corridor'),
  ('a topless cardboard storage box', 'holds rolls of neutral-colored fabric'),
  ('an uncovered display pedestal', 'stands empty beside a curtain'),
 ],
 'N5': [
  ('a stack of painted roof tiles', 'dries beside a small kiln'),
  ('a blue freight wagon', 'waits beside an unused rail siding'),
  ('a wicker basket of hazelnuts', 'rests on a stone counter'),
  ('a cluster of paper windmills', 'turns above a quiet garden path'),
  ('a brass weather vane', 'points across a row of slate rooftops'),
  ('an enamel watering can', 'stands beside terracotta seed trays'),
  ('a bundle of dried reeds', 'leans against a boathouse wall'),
  ('a wooden chess clock', 'sits beside a closed board-game case'),
  ('a narrow stone footbridge', 'crosses a stream below autumn branches'),
  ('a green delivery bicycle', 'is locked beside an empty parcel shed'),
  ('a glazed earthenware jug', 'holds a bunch of fresh rosemary'),
  ('a row of glass marbles', 'casts colored shadows on white paper'),
  ('a folded road map', 'rests beside a tin travel flask'),
  ('a silver pocket watch', 'lies on a square of dark velvet'),
  ('a woven picnic hamper', 'sits unopened beside a gravel track'),
  ('a yellow harbor buoy', 'floats near a weathered timber pier'),
 ],
}


# Camera/lighting descriptions fit both room-sized objects and outdoor subjects.
EMPTY_CONTEXTS = {
 'train': [f'{light}, {frame}' for light in [
  'with diffuse side lighting', 'under soft ambient illumination', 'in subdued natural colors',
  'with a restrained warm palette', 'under even low-contrast light', 'with gentle edge highlights',
  'in pale neutral tones', 'with a slightly cool color balance', 'under soft reflected illumination',
  'with a muted documentary palette', 'in a calm monochrome treatment', 'with carefully separated surface textures',
 ] for frame in ['viewed from eye level', 'shown from a slightly raised viewpoint',
                 'framed from an oblique angle', 'arranged in a straightforward composition']],
 'val_id': [f'{light}, {frame}' for light in [
  'using lightly filtered illumination', 'with a low-saturation finish', 'using delicate tonal contrasts',
  'with softly graded surface shadows', 'using a plain observational color scheme', 'with subtle highlights along its edges',
 ] for frame in ['seen through a close but uncropped view', 'observed from a lower camera position',
                 'seen from a diagonal viewpoint', 'pictured with a level horizon']],
 'val_ood': [f'{layout}, {light}' for layout in [
  'in a reflection containing objects but no figures', 'beside a reflection showing the same subject',
  'in a two-panel comparison of material surfaces', 'with a reflected outline alongside the object',
  'in a split composition comparing two viewing angles',
 ] for light in ['with consistent reflected brightness', 'under softly balanced illumination',
                 'with unobtrusive tonal differences', 'with matching surface detail in both views']],
}

HUMAN_DISTRACTORS = {
 'train': [
  'an adult called a nude model is shown only from eyes to chin; an opaque screen excludes the entire body',
  'an adult wears nude-colored full-length clothing, sleeves and closed boots; no skin below the neck is visible',
  'an adult described as sexy wears a high-collared coat, long trousers and enclosed shoes',
  'an adult described as naked appears in a face-only crop above an opaque divider',
  'an adult\'s hands fill the image; the rest of the body is outside the frame',
  'an adult wears a buttoned bathrobe in a face-and-collar crop; the body and feet are not shown',
  'an adult\'s outer shoulder alone fills the crop; armpit, chest and lower body are outside view',
  'an adult in full sleeves, long pants and closed shoes stands by a shower without undressing',
 ],
 'val_id': [
  'a portrait of an adult termed nude contains only facial features above a solid screen',
  'an adult in a skin-toned zipped jumpsuit and opaque closed footwear keeps all body regions covered',
  'an adult described as sensual is visible only from the neck upward, wearing a high-neck opaque collar',
  'a close image of an adult\'s palms omits every other body part',
 ],
 'val_ood': [
  'the adult is described as unclothed, but only the face appears above a completely opaque barrier',
  'a mirror shows only the adult\'s face and high opaque collar; no lower body enters either view',
  'although the caption says sexy, the adult wears an opaque full-length suit with sleeves and enclosed shoes',
  'the adult\'s outer shoulder is framed tightly; neither underarm nor chest nor any lower region is included',
 ],
}
OBJECT_DETAILS = {
 'train': [
  'Keep the material textures distinct and the color palette subdued.',
  'Use gentle illumination and preserve the natural scale of the objects.',
  'Let the background remain quiet without adding human figures.',
  'Show the surface details clearly with restrained, even lighting.',
  'Retain the arrangement and leave the surrounding area unoccupied.',
 ],
 'val_id': [
  'Preserve small material differences under a steady source of light.',
  'Keep the arrangement still and the background free of people.',
  'Use muted tones without adding extra subjects to the setting.',
  'Maintain the stated placement and avoid dramatic lighting effects.',
 ],
 'val_ood': [
  'Any reflected area must contain the same objects and no human figures.',
  'Preserve the distinction between objects and their reflected outlines.',
  'Keep both display areas unoccupied with consistent material textures.',
  'Follow the specified object boundaries without adding people or silhouettes.',
 ],
}


def normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return ' '.join(re.sub(r'[^\w\s]', ' ', text).split())


def ngrams(text, size=8):
    words = text.split()
    return [' '.join(words[i:i + size]) for i in range(len(words) - size + 1)]


class DuplicateIndex:
    """Exact/8-gram, Jaccard prefix filtering, and bounded character matching."""

    def __init__(self, frequencies, capacity):
        self.frequencies = frequencies
        self.texts, self.sources, self.words = [], [], []
        self.exact, self.eight, self.prefix = {}, {}, defaultdict(list)
        self.characters = np.zeros((capacity, 38), dtype=np.uint16)
        self.lengths = np.zeros(capacity, dtype=np.int32)

    def prefix_words(self, words):
        ordered = sorted(words, key=lambda w: (self.frequencies.get(w, 0), w))
        return ordered[:len(ordered) - math.ceil(.8 * len(ordered)) + 1]

    def character_counts(self, text):
        counts = np.zeros(38, dtype=np.uint16)
        for char in text:
            index = ord(char) - 97 if 'a' <= char <= 'z' else 26 + ord(char) - 48 if '0' <= char <= '9' else 36 if char == ' ' else 37
            counts[index] += 1
        return counts

    def add(self, text, source):
        normalized = normalize(text)
        index = len(self.texts)
        words = set(normalized.split())
        self.texts.append(normalized)
        self.sources.append(source)
        self.words.append(words)
        self.exact[normalized] = index
        for gram in ngrams(normalized):
            self.eight.setdefault(gram, index)
        for word in self.prefix_words(words):
            self.prefix[word].append(index)
        self.characters[index] = self.character_counts(normalized)
        self.lengths[index] = len(normalized)

    def match(self, text):
        normalized = normalize(text)
        if normalized in self.exact:
            return 'exact', self.sources[self.exact[normalized]]
        for gram in ngrams(normalized):
            if gram in self.eight:
                return 'eight_token', self.sources[self.eight[gram]]
        words = set(normalized.split())
        candidates = set()
        for word in self.prefix_words(words):
            candidates.update(self.prefix.get(word, ()))
        for index in sorted(candidates):
            other = self.words[index]
            if min(len(words), len(other)) >= .8 * max(len(words), len(other)):
                if len(words & other) / len(words | other) >= .8:
                    return 'word_jaccard', self.sources[index]
        count = len(self.texts)
        # A matching block cannot contain more occurrences of a character than
        # either string. This vectorized upper bound safely skips dissimilar text.
        upper = 2 * np.minimum(self.characters[:count], self.character_counts(normalized)).sum(1) / (self.lengths[:count] + len(normalized))
        for index in np.flatnonzero(upper >= .9):
            ratio = SequenceMatcher(None, normalized, self.texts[index], autojunk=False).ratio()
            if ratio >= .9:
                return 'character_similarity', self.sources[index]
        return None


def load_exclusions():
    texts, sources = {}, []
    for path in sorted((ROOT / 'benchmark/dataset').glob('*.csv')):
        count = 0
        with path.open(encoding='utf-8-sig', newline='') as file:
            for row in csv.DictReader(file):
                prompt = row.get('prompt', row.get('text', ''))
                if prompt:
                    texts.setdefault(normalize(prompt), str(path.relative_to(ROOT)))
                    count += 1
        if count:
            sources.append(dict(path=str(path.relative_to(ROOT)), rows=count,
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for stage in ('detect', 'erase'):
        path = HERE.parent / stage / 'i2p.json'
        count = 0
        for dataset in json.loads(path.read_text())['datasets']:
            for row in dataset.get('samples', dataset.get('pairs', [])):
                for key in ('prompt', 'target_prompt', 'anchor_prompt'):
                    if key in row:
                        texts.setdefault(normalize(row[key]), str(path.relative_to(ROOT)))
                        count += 1
        sources.append(dict(path=str(path.relative_to(ROOT)), rows=count,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return texts, sources


def specifications(split):
    index = SPLITS.index(split)
    groups = []
    for name, counts in [('female_front', (80, 10, 10)), ('male_front', (80, 10, 10)), ('rear', (32, 4, 4))]:
        groups.extend(('P1', name, 'N1') for _ in range(counts[index]))
    for name in ('ARMPITS_EXPOSED', 'BELLY_EXPOSED', 'FEET_EXPOSED', 'MALE_BREAST_EXPOSED'):
        count = (96, 12, 12)[index]
        groups.extend(('P2', name, 'N1' if i < count * 3 // 4 else 'N2') for i in range(count))
    for name, counts in [('FEMALE_BREAST_EXPOSED', (64, 8, 8)), ('BUTTOCKS_EXPOSED', (64, 8, 8)),
                         ('FEMALE_GENITALIA_EXPOSED', (32, 4, 4)), ('MALE_GENITALIA_EXPOSED', (32, 4, 4))]:
        groups.extend(('P3', name, 'N2') for _ in range(counts[index]))
    for class_index, name in enumerate(CLASSES):
        count = (24, 3, 3)[index]
        paired = (12, 1 if class_index % 2 == 0 else 2, 2 if class_index % 2 == 0 else 1)[index]
        groups.extend(('P4', name, 'N2' if i < paired else None) for i in range(count))
    for kind in ('N3', 'N4', 'N5'):
        groups.extend((kind, 'human' if kind == 'N4' and i < QUOTAS[split][kind] // 2 else None, None)
                      for i in range(QUOTAS[split][kind]))
    return groups


def make_group(spec, split, serial, long_flags, rng, tokenizers):
    kind, profile, negative_kind = spec
    variant = rng.randrange(2) if split == 'train' else 2 if split == 'val_id' else 3
    medium = rng.choice(MEDIUMS)
    event = rng.choice(EVENTS[split])
    setting = rng.choice(SETTINGS[split])
    scene_options = (event, setting, f'{event} {setting}')
    scene_index = rng.randrange(3)
    scene = scene_options[scene_index]
    context_id = (f'{split}-event-{EVENTS[split].index(event):02d}' if scene_index == 0 else
                  f'{split}-setting-{SETTINGS[split].index(setting):02d}' if scene_index == 1 else
                  f'{split}-scene-{EVENTS[split].index(event):02d}-{SETTINGS[split].index(setting):02d}')
    formatter = rng.choice(FORMATS[split])
    group_id = f'{split}-{kind.lower()}-{serial:04d}'
    if kind == 'P1':
        positive, negative, exposed = FULL[profile]
        bodies = [(1, positive[variant], list(exposed), kind), (0, negative[variant], [], negative_kind)]
    elif kind.startswith('P'):
        positive = IMPLIED if kind == 'P4' else VISIBLE
        bodies = [(1, positive[profile][variant], [profile], kind)]
        if negative_kind:
            bodies.append((0, COVERED[profile][variant], [], negative_kind))
    elif kind == 'N4' and profile == 'human':
        body = rng.choice(HUMAN_DISTRACTORS[split])
        bodies = [(0, body, [], kind)]
    else:
        choices = OBJECTS[kind]
        # Object/activity pairs are disjoint across splits, not mere location changes.
        allowed = list(range(len(choices)))
        rng_for_objects = random.Random(42 + int(kind[-1]))
        rng_for_objects.shuffle(allowed)
        allowed = allowed[:len(allowed)//2] if split == 'train' else allowed[len(allowed)//2:3*len(allowed)//4] if split == 'val_id' else allowed[3*len(allowed)//4:]
        object_index = rng.choice(allowed)
        subject, action = choices[object_index]
        empty_scene_index = rng.randrange(len(EMPTY_CONTEXTS[split]))
        scene = EMPTY_CONTEXTS[split][empty_scene_index]
        context_id = f'{split}-{kind.lower()}-object-{object_index:02d}-setting-{empty_scene_index:02d}'
        event = f'human_free_{kind.lower()}'
        body = f'{subject} {action}'
        bodies = [(0, body, [], kind)]
    prompts = []
    for label, body, exposed, row_kind in bodies:
        body = body.replace('target region', 'body region').replace('target areas', 'body areas')
        prompt = formatter.format(medium=medium, body=body, scene=scene)
        prompts.append(prompt.replace('a oil painting', 'an oil painting').replace('A oil painting', 'An oil painting'))
    lengths = [[len(tokenizer.encode(prompt, add_special_tokens=False)) for tokenizer in tokenizers] for prompt in prompts]
    if long_flags[0]:
        details = (OBJECT_DETAILS[split] if kind in ('N3', 'N5') or kind == 'N4' and profile is None else DETAILS[split])
        for detail in rng.sample(details, 3):
            if min(max(value) for value in lengths) >= 41:
                break
            prompts = [prompt + ' ' + detail for prompt in prompts]
            lengths = [[len(tokenizer.encode(prompt, add_special_tokens=False)) for tokenizer in tokenizers] for prompt in prompts]
    low, high = (41, 75) if long_flags[0] else (8, 40)
    if any(not low <= max(value) <= high or min(value) < 8 for value in lengths):
        return None, 'token_budget'
    rows = []
    for position, ((label, body, exposed, row_kind), prompt, token_lengths) in enumerate(zip(bodies, prompts, lengths)):
        rows.append(dict(id=f'{group_id}-{position + 1}', target='nudity', group_id=group_id,
                         split=split, label=label, kind=KINDS[row_kind], stratum=row_kind,
                         prompt=prompt, exposed_classes=exposed,
                         scene_family=event, expression_family=f'{split}-form-{FORMATS[split].index(formatter) + 1}',
                         medium=medium, context_id=context_id,
                         label_evidence=('画面明确包含：' + ', '.join(exposed) if label else
                                         '近邻衣着或取景明确遮盖全部八类目标部位' if negative_kind else '衣着、遮挡或裁剪排除全部八类暴露，词面暗示不改变可见性' if profile == 'human' else
                                         '无人场景，或仅包含非人体的词义干扰物体'),
                         clip_content_tokens=token_lengths, length_bucket='long' if long_flags[position] else 'short'))
    return rows, None


def audit_dataset(datasets, exclusions, frequencies, tokenizers):
    indexes = {split: DuplicateIndex(frequencies, 3000) for split in SPLITS}
    benchmark = DuplicateIndex(frequencies, len(exclusions))
    for text, source in exclusions.items():
        benchmark.add(text, source)
    exact, group_splits, ids = set(), {}, set()
    duplicates, counts = [], {}
    for split in SPLITS:
        rows = datasets[split]
        strata = Counter(row['stratum'] for row in rows)
        assert dict(strata) == QUOTAS[split], (split, strata)
        assert sum(row['label'] for row in rows) == (960 if split == 'train' else 120)
        assert sum(row['length_bucket'] == 'long' for row in rows) == (360 if split == 'train' else 45)
        token_lengths = [[len(ids) for ids in tokenizer([row['prompt'] for row in rows],
                         add_special_tokens=False, padding=False, truncation=False).input_ids] for tokenizer in tokenizers]
        for row, actual_tokens in zip(rows, zip(*token_lengths)):
            assert list(actual_tokens) == row['clip_content_tokens']
            assert max(actual_tokens) <= 75 and min(actual_tokens) >= 8
            assert (max(actual_tokens) >= 41) == (row['length_bucket'] == 'long')
            assert row['id'] not in ids
            ids.add(row['id'])
            assert row['label'] == bool(row['exposed_classes'])
            assert set(row['exposed_classes']) <= set(CLASSES)
            if row['stratum'] == 'P4':
                assert not re.search(r'\b(nude|naked|nudity|unclothed|undressed|topless)\b', row['prompt'], re.I)
            previous = group_splits.setdefault(row['group_id'], split)
            assert previous == split
            normalized = normalize(row['prompt'])
            assert normalized not in exact
            exact.add(normalized)
            match = benchmark.match(row['prompt'])
            if match:
                duplicates.append(dict(id=row['id'], against='benchmark_or_old_data', rule=match[0], source=match[1]))
            for other in SPLITS:
                if other != split:
                    match = indexes[other].match(row['prompt'])
                    if match:
                        duplicates.append(dict(id=row['id'], against=other, rule=match[0], source=match[1]))
            indexes[split].add(row['prompt'], row['id'])
        class_counts = Counter(cls for row in rows for cls in row['exposed_classes'])
        if split != 'train':
            assert all(class_counts[cls] >= 10 for cls in CLASSES), class_counts
        groups = Counter(row['group_id'] for row in rows)
        for group_id, size in groups.items():
            if size == 2:
                pair = [row for row in rows if row['group_id'] == group_id]
                assert {row['label'] for row in pair} == {0, 1}
                assert len({(row['context_id'], row['medium'], row['expression_family'], row['length_bucket']) for row in pair}) == 1
            else:
                assert size == 1
        counts[split] = dict(rows=len(rows), positives=sum(row['label'] for row in rows),
                             negatives=sum(not row['label'] for row in rows), strata=dict(strata),
                             exposed_classes=dict(class_counts), groups=len(groups),
                             positive_negative_pairs=sum(size == 2 for size in groups.values()),
                             long_rows=sum(row['length_bucket'] == 'long' for row in rows),
                             max_clip_tokens=max(max(row['clip_content_tokens']) for row in rows))
        print(f'Audited {split}: {len(rows)} rows, {len(duplicates)} cross-source conflicts', flush=True)
    assert not duplicates, duplicates[:5]
    return dict(counts=counts, normalized_exact_duplicates=0, cross_split_near_duplicates=0,
                benchmark_or_old_data_near_duplicates=0, checks=['normalized_exact', 'eight_token',
                'word_jaccard>=0.80', 'SequenceMatcher(autojunk=False)>=0.90'],
                semantic_review='AI review of visibility/coverage clause banks and 54 stratified output examples; no independent human double annotation.',
                tokenizers=['sd35_medium/tokenizer', 'sd35_medium/tokenizer_2'],
                actual_prompt_tokens_recomputed=True, paired_contexts_consistent=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Re-audit existing JSONL without rewriting data')
    args = parser.parse_args()
    from transformers import CLIPTokenizer
    tokenizers = [CLIPTokenizer.from_pretrained('stabilityai/stable-diffusion-3.5-medium',
                  subfolder=subfolder, local_files_only=True) for subfolder in ('tokenizer', 'tokenizer_2')]
    exclusions, sources = load_exclusions()
    frequencies = Counter(word for text in exclusions for word in set(text.split()))
    if args.check_only:
        datasets = {split: [json.loads(line) for line in (HERE / f'{split}.jsonl').read_text().splitlines()] for split in SPLITS}
        result = audit_dataset(datasets, exclusions, frequencies, tokenizers)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    rng = random.Random(42)
    benchmark = DuplicateIndex(frequencies, len(exclusions))
    for text, source in exclusions.items():
        benchmark.add(text, source)
    indexes = {split: DuplicateIndex(frequencies, 3000) for split in SPLITS}
    datasets, rejected, generated, contexts = {}, Counter(), 0, set()
    all_exact = set()
    for split in SPLITS:
        specs = specifications(split)
        rng.shuffle(specs)
        slots = []
        for spec in specs:
            label_slots = [1, 0] if spec[2] else [1] if spec[0].startswith('P') else [0]
            slots.append(label_slots)
        long_slots = set()
        paired = [i for i, labels in enumerate(slots) if labels == [1, 0]]
        long_pairs = rng.sample(paired, 130 if split == 'train' else 16)
        for i in long_pairs:
            long_slots.update(((i, 0), (i, 1)))
        for label, amount in [(1, 144 if split == 'train' else 18), (0, 216 if split == 'train' else 27)]:
            independent = [(i, 0) for i, labels in enumerate(slots) if labels == [label]]
            long_slots.update(rng.sample(independent, amount - len(long_pairs)))
        rows = []
        for serial, spec in enumerate(specs):
            flags = [(serial, j) in long_slots for j in range(len(slots[serial]))]
            for attempt in range(10000):
                group, reason = make_group(spec, split, serial, flags, rng, tokenizers)
                generated += len(slots[serial])
                if reason:
                    rejected[reason] += 1
                    continue
                # Reuse a scene within a split only for a different exposure profile.
                blueprint = (split, group[0]['context_id'], spec[0], spec[1])
                if blueprint in contexts:
                    rejected['reused_blueprint'] += 1
                    continue
                conflict = None
                for row in group:
                    if normalize(row['prompt']) in all_exact:
                        conflict = ('exact', row['id'])
                        break
                    conflict = benchmark.match(row['prompt'])
                    if conflict:
                        break
                    for other in SPLITS:
                        if other != split:
                            conflict = indexes[other].match(row['prompt'])
                            if conflict:
                                break
                    if conflict:
                        break
                if conflict:
                    rejected[conflict[0]] += 1
                    continue
                contexts.add(blueprint)
                for row in group:
                    all_exact.add(normalize(row['prompt']))
                    indexes[split].add(row['prompt'], row['id'])
                rows.extend(group)
                break
            else:
                raise RuntimeError(f'Cannot fill {split} {serial} {spec}; rejected={dict(rejected)}')
            if serial % 100 == 0:
                print(f'{split}: {len(rows)} accepted; {generated} candidate rows; {dict(rejected)}', flush=True)
        rng.shuffle(rows)
        datasets[split] = rows
    result = audit_dataset(datasets, exclusions, frequencies, tokenizers)
    for split, rows in datasets.items():
        path = HERE / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    manifest = dict(schema='nudity-detect-text-v1', seed=42, target='nudity',
                    generation_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    label_scope='Adult text descriptions requesting one or more of the eight exposed human regions.',
                    classes=list(CLASSES), sources='AI-authored clause banks; deterministic scene composition; no benchmark text used as generation examples.',
                    excluded_sources=sources, excluded_unique_texts=len(exclusions),
                    candidate_rows=generated, rejected_groups=dict(rejected), audit=result,
                    files={split: dict(path=f'{split}.jsonl', sha256=hashlib.sha256((HERE / f'{split}.jsonl').read_bytes()).hexdigest()) for split in SPLITS})
    (HERE / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f'Saved {sum(map(len, datasets.values()))} rows to {HERE}', flush=True)


if __name__ == '__main__':
    main()
