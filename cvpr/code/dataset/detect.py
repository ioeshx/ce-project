

# IP, Obejct testing template
imagenet_templates = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]


# style testing template
painting_templates = [
    "{} style painting of the night sky with bold strokes.",
    "{} style landscape of rolling hills with dramatic brushwork.",
    "Self-portrait in {} style, showcasing distinctive color and texture.",
    "{} style rural scene with dynamic movement and intensity.",
    "Wheat field under a swirling sky in the style of {}.",
    "{} style transformation of simple cottages into profound scenes.",
    "{} style portrait conveying inner turmoil and vitality.",
    "Seascape in {} style capturing tumultuous waves and ocean movement.",
    "{} style depiction of light on water with delicate brushstrokes.",
    "Garden scene in {} style, emphasizing sunlight and shadow effects.",
    "{} style atmospheric haze enveloping the landscape.",
    "Haystacks in {} style, capturing light and color changes.",
    "Cathedral facade in {} style with shifting light and shadow.",
    "{} style vibrant flower garden in full bloom.",
    "{} style painting of boats as a study of color and atmosphere.",
    "{} style cliffs along the coastline with blended sea and sky.",
    "{} style bustling Paris streets with movement and vitality.",
    "Sunrise scene in {} style, capturing the beauty of dawn.",
    "{} style exploration of fog and mist in the landscape.",
    "Olive trees painted in {} style.",
    "Tree painted in {} style.",
    "Women working in the garden, in {} style.",
    "Wheat field painting by {}.",
    "Trees in bloom painted in {} style.",
    "{} style painting of a tree.",
    "Wheat field in {} style.",
    "{} style painting of a field with mountains in the background.",
    "Trees in bloom in the style of {}.",
    "Rocks in the ocean, in {} style.",
    "City painting in {} style.",
]


# Shared by IP and object detection training. Fill each slot with the target
# concept for a positive prompt or a related or unrelated concept for a negative.
detection_training_templates = [
    "At the edge of a courtyard, place {} where a band of sunlight crosses the paving.",
    "Beyond an open archway, reveal {} against the muted colors of a market at dawn.",
    "Let {} anchor a scene on an otherwise empty salt flat after rainfall.",
    "Set {} on the right side of a riverbank scene, across from a line of reeds.",
    "In a misty orchard, leave a gap between two rows of trees for {}.",
    "A red awning frames the upper edge of the scene while {} occupies the open space below.",
    "Arrange the scene around {} where a gravel road forks at the foot of a hill.",
    "Include {} in a snowy square ringed by warm street lamps and quiet storefronts.",
    "Use an open field of lavender as the setting for {}, with a low stone wall in the distance.",
    "Show {} through a gap in the tall grass, with the ground and sky visible beyond.",
    "The wet pavement reflects blue evening light around {}, with warm light spilling from a doorway.",
    "Place {} beside a tidal pool, surrounded by smooth rocks and traces of sea foam.",
    "A strip of pale sky separates {} from a distant row of dark hills.",
    "Compose a scene around {} beside an empty railway platform with a long row of columns.",
    "In a garden of clipped hedges, make {} the focal point at the intersection of two paths.",
    "Frame {} between a bank of ferns in the foreground and a shaded stone wall behind.",
    "Set the scene under an overpass, with {} in the patch of daylight beyond the concrete supports.",
    "Place {} near the bend of a boardwalk that disappears into a marsh.",
    "Build a dusk scene around {}, with lantern light falling across open ground.",
    "Feature {} on the open side of a harbor scene, away from the mooring posts and coiled ropes.",
    "Set {} against a hillside of dry grass, with one dark cloud casting a moving shadow across the slope.",
    "Use the lines of a colonnade to draw the eye toward {} at the far end.",
    "Place {} where a mosaic path meets a broad expanse of pale terracotta tiles.",
    "In a greenhouse shaded by broad leaves, let the light coming through the glass fall on {}.",
    "Set {} apart from the background by leaving clear space between it and a row of striped windbreaks.",
    "Detail the worn paving of a narrow lane, with {} where the lane opens into a square.",
    "Include {} at the edge of a clearing surrounded by tall pines and scattered patches of snow.",
    "A footbridge and its reflection form a circle in the background, with {} nearest the viewer.",
    "On a hillside divided into terraces, reserve the lowest level of the scene for {}.",
    "Arrange {} and a shallow stream on opposite sides of a frame enclosed by rock faces.",
    "Keep {} visible near a row of closed shutters while rain streaks through the background.",
    "Beyond a row of fabric banners, put {} in the center of an otherwise open plaza.",
    "Render the scene at low tide, with {} separated from the sea by a wide strip of rippled sand.",
    "The foreground opens onto a canyon overlook, with {} set against the layers of rock beyond.",
    "Place {} in a roofed passageway, with daylight entering from both ends.",
    "Set {} beside a neatly stacked woodpile, with frost visible across the ground.",
    "Include {} where an open gate leads from a gravel yard into an expanse of meadow.",
    "Let {} provide the focal point among the horizontal bands of fog, water, and willow trees.",
    "An empty stone amphitheater curves around the background, with {} in its open center.",
    "Set {} at the end of a path bordered by shallow pools and low grasses.",
    "Give {} the center of the frame beneath a sky full of lanterns, with plain ground below.",
    "Between a chalk cliff and a quiet beach, set {} near the line where gravel gives way to sand.",
    "Include {} against the patterned shadows cast by a latticework screen.",
    "Start with an open-air passage lined with empty stalls, and place {} at the nearest end.",
    "A stepped embankment descends toward still water, with {} on the stretch of open ground above it.",
    "Place {} along the outer edge of a stone labyrinth, with the curved paths visible behind.",
    "Use a row of translucent panels as a backdrop for {}, with soft light coming through from behind.",
    "After a dust storm, set {} against layered ridges under a copper-colored sky.",
    "At the mouth of an underground passage, put {} where daylight meets the shadow.",
    "Place {} in the sheltered corner of a windswept promenade, with flags lifting in the distance.",
]


style_detection_training_templates = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a good painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


detection_negative_concepts = {
    "Pikachu": {"related": ("Raichu", "Pichu"), "unrelated": ("teapot", "bicycle")},
    "Snoopy": {"related": ("Droopy", "Brian Griffin"), "unrelated": ("teapot", "bicycle")},
    "Van Gogh": {"related": ("Monet", "Paul Gauguin"), "unrelated": ("pixel art", "minimalism")},
    "Monet": {"related": ("Camille Pissarro", "Alfred Sisley"), "unrelated": ("pixel art", "minimalism")},
    "Picasso": {"related": ("Georges Braque", "Juan Gris"), "unrelated": ("pixel art", "minimalism")},
    "church": {"related": ("library", "town hall"), "unrelated": ("apple", "butterfly")},
    "golf ball": {"related": ("tennis ball", "table tennis ball"), "unrelated": ("teapot", "bicycle")},
    "parachute": {"related": ("umbrella", "hot air balloon"), "unrelated": ("teapot", "bicycle")},
    "cassette player": {"related": ("radio", "CD player"), "unrelated": ("apple", "butterfly")},
    "banana": {"related": ("lemon", "pear"), "unrelated": ("teapot", "bicycle")},
}

# Number of rows per target before the joint loss deduplicates the text pool.
detection_expansion_counts = {
    "positive": 30,
    "related_negative": 10,
    "unrelated_negative": 10,
}


def expand_detection_prompts(target):
    """Return the ordered prompt rows with their labels and negative types."""
    if target in ("Van Gogh", "Monet", "Picasso"):
        templates = style_detection_training_templates
    else:
        templates = detection_training_templates

    negatives = detection_negative_concepts[target]
    groups = (
        ("positive", 1, (target,)),
        ("related_negative", 0, negatives["related"]),
        ("unrelated_negative", 0, negatives["unrelated"]),
    )
    samples = []
    for kind, label, concepts in groups:
        # Start each label group at the first template and cycle the substitutions in order.
        for index in range(detection_expansion_counts[kind]):
            concept = concepts[index % len(concepts)]
            template = templates[index % len(templates)]
            samples.append({"prompt": template.format(concept), "label": label, "kind": kind})
    return samples


# Unique text counts after deduplication in the joint detection objective.
detection_training_counts = {
    target: {
        kind: len({row["prompt"] for row in expand_detection_prompts(target) if row["kind"] == kind})
        for kind in detection_expansion_counts
    }
    for target in detection_negative_concepts
}

