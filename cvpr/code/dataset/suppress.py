"""Suppression training prompt pairs grouped by task and target concept."""

# Each pair contains (target_prompt, safe_reference_prompt).
suppression_training_pairs = {
    "i2p": {
        "nudity": [
            (
                "A nude adult standing beside a wooden chair in a studio lit by a skylight.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, standing beside a wooden chair in a studio lit by a skylight.",
            ),
            (
                "A nude adult sitting on a stone bench beside a courtyard fountain.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, sitting on a stone bench beside a courtyard fountain.",
            ),
            (
                "A nude adult reading a newspaper at a kitchen table in morning light.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, reading a newspaper at a kitchen table in morning light.",
            ),
            (
                "A nude adult walking beside a garden hedge on a cloudy afternoon.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, walking beside a garden hedge on a cloudy afternoon.",
            ),
            (
                "A nude adult holding a ceramic bowl in front of a brick wall.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, holding a ceramic bowl in front of a brick wall.",
            ),
            (
                "A nude adult looking through a telescope on a private rooftop at dusk.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, looking through a telescope on a private rooftop at dusk.",
            ),
            (
                "A nude adult sitting beside a chessboard in a room with pale blue walls.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, sitting beside a chessboard in a room with pale blue walls.",
            ),
            (
                "A nude adult watering a fern beside a frosted bathroom window.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, watering a fern beside a frosted bathroom window.",
            ),
            (
                "A nude adult standing at a workbench while shaping a small clay pot.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, standing at a workbench while shaping a small clay pot.",
            ),
            (
                "A nude adult resting on a wooden stool near a closed grand piano.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, resting on a wooden stool near a closed grand piano.",
            ),
            (
                "A nude adult walking across an empty shoreline beneath distant cliffs.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, walking across an empty shoreline beneath distant cliffs.",
            ),
            (
                "A nude adult standing beside a pine tree at the edge of a quiet campsite.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, standing beside a pine tree at the edge of a quiet campsite.",
            ),
            (
                "A nude adult seated on a woven mat overlooking a secluded valley.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, seated on a woven mat overlooking a secluded valley.",
            ),
            (
                "A nude adult examining a seashell beside a shallow rock pool.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, examining a seashell beside a shallow rock pool.",
            ),
            (
                "A nude adult leaning against a balcony railing above a private courtyard.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, leaning against a balcony railing above a private courtyard.",
            ),
            (
                "A nude adult holding a sketchbook in a white-walled art classroom.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, holding a sketchbook in a white-walled art classroom.",
            ),
            (
                "A nude adult sitting on a cushion beside a low table with a tea set.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, sitting on a cushion beside a low table with a tea set.",
            ),
            (
                "A nude adult arranging books on a shelf beneath a brass reading lamp.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, arranging books on a shelf beneath a brass reading lamp.",
            ),
            (
                "A nude adult folding a towel beside a closed wardrobe in a bedroom.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, folding a towel beside a closed wardrobe in a bedroom.",
            ),
            (
                "A nude adult standing near a vintage camera on a tripod in a loft.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, standing near a vintage camera on a tripod in a loft.",
            ),
            (
                "A nude adult walking along a secluded stone path through a bamboo grove.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, walking along a secluded stone path through a bamboo grove.",
            ),
            (
                "A nude adult sitting beside a fireplace in a cabin during a rainstorm.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, sitting beside a fireplace in a cabin during a rainstorm.",
            ),
            (
                "A nude adult holding an apple beside a sunlit orchard gate.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, holding an apple beside a sunlit orchard gate.",
            ),
            (
                "A nude adult looking at a landscape painting hanging above a sideboard.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, looking at a landscape painting hanging above a sideboard.",
            ),
            (
                "A nude adult turning a pottery wheel in a workshop with large windows.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, turning a pottery wheel in a workshop with large windows.",
            ),
            (
                "A nude adult standing beside a grandfather clock in a quiet hallway.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, standing beside a grandfather clock in a quiet hallway.",
            ),
            (
                "A nude adult sitting on a wooden dock with a rowboat tied nearby.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, sitting on a wooden dock with a rowboat tied nearby.",
            ),
            (
                "A nude adult stretching both arms beside a tall indoor plant.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, stretching both arms beside a tall indoor plant.",
            ),
            (
                "A nude adult playing a violin in a room with an arched window.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, playing a violin in a room with an arched window.",
            ),
            (
                "A nude adult reading a map on a table inside a private mountain cabin.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, reading a map on a table inside a private mountain cabin.",
            ),
            (
                "A nude adult watching distant sailboats from a secluded seaside terrace.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, watching distant sailboats from a secluded seaside terrace.",
            ),
            (
                "A nude adult sitting beneath a pergola covered with grapevines.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, sitting beneath a pergola covered with grapevines.",
            ),
            (
                "A nude adult holding a watering can beside a raised vegetable bed.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, holding a watering can beside a raised vegetable bed.",
            ),
            (
                "A nude adult standing beside an unmade bed in a softly lit room.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, standing beside an unmade bed in a softly lit room.",
            ),
            (
                "A nude adult arranging smooth stones beside a small indoor fountain.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, arranging smooth stones beside a small indoor fountain.",
            ),
            (
                "A nude adult drawing on an easel beside a window facing a garden.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, drawing on an easel beside a window facing a garden.",
            ),
            (
                "A nude adult sitting on a rocking chair beside a basket of folded blankets.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, sitting on a rocking chair beside a basket of folded blankets.",
            ),
            (
                "A nude adult walking through a secluded meadow under a clear evening sky.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, walking through a secluded meadow under a clear evening sky.",
            ),
            (
                "A nude adult holding a folded umbrella beside a closed glass door.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, holding a folded umbrella beside a closed glass door.",
            ),
            (
                "A nude adult standing near a bookcase while listening to a tabletop radio.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, standing near a bookcase while listening to a tabletop radio.",
            ),
            (
                "A nude adult seated at a writing desk with an envelope and a fountain pen.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, seated at a writing desk with an envelope and a fountain pen.",
            ),
            (
                "A nude adult looking into a fish pond in a private walled garden.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, looking into a fish pond in a private walled garden.",
            ),
            (
                "A nude adult standing beside a wooden ladder in an empty painting studio.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, standing beside a wooden ladder in an empty painting studio.",
            ),
            (
                "A nude adult sitting near a stained-glass window in a quiet reading room.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, sitting near a stained-glass window in a quiet reading room.",
            ),
            (
                "A nude adult holding a linen cushion beside a low sofa.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, holding a linen cushion beside a low sofa.",
            ),
            (
                "A nude adult standing on a cabin porch beneath a string of unlit lanterns.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, standing on a cabin porch beneath a string of unlit lanterns.",
            ),
            (
                "A nude adult sorting dried flowers at a table in a conservatory.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, sorting dried flowers at a table in a conservatory.",
            ),
            (
                "A nude adult sitting beside a closed suitcase in a sunlit guest room.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, sitting beside a closed suitcase in a sunlit guest room.",
            ),
            (
                "A nude adult standing near a stone basin in a secluded outdoor courtyard.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, standing near a stone basin in a secluded outdoor courtyard.",
            ),
            (
                "A nude adult observing the moon from a private balcony with potted herbs.",
                "A fully clothed adult wearing a long-sleeved shirt and long trousers, observing the moon from a private balcony with potted herbs.",
            ),
        ],
    },
    "ip": {
        "Pikachu": [
            (
                "A covered market has shuttered stalls and strings of red pennants above the central aisle. Pikachu peeks out from behind one of the wooden posts. Use a wide composition with the surroundings visible on both sides.",
                "A covered market has shuttered stalls and strings of red pennants above the central aisle. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A covered market has shuttered stalls and strings of red pennants above the central aisle. Pikachu peeks out from behind one of the wooden posts. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A covered market has shuttered stalls and strings of red pennants above the central aisle. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A covered market has shuttered stalls and strings of red pennants above the central aisle. Pikachu peeks out from behind one of the wooden posts. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A covered market has shuttered stalls and strings of red pennants above the central aisle. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A covered market has shuttered stalls and strings of red pennants above the central aisle. Pikachu peeks out from behind one of the wooden posts. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A covered market has shuttered stalls and strings of red pennants above the central aisle. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A covered market has shuttered stalls and strings of red pennants above the central aisle. Pikachu peeks out from behind one of the wooden posts. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A covered market has shuttered stalls and strings of red pennants above the central aisle. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A train platform runs beside a tiled wall with a green clock and a blue bench. Pikachu waits at the near end of the bench. Use a wide composition with the surroundings visible on both sides.",
                "A train platform runs beside a tiled wall with a green clock and a blue bench. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A train platform runs beside a tiled wall with a green clock and a blue bench. Pikachu waits at the near end of the bench. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A train platform runs beside a tiled wall with a green clock and a blue bench. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A train platform runs beside a tiled wall with a green clock and a blue bench. Pikachu waits at the near end of the bench. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A train platform runs beside a tiled wall with a green clock and a blue bench. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A train platform runs beside a tiled wall with a green clock and a blue bench. Pikachu waits at the near end of the bench. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A train platform runs beside a tiled wall with a green clock and a blue bench. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A train platform runs beside a tiled wall with a green clock and a blue bench. Pikachu waits at the near end of the bench. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A train platform runs beside a tiled wall with a green clock and a blue bench. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A climbing wall rises above a set of folded safety mats in a recreation hall. Pikachu looks up at the colored holds from the edge of the mats. Use a wide composition with the surroundings visible on both sides.",
                "A climbing wall rises above a set of folded safety mats in a recreation hall. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A climbing wall rises above a set of folded safety mats in a recreation hall. Pikachu looks up at the colored holds from the edge of the mats. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A climbing wall rises above a set of folded safety mats in a recreation hall. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A climbing wall rises above a set of folded safety mats in a recreation hall. Pikachu looks up at the colored holds from the edge of the mats. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A climbing wall rises above a set of folded safety mats in a recreation hall. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A climbing wall rises above a set of folded safety mats in a recreation hall. Pikachu looks up at the colored holds from the edge of the mats. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A climbing wall rises above a set of folded safety mats in a recreation hall. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A climbing wall rises above a set of folded safety mats in a recreation hall. Pikachu looks up at the colored holds from the edge of the mats. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A climbing wall rises above a set of folded safety mats in a recreation hall. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A laundromat has a row of round glass doors and a yellow stripe running across the floor. Pikachu sits beside the yellow stripe. Use a wide composition with the surroundings visible on both sides.",
                "A laundromat has a row of round glass doors and a yellow stripe running across the floor. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A laundromat has a row of round glass doors and a yellow stripe running across the floor. Pikachu sits beside the yellow stripe. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A laundromat has a row of round glass doors and a yellow stripe running across the floor. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A laundromat has a row of round glass doors and a yellow stripe running across the floor. Pikachu sits beside the yellow stripe. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A laundromat has a row of round glass doors and a yellow stripe running across the floor. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A laundromat has a row of round glass doors and a yellow stripe running across the floor. Pikachu sits beside the yellow stripe. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A laundromat has a row of round glass doors and a yellow stripe running across the floor. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A laundromat has a row of round glass doors and a yellow stripe running across the floor. Pikachu sits beside the yellow stripe. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A laundromat has a row of round glass doors and a yellow stripe running across the floor. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A museum corridor has a checkerboard floor and a closed brass elevator at the far end. Pikachu stands next to the elevator doors. Use a wide composition with the surroundings visible on both sides.",
                "A museum corridor has a checkerboard floor and a closed brass elevator at the far end. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A museum corridor has a checkerboard floor and a closed brass elevator at the far end. Pikachu stands next to the elevator doors. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A museum corridor has a checkerboard floor and a closed brass elevator at the far end. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A museum corridor has a checkerboard floor and a closed brass elevator at the far end. Pikachu stands next to the elevator doors. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A museum corridor has a checkerboard floor and a closed brass elevator at the far end. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A museum corridor has a checkerboard floor and a closed brass elevator at the far end. Pikachu stands next to the elevator doors. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A museum corridor has a checkerboard floor and a closed brass elevator at the far end. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A museum corridor has a checkerboard floor and a closed brass elevator at the far end. Pikachu stands next to the elevator doors. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A museum corridor has a checkerboard floor and a closed brass elevator at the far end. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A riverside path passes an iron railing and a cluster of yellow warning buoys. Pikachu pauses at the bend in the path. Use a wide composition with the surroundings visible on both sides.",
                "A riverside path passes an iron railing and a cluster of yellow warning buoys. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A riverside path passes an iron railing and a cluster of yellow warning buoys. Pikachu pauses at the bend in the path. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A riverside path passes an iron railing and a cluster of yellow warning buoys. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A riverside path passes an iron railing and a cluster of yellow warning buoys. Pikachu pauses at the bend in the path. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A riverside path passes an iron railing and a cluster of yellow warning buoys. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A riverside path passes an iron railing and a cluster of yellow warning buoys. Pikachu pauses at the bend in the path. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A riverside path passes an iron railing and a cluster of yellow warning buoys. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A riverside path passes an iron railing and a cluster of yellow warning buoys. Pikachu pauses at the bend in the path. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A riverside path passes an iron railing and a cluster of yellow warning buoys. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "An observatory lobby contains a circular information desk and a staircase leading upstairs. Pikachu looks toward the staircase from beside the desk. Use a wide composition with the surroundings visible on both sides.",
                "An observatory lobby contains a circular information desk and a staircase leading upstairs. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "An observatory lobby contains a circular information desk and a staircase leading upstairs. Pikachu looks toward the staircase from beside the desk. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "An observatory lobby contains a circular information desk and a staircase leading upstairs. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "An observatory lobby contains a circular information desk and a staircase leading upstairs. Pikachu looks toward the staircase from beside the desk. Compose a vertical frame that keeps both the near ground and the background visible.",
                "An observatory lobby contains a circular information desk and a staircase leading upstairs. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "An observatory lobby contains a circular information desk and a staircase leading upstairs. Pikachu looks toward the staircase from beside the desk. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "An observatory lobby contains a circular information desk and a staircase leading upstairs. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "An observatory lobby contains a circular information desk and a staircase leading upstairs. Pikachu looks toward the staircase from beside the desk. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "An observatory lobby contains a circular information desk and a staircase leading upstairs. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A greenhouse aisle runs between racks of clay pots and a wall of translucent glass. Pikachu looks through the glass from the end of the aisle. Use a wide composition with the surroundings visible on both sides.",
                "A greenhouse aisle runs between racks of clay pots and a wall of translucent glass. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A greenhouse aisle runs between racks of clay pots and a wall of translucent glass. Pikachu looks through the glass from the end of the aisle. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A greenhouse aisle runs between racks of clay pots and a wall of translucent glass. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A greenhouse aisle runs between racks of clay pots and a wall of translucent glass. Pikachu looks through the glass from the end of the aisle. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A greenhouse aisle runs between racks of clay pots and a wall of translucent glass. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A greenhouse aisle runs between racks of clay pots and a wall of translucent glass. Pikachu looks through the glass from the end of the aisle. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A greenhouse aisle runs between racks of clay pots and a wall of translucent glass. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A greenhouse aisle runs between racks of clay pots and a wall of translucent glass. Pikachu looks through the glass from the end of the aisle. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A greenhouse aisle runs between racks of clay pots and a wall of translucent glass. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A baseball bullpen has a wire fence, a dirt lane, and folded canvas seats. Pikachu sits at the edge of the dirt lane. Use a wide composition with the surroundings visible on both sides.",
                "A baseball bullpen has a wire fence, a dirt lane, and folded canvas seats. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A baseball bullpen has a wire fence, a dirt lane, and folded canvas seats. Pikachu sits at the edge of the dirt lane. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A baseball bullpen has a wire fence, a dirt lane, and folded canvas seats. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A baseball bullpen has a wire fence, a dirt lane, and folded canvas seats. Pikachu sits at the edge of the dirt lane. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A baseball bullpen has a wire fence, a dirt lane, and folded canvas seats. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A baseball bullpen has a wire fence, a dirt lane, and folded canvas seats. Pikachu sits at the edge of the dirt lane. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A baseball bullpen has a wire fence, a dirt lane, and folded canvas seats. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A baseball bullpen has a wire fence, a dirt lane, and folded canvas seats. Pikachu sits at the edge of the dirt lane. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A baseball bullpen has a wire fence, a dirt lane, and folded canvas seats. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A municipal theater foyer has burgundy curtains and dark terrazzo flooring. Pikachu waits beside the gap in the curtains. Use a wide composition with the surroundings visible on both sides.",
                "A municipal theater foyer has burgundy curtains and dark terrazzo flooring. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A municipal theater foyer has burgundy curtains and dark terrazzo flooring. Pikachu waits beside the gap in the curtains. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A municipal theater foyer has burgundy curtains and dark terrazzo flooring. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A municipal theater foyer has burgundy curtains and dark terrazzo flooring. Pikachu waits beside the gap in the curtains. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A municipal theater foyer has burgundy curtains and dark terrazzo flooring. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A municipal theater foyer has burgundy curtains and dark terrazzo flooring. Pikachu waits beside the gap in the curtains. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A municipal theater foyer has burgundy curtains and dark terrazzo flooring. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A municipal theater foyer has burgundy curtains and dark terrazzo flooring. Pikachu waits beside the gap in the curtains. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A municipal theater foyer has burgundy curtains and dark terrazzo flooring. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A fenced rooftop has striped shade cloth, a ventilation duct, and a row of square planters. Pikachu sits between the first two planters. Use a wide composition with the surroundings visible on both sides.",
                "A fenced rooftop has striped shade cloth, a ventilation duct, and a row of square planters. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A fenced rooftop has striped shade cloth, a ventilation duct, and a row of square planters. Pikachu sits between the first two planters. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A fenced rooftop has striped shade cloth, a ventilation duct, and a row of square planters. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A fenced rooftop has striped shade cloth, a ventilation duct, and a row of square planters. Pikachu sits between the first two planters. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A fenced rooftop has striped shade cloth, a ventilation duct, and a row of square planters. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A fenced rooftop has striped shade cloth, a ventilation duct, and a row of square planters. Pikachu sits between the first two planters. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A fenced rooftop has striped shade cloth, a ventilation duct, and a row of square planters. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A fenced rooftop has striped shade cloth, a ventilation duct, and a row of square planters. Pikachu sits between the first two planters. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A fenced rooftop has striped shade cloth, a ventilation duct, and a row of square planters. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A bakery loading yard has a stack of clean wooden crates and a closed roll-up door. Pikachu looks around the side of the crates. Use a wide composition with the surroundings visible on both sides.",
                "A bakery loading yard has a stack of clean wooden crates and a closed roll-up door. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A bakery loading yard has a stack of clean wooden crates and a closed roll-up door. Pikachu looks around the side of the crates. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A bakery loading yard has a stack of clean wooden crates and a closed roll-up door. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A bakery loading yard has a stack of clean wooden crates and a closed roll-up door. Pikachu looks around the side of the crates. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A bakery loading yard has a stack of clean wooden crates and a closed roll-up door. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A bakery loading yard has a stack of clean wooden crates and a closed roll-up door. Pikachu looks around the side of the crates. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A bakery loading yard has a stack of clean wooden crates and a closed roll-up door. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A bakery loading yard has a stack of clean wooden crates and a closed roll-up door. Pikachu looks around the side of the crates. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A bakery loading yard has a stack of clean wooden crates and a closed roll-up door. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A ferry terminal has a covered footpath, metal turnstiles, and a timetable on the far wall. Pikachu waits on the near side of the turnstiles. Use a wide composition with the surroundings visible on both sides.",
                "A ferry terminal has a covered footpath, metal turnstiles, and a timetable on the far wall. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A ferry terminal has a covered footpath, metal turnstiles, and a timetable on the far wall. Pikachu waits on the near side of the turnstiles. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A ferry terminal has a covered footpath, metal turnstiles, and a timetable on the far wall. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A ferry terminal has a covered footpath, metal turnstiles, and a timetable on the far wall. Pikachu waits on the near side of the turnstiles. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A ferry terminal has a covered footpath, metal turnstiles, and a timetable on the far wall. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A ferry terminal has a covered footpath, metal turnstiles, and a timetable on the far wall. Pikachu waits on the near side of the turnstiles. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A ferry terminal has a covered footpath, metal turnstiles, and a timetable on the far wall. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A ferry terminal has a covered footpath, metal turnstiles, and a timetable on the far wall. Pikachu waits on the near side of the turnstiles. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A ferry terminal has a covered footpath, metal turnstiles, and a timetable on the far wall. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A small library has a book return slot and two empty wicker chairs by the entrance. Pikachu sits on the floor between the chairs. Use a wide composition with the surroundings visible on both sides.",
                "A small library has a book return slot and two empty wicker chairs by the entrance. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A small library has a book return slot and two empty wicker chairs by the entrance. Pikachu sits on the floor between the chairs. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A small library has a book return slot and two empty wicker chairs by the entrance. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A small library has a book return slot and two empty wicker chairs by the entrance. Pikachu sits on the floor between the chairs. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A small library has a book return slot and two empty wicker chairs by the entrance. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A small library has a book return slot and two empty wicker chairs by the entrance. Pikachu sits on the floor between the chairs. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A small library has a book return slot and two empty wicker chairs by the entrance. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A small library has a book return slot and two empty wicker chairs by the entrance. Pikachu sits on the floor between the chairs. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A small library has a book return slot and two empty wicker chairs by the entrance. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A swimming pool entrance has a ticket window, pale blue tile, and a rack of swim caps. Pikachu stands beside the rack. Use a wide composition with the surroundings visible on both sides.",
                "A swimming pool entrance has a ticket window, pale blue tile, and a rack of swim caps. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A swimming pool entrance has a ticket window, pale blue tile, and a rack of swim caps. Pikachu stands beside the rack. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A swimming pool entrance has a ticket window, pale blue tile, and a rack of swim caps. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A swimming pool entrance has a ticket window, pale blue tile, and a rack of swim caps. Pikachu stands beside the rack. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A swimming pool entrance has a ticket window, pale blue tile, and a rack of swim caps. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A swimming pool entrance has a ticket window, pale blue tile, and a rack of swim caps. Pikachu stands beside the rack. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A swimming pool entrance has a ticket window, pale blue tile, and a rack of swim caps. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A swimming pool entrance has a ticket window, pale blue tile, and a rack of swim caps. Pikachu stands beside the rack. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A swimming pool entrance has a ticket window, pale blue tile, and a rack of swim caps. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A tool library has a plywood workbench, wall hooks, and a partly open side door. Pikachu looks in from the doorway. Use a wide composition with the surroundings visible on both sides.",
                "A tool library has a plywood workbench, wall hooks, and a partly open side door. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A tool library has a plywood workbench, wall hooks, and a partly open side door. Pikachu looks in from the doorway. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A tool library has a plywood workbench, wall hooks, and a partly open side door. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A tool library has a plywood workbench, wall hooks, and a partly open side door. Pikachu looks in from the doorway. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A tool library has a plywood workbench, wall hooks, and a partly open side door. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A tool library has a plywood workbench, wall hooks, and a partly open side door. Pikachu looks in from the doorway. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A tool library has a plywood workbench, wall hooks, and a partly open side door. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A tool library has a plywood workbench, wall hooks, and a partly open side door. Pikachu looks in from the doorway. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A tool library has a plywood workbench, wall hooks, and a partly open side door. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
        ],
        "Snoopy": [
            (
                "A covered market has shuttered stalls and strings of red pennants above the central aisle. Snoopy peeks out from behind one of the wooden posts. Use a wide composition with the surroundings visible on both sides.",
                "A covered market has shuttered stalls and strings of red pennants above the central aisle. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A covered market has shuttered stalls and strings of red pennants above the central aisle. Snoopy peeks out from behind one of the wooden posts. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A covered market has shuttered stalls and strings of red pennants above the central aisle. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A covered market has shuttered stalls and strings of red pennants above the central aisle. Snoopy peeks out from behind one of the wooden posts. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A covered market has shuttered stalls and strings of red pennants above the central aisle. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A covered market has shuttered stalls and strings of red pennants above the central aisle. Snoopy peeks out from behind one of the wooden posts. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A covered market has shuttered stalls and strings of red pennants above the central aisle. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A covered market has shuttered stalls and strings of red pennants above the central aisle. Snoopy peeks out from behind one of the wooden posts. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A covered market has shuttered stalls and strings of red pennants above the central aisle. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A train platform runs beside a tiled wall with a green clock and a blue bench. Snoopy waits at the near end of the bench. Use a wide composition with the surroundings visible on both sides.",
                "A train platform runs beside a tiled wall with a green clock and a blue bench. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A train platform runs beside a tiled wall with a green clock and a blue bench. Snoopy waits at the near end of the bench. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A train platform runs beside a tiled wall with a green clock and a blue bench. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A train platform runs beside a tiled wall with a green clock and a blue bench. Snoopy waits at the near end of the bench. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A train platform runs beside a tiled wall with a green clock and a blue bench. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A train platform runs beside a tiled wall with a green clock and a blue bench. Snoopy waits at the near end of the bench. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A train platform runs beside a tiled wall with a green clock and a blue bench. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A train platform runs beside a tiled wall with a green clock and a blue bench. Snoopy waits at the near end of the bench. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A train platform runs beside a tiled wall with a green clock and a blue bench. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A climbing wall rises above a set of folded safety mats in a recreation hall. Snoopy looks up at the colored holds from the edge of the mats. Use a wide composition with the surroundings visible on both sides.",
                "A climbing wall rises above a set of folded safety mats in a recreation hall. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A climbing wall rises above a set of folded safety mats in a recreation hall. Snoopy looks up at the colored holds from the edge of the mats. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A climbing wall rises above a set of folded safety mats in a recreation hall. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A climbing wall rises above a set of folded safety mats in a recreation hall. Snoopy looks up at the colored holds from the edge of the mats. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A climbing wall rises above a set of folded safety mats in a recreation hall. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A climbing wall rises above a set of folded safety mats in a recreation hall. Snoopy looks up at the colored holds from the edge of the mats. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A climbing wall rises above a set of folded safety mats in a recreation hall. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A climbing wall rises above a set of folded safety mats in a recreation hall. Snoopy looks up at the colored holds from the edge of the mats. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A climbing wall rises above a set of folded safety mats in a recreation hall. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A laundromat has a row of round glass doors and a yellow stripe running across the floor. Snoopy sits beside the yellow stripe. Use a wide composition with the surroundings visible on both sides.",
                "A laundromat has a row of round glass doors and a yellow stripe running across the floor. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A laundromat has a row of round glass doors and a yellow stripe running across the floor. Snoopy sits beside the yellow stripe. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A laundromat has a row of round glass doors and a yellow stripe running across the floor. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A laundromat has a row of round glass doors and a yellow stripe running across the floor. Snoopy sits beside the yellow stripe. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A laundromat has a row of round glass doors and a yellow stripe running across the floor. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A laundromat has a row of round glass doors and a yellow stripe running across the floor. Snoopy sits beside the yellow stripe. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A laundromat has a row of round glass doors and a yellow stripe running across the floor. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A laundromat has a row of round glass doors and a yellow stripe running across the floor. Snoopy sits beside the yellow stripe. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A laundromat has a row of round glass doors and a yellow stripe running across the floor. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A museum corridor has a checkerboard floor and a closed brass elevator at the far end. Snoopy stands next to the elevator doors. Use a wide composition with the surroundings visible on both sides.",
                "A museum corridor has a checkerboard floor and a closed brass elevator at the far end. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A museum corridor has a checkerboard floor and a closed brass elevator at the far end. Snoopy stands next to the elevator doors. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A museum corridor has a checkerboard floor and a closed brass elevator at the far end. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A museum corridor has a checkerboard floor and a closed brass elevator at the far end. Snoopy stands next to the elevator doors. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A museum corridor has a checkerboard floor and a closed brass elevator at the far end. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A museum corridor has a checkerboard floor and a closed brass elevator at the far end. Snoopy stands next to the elevator doors. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A museum corridor has a checkerboard floor and a closed brass elevator at the far end. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A museum corridor has a checkerboard floor and a closed brass elevator at the far end. Snoopy stands next to the elevator doors. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A museum corridor has a checkerboard floor and a closed brass elevator at the far end. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A riverside path passes an iron railing and a cluster of yellow warning buoys. Snoopy pauses at the bend in the path. Use a wide composition with the surroundings visible on both sides.",
                "A riverside path passes an iron railing and a cluster of yellow warning buoys. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A riverside path passes an iron railing and a cluster of yellow warning buoys. Snoopy pauses at the bend in the path. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A riverside path passes an iron railing and a cluster of yellow warning buoys. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A riverside path passes an iron railing and a cluster of yellow warning buoys. Snoopy pauses at the bend in the path. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A riverside path passes an iron railing and a cluster of yellow warning buoys. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A riverside path passes an iron railing and a cluster of yellow warning buoys. Snoopy pauses at the bend in the path. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A riverside path passes an iron railing and a cluster of yellow warning buoys. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A riverside path passes an iron railing and a cluster of yellow warning buoys. Snoopy pauses at the bend in the path. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A riverside path passes an iron railing and a cluster of yellow warning buoys. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "An observatory lobby contains a circular information desk and a staircase leading upstairs. Snoopy looks toward the staircase from beside the desk. Use a wide composition with the surroundings visible on both sides.",
                "An observatory lobby contains a circular information desk and a staircase leading upstairs. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "An observatory lobby contains a circular information desk and a staircase leading upstairs. Snoopy looks toward the staircase from beside the desk. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "An observatory lobby contains a circular information desk and a staircase leading upstairs. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "An observatory lobby contains a circular information desk and a staircase leading upstairs. Snoopy looks toward the staircase from beside the desk. Compose a vertical frame that keeps both the near ground and the background visible.",
                "An observatory lobby contains a circular information desk and a staircase leading upstairs. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "An observatory lobby contains a circular information desk and a staircase leading upstairs. Snoopy looks toward the staircase from beside the desk. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "An observatory lobby contains a circular information desk and a staircase leading upstairs. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "An observatory lobby contains a circular information desk and a staircase leading upstairs. Snoopy looks toward the staircase from beside the desk. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "An observatory lobby contains a circular information desk and a staircase leading upstairs. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A greenhouse aisle runs between racks of clay pots and a wall of translucent glass. Snoopy looks through the glass from the end of the aisle. Use a wide composition with the surroundings visible on both sides.",
                "A greenhouse aisle runs between racks of clay pots and a wall of translucent glass. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A greenhouse aisle runs between racks of clay pots and a wall of translucent glass. Snoopy looks through the glass from the end of the aisle. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A greenhouse aisle runs between racks of clay pots and a wall of translucent glass. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A greenhouse aisle runs between racks of clay pots and a wall of translucent glass. Snoopy looks through the glass from the end of the aisle. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A greenhouse aisle runs between racks of clay pots and a wall of translucent glass. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A greenhouse aisle runs between racks of clay pots and a wall of translucent glass. Snoopy looks through the glass from the end of the aisle. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A greenhouse aisle runs between racks of clay pots and a wall of translucent glass. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A greenhouse aisle runs between racks of clay pots and a wall of translucent glass. Snoopy looks through the glass from the end of the aisle. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A greenhouse aisle runs between racks of clay pots and a wall of translucent glass. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A baseball bullpen has a wire fence, a dirt lane, and folded canvas seats. Snoopy sits at the edge of the dirt lane. Use a wide composition with the surroundings visible on both sides.",
                "A baseball bullpen has a wire fence, a dirt lane, and folded canvas seats. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A baseball bullpen has a wire fence, a dirt lane, and folded canvas seats. Snoopy sits at the edge of the dirt lane. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A baseball bullpen has a wire fence, a dirt lane, and folded canvas seats. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A baseball bullpen has a wire fence, a dirt lane, and folded canvas seats. Snoopy sits at the edge of the dirt lane. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A baseball bullpen has a wire fence, a dirt lane, and folded canvas seats. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A baseball bullpen has a wire fence, a dirt lane, and folded canvas seats. Snoopy sits at the edge of the dirt lane. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A baseball bullpen has a wire fence, a dirt lane, and folded canvas seats. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A baseball bullpen has a wire fence, a dirt lane, and folded canvas seats. Snoopy sits at the edge of the dirt lane. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A baseball bullpen has a wire fence, a dirt lane, and folded canvas seats. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A municipal theater foyer has burgundy curtains and dark terrazzo flooring. Snoopy waits beside the gap in the curtains. Use a wide composition with the surroundings visible on both sides.",
                "A municipal theater foyer has burgundy curtains and dark terrazzo flooring. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A municipal theater foyer has burgundy curtains and dark terrazzo flooring. Snoopy waits beside the gap in the curtains. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A municipal theater foyer has burgundy curtains and dark terrazzo flooring. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A municipal theater foyer has burgundy curtains and dark terrazzo flooring. Snoopy waits beside the gap in the curtains. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A municipal theater foyer has burgundy curtains and dark terrazzo flooring. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A municipal theater foyer has burgundy curtains and dark terrazzo flooring. Snoopy waits beside the gap in the curtains. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A municipal theater foyer has burgundy curtains and dark terrazzo flooring. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A municipal theater foyer has burgundy curtains and dark terrazzo flooring. Snoopy waits beside the gap in the curtains. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A municipal theater foyer has burgundy curtains and dark terrazzo flooring. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A fenced rooftop has striped shade cloth, a ventilation duct, and a row of square planters. Snoopy sits between the first two planters. Use a wide composition with the surroundings visible on both sides.",
                "A fenced rooftop has striped shade cloth, a ventilation duct, and a row of square planters. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A fenced rooftop has striped shade cloth, a ventilation duct, and a row of square planters. Snoopy sits between the first two planters. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A fenced rooftop has striped shade cloth, a ventilation duct, and a row of square planters. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A fenced rooftop has striped shade cloth, a ventilation duct, and a row of square planters. Snoopy sits between the first two planters. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A fenced rooftop has striped shade cloth, a ventilation duct, and a row of square planters. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A fenced rooftop has striped shade cloth, a ventilation duct, and a row of square planters. Snoopy sits between the first two planters. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A fenced rooftop has striped shade cloth, a ventilation duct, and a row of square planters. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A fenced rooftop has striped shade cloth, a ventilation duct, and a row of square planters. Snoopy sits between the first two planters. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A fenced rooftop has striped shade cloth, a ventilation duct, and a row of square planters. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A bakery loading yard has a stack of clean wooden crates and a closed roll-up door. Snoopy looks around the side of the crates. Use a wide composition with the surroundings visible on both sides.",
                "A bakery loading yard has a stack of clean wooden crates and a closed roll-up door. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A bakery loading yard has a stack of clean wooden crates and a closed roll-up door. Snoopy looks around the side of the crates. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A bakery loading yard has a stack of clean wooden crates and a closed roll-up door. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A bakery loading yard has a stack of clean wooden crates and a closed roll-up door. Snoopy looks around the side of the crates. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A bakery loading yard has a stack of clean wooden crates and a closed roll-up door. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A bakery loading yard has a stack of clean wooden crates and a closed roll-up door. Snoopy looks around the side of the crates. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A bakery loading yard has a stack of clean wooden crates and a closed roll-up door. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A bakery loading yard has a stack of clean wooden crates and a closed roll-up door. Snoopy looks around the side of the crates. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A bakery loading yard has a stack of clean wooden crates and a closed roll-up door. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A ferry terminal has a covered footpath, metal turnstiles, and a timetable on the far wall. Snoopy waits on the near side of the turnstiles. Use a wide composition with the surroundings visible on both sides.",
                "A ferry terminal has a covered footpath, metal turnstiles, and a timetable on the far wall. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A ferry terminal has a covered footpath, metal turnstiles, and a timetable on the far wall. Snoopy waits on the near side of the turnstiles. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A ferry terminal has a covered footpath, metal turnstiles, and a timetable on the far wall. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A ferry terminal has a covered footpath, metal turnstiles, and a timetable on the far wall. Snoopy waits on the near side of the turnstiles. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A ferry terminal has a covered footpath, metal turnstiles, and a timetable on the far wall. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A ferry terminal has a covered footpath, metal turnstiles, and a timetable on the far wall. Snoopy waits on the near side of the turnstiles. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A ferry terminal has a covered footpath, metal turnstiles, and a timetable on the far wall. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A ferry terminal has a covered footpath, metal turnstiles, and a timetable on the far wall. Snoopy waits on the near side of the turnstiles. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A ferry terminal has a covered footpath, metal turnstiles, and a timetable on the far wall. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A small library has a book return slot and two empty wicker chairs by the entrance. Snoopy sits on the floor between the chairs. Use a wide composition with the surroundings visible on both sides.",
                "A small library has a book return slot and two empty wicker chairs by the entrance. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A small library has a book return slot and two empty wicker chairs by the entrance. Snoopy sits on the floor between the chairs. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A small library has a book return slot and two empty wicker chairs by the entrance. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A small library has a book return slot and two empty wicker chairs by the entrance. Snoopy sits on the floor between the chairs. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A small library has a book return slot and two empty wicker chairs by the entrance. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A small library has a book return slot and two empty wicker chairs by the entrance. Snoopy sits on the floor between the chairs. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A small library has a book return slot and two empty wicker chairs by the entrance. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A small library has a book return slot and two empty wicker chairs by the entrance. Snoopy sits on the floor between the chairs. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A small library has a book return slot and two empty wicker chairs by the entrance. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A swimming pool entrance has a ticket window, pale blue tile, and a rack of swim caps. Snoopy stands beside the rack. Use a wide composition with the surroundings visible on both sides.",
                "A swimming pool entrance has a ticket window, pale blue tile, and a rack of swim caps. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A swimming pool entrance has a ticket window, pale blue tile, and a rack of swim caps. Snoopy stands beside the rack. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A swimming pool entrance has a ticket window, pale blue tile, and a rack of swim caps. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A swimming pool entrance has a ticket window, pale blue tile, and a rack of swim caps. Snoopy stands beside the rack. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A swimming pool entrance has a ticket window, pale blue tile, and a rack of swim caps. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A swimming pool entrance has a ticket window, pale blue tile, and a rack of swim caps. Snoopy stands beside the rack. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A swimming pool entrance has a ticket window, pale blue tile, and a rack of swim caps. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A swimming pool entrance has a ticket window, pale blue tile, and a rack of swim caps. Snoopy stands beside the rack. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A swimming pool entrance has a ticket window, pale blue tile, and a rack of swim caps. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A tool library has a plywood workbench, wall hooks, and a partly open side door. Snoopy looks in from the doorway. Use a wide composition with the surroundings visible on both sides.",
                "A tool library has a plywood workbench, wall hooks, and a partly open side door. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A tool library has a plywood workbench, wall hooks, and a partly open side door. Snoopy looks in from the doorway. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A tool library has a plywood workbench, wall hooks, and a partly open side door. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A tool library has a plywood workbench, wall hooks, and a partly open side door. Snoopy looks in from the doorway. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A tool library has a plywood workbench, wall hooks, and a partly open side door. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A tool library has a plywood workbench, wall hooks, and a partly open side door. Snoopy looks in from the doorway. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A tool library has a plywood workbench, wall hooks, and a partly open side door. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A tool library has a plywood workbench, wall hooks, and a partly open side door. Snoopy looks in from the doorway. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A tool library has a plywood workbench, wall hooks, and a partly open side door. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
        ],
    },
    "style": {
        "Van Gogh": [
            (
                "Two cooks stand at opposite ends of a bakery counter while rows of dough rest on trays. Render the scene in the visual style of Van Gogh.",
                "Two cooks stand at opposite ends of a bakery counter while rows of dough rest on trays. Depict the scene as a realistic photograph.",
            ),
            (
                "A freight elevator opens onto a storage floor with stacked archive boxes and a red fire door. Use Van Gogh's visual style throughout.",
                "A freight elevator opens onto a storage floor with stacked archive boxes and a red fire door. Depict the scene as a realistic photograph.",
            ),
            (
                "A pair of cyclists repairs a tire beside an orange locker on an indoor practice track. Interpret the scene through the visual language of Van Gogh.",
                "A pair of cyclists repairs a tire beside an orange locker on an indoor practice track. Depict the scene as a realistic photograph.",
            ),
            (
                "Inside a laundromat, a customer folds pale towels under a line of round dryer doors. Render the scene in the visual style of Van Gogh.",
                "Inside a laundromat, a customer folds pale towels under a line of round dryer doors. Depict the scene as a realistic photograph.",
            ),
            (
                "A tailor threads a needle beneath a floor lamp, between two mannequins and an open cabinet of fabric. Use Van Gogh's visual style throughout.",
                "A tailor threads a needle beneath a floor lamp, between two mannequins and an open cabinet of fabric. Depict the scene as a realistic photograph.",
            ),
            (
                "A brass bell and a roll of numbered tickets sit on a terracotta counter in a museum foyer. Interpret the scene through the visual language of Van Gogh.",
                "A brass bell and a roll of numbered tickets sit on a terracotta counter in a museum foyer. Depict the scene as a realistic photograph.",
            ),
            (
                "A station cleaner sweeps a tiled platform while the final carriage waits beyond the posts. Render the scene in the visual style of Van Gogh.",
                "A station cleaner sweeps a tiled platform while the final carriage waits beyond the posts. Depict the scene as a realistic photograph.",
            ),
            (
                "An archivist opens a document case beside a tall steel cart and a stack of cardboard folders. Use Van Gogh's visual style throughout.",
                "An archivist opens a document case beside a tall steel cart and a stack of cardboard folders. Depict the scene as a realistic photograph.",
            ),
            (
                "A narrow machine shop has two facing benches and a row of numbered drawers painted green. Interpret the scene through the visual language of Van Gogh.",
                "A narrow machine shop has two facing benches and a row of numbered drawers painted green. Depict the scene as a realistic photograph.",
            ),
            (
                "A sound editor sits at a desk with two small speakers and a shelf of labeled tape boxes. Render the scene in the visual style of Van Gogh.",
                "A sound editor sits at a desk with two small speakers and a shelf of labeled tape boxes. Depict the scene as a realistic photograph.",
            ),
            (
                "In an indoor pool, a person stands at the end of a lane divided by red and white floats. Use Van Gogh's visual style throughout.",
                "In an indoor pool, a person stands at the end of a lane divided by red and white floats. Depict the scene as a realistic photograph.",
            ),
            (
                "A kitchen rack holds stacked plates, a copper pan, and a folded apron beside a closed service window. Interpret the scene through the visual language of Van Gogh.",
                "A kitchen rack holds stacked plates, a copper pan, and a folded apron beside a closed service window. Depict the scene as a realistic photograph.",
            ),
            (
                "A young adult laces a roller skate on a wooden bench while another holds the door open. Render the scene in the visual style of Van Gogh.",
                "A young adult laces a roller skate on a wooden bench while another holds the door open. Depict the scene as a realistic photograph.",
            ),
            (
                "An empty cinema lobby has a low ticket booth and a short flight of stairs between dark curtains. Use Van Gogh's visual style throughout.",
                "An empty cinema lobby has a low ticket booth and a short flight of stairs between dark curtains. Depict the scene as a realistic photograph.",
            ),
            (
                "A row of subway turnstiles meets a tiled wall with a route diagram above it. Interpret the scene through the visual language of Van Gogh.",
                "A row of subway turnstiles meets a tiled wall with a route diagram above it. Depict the scene as a realistic photograph.",
            ),
            (
                "A mechanic checks a chain at a bicycle stand while a customer waits near a steel cabinet. Render the scene in the visual style of Van Gogh.",
                "A mechanic checks a chain at a bicycle stand while a customer waits near a steel cabinet. Depict the scene as a realistic photograph.",
            ),
            (
                "Three ceramic cups sit in a drying rack beside a stack of folded towels and a basin. Use Van Gogh's visual style throughout.",
                "Three ceramic cups sit in a drying rack beside a stack of folded towels and a basin. Depict the scene as a realistic photograph.",
            ),
            (
                "A student walks through a school corridor lined with upright lockers and white tiles. Interpret the scene through the visual language of Van Gogh.",
                "A student walks through a school corridor lined with upright lockers and white tiles. Depict the scene as a realistic photograph.",
            ),
            (
                "An empty rehearsal room contains two low stools and a wall of acoustic panels. Render the scene in the visual style of Van Gogh.",
                "An empty rehearsal room contains two low stools and a wall of acoustic panels. Depict the scene as a realistic photograph.",
            ),
            (
                "A square dining table holds a glass jug and two place settings beneath a white fabric shade. Use Van Gogh's visual style throughout.",
                "A square dining table holds a glass jug and two place settings beneath a white fabric shade. Depict the scene as a realistic photograph.",
            ),
            (
                "At an airport baggage carousel, a person waits beside a short stack of travel cases. Interpret the scene through the visual language of Van Gogh.",
                "At an airport baggage carousel, a person waits beside a short stack of travel cases. Depict the scene as a realistic photograph.",
            ),
            (
                "Two volunteers fold fabric banners in a sports hall with markings on the wooden floor. Render the scene in the visual style of Van Gogh.",
                "Two volunteers fold fabric banners in a sports hall with markings on the wooden floor. Depict the scene as a realistic photograph.",
            ),
            (
                "A technician stands at a lab bench with three glass containers arranged by size. Use Van Gogh's visual style throughout.",
                "A technician stands at a lab bench with three glass containers arranged by size. Depict the scene as a realistic photograph.",
            ),
            (
                "A green workbench holds a shallow tray of fasteners and a folded sheet of instructions. Interpret the scene through the visual language of Van Gogh.",
                "A green workbench holds a shallow tray of fasteners and a folded sheet of instructions. Depict the scene as a realistic photograph.",
            ),
            (
                "A delivery worker carries a cardboard box past a row of sheltered vending machines. Render the scene in the visual style of Van Gogh.",
                "A delivery worker carries a cardboard box past a row of sheltered vending machines. Depict the scene as a realistic photograph.",
            ),
            (
                "An attendant arranges a stack of towels at a reception desk beside a glass partition. Use Van Gogh's visual style throughout.",
                "An attendant arranges a stack of towels at a reception desk beside a glass partition. Depict the scene as a realistic photograph.",
            ),
            (
                "A reading desk in an archive has a padded book cradle and a small square reading lamp. Interpret the scene through the visual language of Van Gogh.",
                "A reading desk in an archive has a padded book cradle and a small square reading lamp. Depict the scene as a realistic photograph.",
            ),
            (
                "A bookstore worker rolls a short ladder past three shelves of upright books. Render the scene in the visual style of Van Gogh.",
                "A bookstore worker rolls a short ladder past three shelves of upright books. Depict the scene as a realistic photograph.",
            ),
            (
                "A diner booth holds a stainless steel napkin holder, a water glass, and two paper menus. Use Van Gogh's visual style throughout.",
                "A diner booth holds a stainless steel napkin holder, a water glass, and two paper menus. Depict the scene as a realistic photograph.",
            ),
            (
                "An empty concert hall surrounds a circular platform with a ring of tall wooden seats. Interpret the scene through the visual language of Van Gogh.",
                "An empty concert hall surrounds a circular platform with a ring of tall wooden seats. Depict the scene as a realistic photograph.",
            ),
        ],
        "Monet": [
            (
                "Two cooks stand at opposite ends of a bakery counter while rows of dough rest on trays. Render the scene in the visual style of Monet.",
                "Two cooks stand at opposite ends of a bakery counter while rows of dough rest on trays. Depict the scene as a realistic photograph.",
            ),
            (
                "A freight elevator opens onto a storage floor with stacked archive boxes and a red fire door. Use Monet's visual style throughout.",
                "A freight elevator opens onto a storage floor with stacked archive boxes and a red fire door. Depict the scene as a realistic photograph.",
            ),
            (
                "A pair of cyclists repairs a tire beside an orange locker on an indoor practice track. Interpret the scene through the visual language of Monet.",
                "A pair of cyclists repairs a tire beside an orange locker on an indoor practice track. Depict the scene as a realistic photograph.",
            ),
            (
                "Inside a laundromat, a customer folds pale towels under a line of round dryer doors. Render the scene in the visual style of Monet.",
                "Inside a laundromat, a customer folds pale towels under a line of round dryer doors. Depict the scene as a realistic photograph.",
            ),
            (
                "A tailor threads a needle beneath a floor lamp, between two mannequins and an open cabinet of fabric. Use Monet's visual style throughout.",
                "A tailor threads a needle beneath a floor lamp, between two mannequins and an open cabinet of fabric. Depict the scene as a realistic photograph.",
            ),
            (
                "A brass bell and a roll of numbered tickets sit on a terracotta counter in a museum foyer. Interpret the scene through the visual language of Monet.",
                "A brass bell and a roll of numbered tickets sit on a terracotta counter in a museum foyer. Depict the scene as a realistic photograph.",
            ),
            (
                "A station cleaner sweeps a tiled platform while the final carriage waits beyond the posts. Render the scene in the visual style of Monet.",
                "A station cleaner sweeps a tiled platform while the final carriage waits beyond the posts. Depict the scene as a realistic photograph.",
            ),
            (
                "An archivist opens a document case beside a tall steel cart and a stack of cardboard folders. Use Monet's visual style throughout.",
                "An archivist opens a document case beside a tall steel cart and a stack of cardboard folders. Depict the scene as a realistic photograph.",
            ),
            (
                "A narrow machine shop has two facing benches and a row of numbered drawers painted green. Interpret the scene through the visual language of Monet.",
                "A narrow machine shop has two facing benches and a row of numbered drawers painted green. Depict the scene as a realistic photograph.",
            ),
            (
                "A sound editor sits at a desk with two small speakers and a shelf of labeled tape boxes. Render the scene in the visual style of Monet.",
                "A sound editor sits at a desk with two small speakers and a shelf of labeled tape boxes. Depict the scene as a realistic photograph.",
            ),
            (
                "In an indoor pool, a person stands at the end of a lane divided by red and white floats. Use Monet's visual style throughout.",
                "In an indoor pool, a person stands at the end of a lane divided by red and white floats. Depict the scene as a realistic photograph.",
            ),
            (
                "A kitchen rack holds stacked plates, a copper pan, and a folded apron beside a closed service window. Interpret the scene through the visual language of Monet.",
                "A kitchen rack holds stacked plates, a copper pan, and a folded apron beside a closed service window. Depict the scene as a realistic photograph.",
            ),
            (
                "A young adult laces a roller skate on a wooden bench while another holds the door open. Render the scene in the visual style of Monet.",
                "A young adult laces a roller skate on a wooden bench while another holds the door open. Depict the scene as a realistic photograph.",
            ),
            (
                "An empty cinema lobby has a low ticket booth and a short flight of stairs between dark curtains. Use Monet's visual style throughout.",
                "An empty cinema lobby has a low ticket booth and a short flight of stairs between dark curtains. Depict the scene as a realistic photograph.",
            ),
            (
                "A row of subway turnstiles meets a tiled wall with a route diagram above it. Interpret the scene through the visual language of Monet.",
                "A row of subway turnstiles meets a tiled wall with a route diagram above it. Depict the scene as a realistic photograph.",
            ),
            (
                "A mechanic checks a chain at a bicycle stand while a customer waits near a steel cabinet. Render the scene in the visual style of Monet.",
                "A mechanic checks a chain at a bicycle stand while a customer waits near a steel cabinet. Depict the scene as a realistic photograph.",
            ),
            (
                "Three ceramic cups sit in a drying rack beside a stack of folded towels and a basin. Use Monet's visual style throughout.",
                "Three ceramic cups sit in a drying rack beside a stack of folded towels and a basin. Depict the scene as a realistic photograph.",
            ),
            (
                "A student walks through a school corridor lined with upright lockers and white tiles. Interpret the scene through the visual language of Monet.",
                "A student walks through a school corridor lined with upright lockers and white tiles. Depict the scene as a realistic photograph.",
            ),
            (
                "An empty rehearsal room contains two low stools and a wall of acoustic panels. Render the scene in the visual style of Monet.",
                "An empty rehearsal room contains two low stools and a wall of acoustic panels. Depict the scene as a realistic photograph.",
            ),
            (
                "A square dining table holds a glass jug and two place settings beneath a white fabric shade. Use Monet's visual style throughout.",
                "A square dining table holds a glass jug and two place settings beneath a white fabric shade. Depict the scene as a realistic photograph.",
            ),
            (
                "At an airport baggage carousel, a person waits beside a short stack of travel cases. Interpret the scene through the visual language of Monet.",
                "At an airport baggage carousel, a person waits beside a short stack of travel cases. Depict the scene as a realistic photograph.",
            ),
            (
                "Two volunteers fold fabric banners in a sports hall with markings on the wooden floor. Render the scene in the visual style of Monet.",
                "Two volunteers fold fabric banners in a sports hall with markings on the wooden floor. Depict the scene as a realistic photograph.",
            ),
            (
                "A technician stands at a lab bench with three glass containers arranged by size. Use Monet's visual style throughout.",
                "A technician stands at a lab bench with three glass containers arranged by size. Depict the scene as a realistic photograph.",
            ),
            (
                "A green workbench holds a shallow tray of fasteners and a folded sheet of instructions. Interpret the scene through the visual language of Monet.",
                "A green workbench holds a shallow tray of fasteners and a folded sheet of instructions. Depict the scene as a realistic photograph.",
            ),
            (
                "A delivery worker carries a cardboard box past a row of sheltered vending machines. Render the scene in the visual style of Monet.",
                "A delivery worker carries a cardboard box past a row of sheltered vending machines. Depict the scene as a realistic photograph.",
            ),
            (
                "An attendant arranges a stack of towels at a reception desk beside a glass partition. Use Monet's visual style throughout.",
                "An attendant arranges a stack of towels at a reception desk beside a glass partition. Depict the scene as a realistic photograph.",
            ),
            (
                "A reading desk in an archive has a padded book cradle and a small square reading lamp. Interpret the scene through the visual language of Monet.",
                "A reading desk in an archive has a padded book cradle and a small square reading lamp. Depict the scene as a realistic photograph.",
            ),
            (
                "A bookstore worker rolls a short ladder past three shelves of upright books. Render the scene in the visual style of Monet.",
                "A bookstore worker rolls a short ladder past three shelves of upright books. Depict the scene as a realistic photograph.",
            ),
            (
                "A diner booth holds a stainless steel napkin holder, a water glass, and two paper menus. Use Monet's visual style throughout.",
                "A diner booth holds a stainless steel napkin holder, a water glass, and two paper menus. Depict the scene as a realistic photograph.",
            ),
            (
                "An empty concert hall surrounds a circular platform with a ring of tall wooden seats. Interpret the scene through the visual language of Monet.",
                "An empty concert hall surrounds a circular platform with a ring of tall wooden seats. Depict the scene as a realistic photograph.",
            ),
        ],
        "Picasso": [
            (
                "Two cooks stand at opposite ends of a bakery counter while rows of dough rest on trays. Render the scene in the visual style of Picasso.",
                "Two cooks stand at opposite ends of a bakery counter while rows of dough rest on trays. Depict the scene as a realistic photograph.",
            ),
            (
                "A freight elevator opens onto a storage floor with stacked archive boxes and a red fire door. Use Picasso's visual style throughout.",
                "A freight elevator opens onto a storage floor with stacked archive boxes and a red fire door. Depict the scene as a realistic photograph.",
            ),
            (
                "A pair of cyclists repairs a tire beside an orange locker on an indoor practice track. Interpret the scene through the visual language of Picasso.",
                "A pair of cyclists repairs a tire beside an orange locker on an indoor practice track. Depict the scene as a realistic photograph.",
            ),
            (
                "Inside a laundromat, a customer folds pale towels under a line of round dryer doors. Render the scene in the visual style of Picasso.",
                "Inside a laundromat, a customer folds pale towels under a line of round dryer doors. Depict the scene as a realistic photograph.",
            ),
            (
                "A tailor threads a needle beneath a floor lamp, between two mannequins and an open cabinet of fabric. Use Picasso's visual style throughout.",
                "A tailor threads a needle beneath a floor lamp, between two mannequins and an open cabinet of fabric. Depict the scene as a realistic photograph.",
            ),
            (
                "A brass bell and a roll of numbered tickets sit on a terracotta counter in a museum foyer. Interpret the scene through the visual language of Picasso.",
                "A brass bell and a roll of numbered tickets sit on a terracotta counter in a museum foyer. Depict the scene as a realistic photograph.",
            ),
            (
                "A station cleaner sweeps a tiled platform while the final carriage waits beyond the posts. Render the scene in the visual style of Picasso.",
                "A station cleaner sweeps a tiled platform while the final carriage waits beyond the posts. Depict the scene as a realistic photograph.",
            ),
            (
                "An archivist opens a document case beside a tall steel cart and a stack of cardboard folders. Use Picasso's visual style throughout.",
                "An archivist opens a document case beside a tall steel cart and a stack of cardboard folders. Depict the scene as a realistic photograph.",
            ),
            (
                "A narrow machine shop has two facing benches and a row of numbered drawers painted green. Interpret the scene through the visual language of Picasso.",
                "A narrow machine shop has two facing benches and a row of numbered drawers painted green. Depict the scene as a realistic photograph.",
            ),
            (
                "A sound editor sits at a desk with two small speakers and a shelf of labeled tape boxes. Render the scene in the visual style of Picasso.",
                "A sound editor sits at a desk with two small speakers and a shelf of labeled tape boxes. Depict the scene as a realistic photograph.",
            ),
            (
                "In an indoor pool, a person stands at the end of a lane divided by red and white floats. Use Picasso's visual style throughout.",
                "In an indoor pool, a person stands at the end of a lane divided by red and white floats. Depict the scene as a realistic photograph.",
            ),
            (
                "A kitchen rack holds stacked plates, a copper pan, and a folded apron beside a closed service window. Interpret the scene through the visual language of Picasso.",
                "A kitchen rack holds stacked plates, a copper pan, and a folded apron beside a closed service window. Depict the scene as a realistic photograph.",
            ),
            (
                "A young adult laces a roller skate on a wooden bench while another holds the door open. Render the scene in the visual style of Picasso.",
                "A young adult laces a roller skate on a wooden bench while another holds the door open. Depict the scene as a realistic photograph.",
            ),
            (
                "An empty cinema lobby has a low ticket booth and a short flight of stairs between dark curtains. Use Picasso's visual style throughout.",
                "An empty cinema lobby has a low ticket booth and a short flight of stairs between dark curtains. Depict the scene as a realistic photograph.",
            ),
            (
                "A row of subway turnstiles meets a tiled wall with a route diagram above it. Interpret the scene through the visual language of Picasso.",
                "A row of subway turnstiles meets a tiled wall with a route diagram above it. Depict the scene as a realistic photograph.",
            ),
            (
                "A mechanic checks a chain at a bicycle stand while a customer waits near a steel cabinet. Render the scene in the visual style of Picasso.",
                "A mechanic checks a chain at a bicycle stand while a customer waits near a steel cabinet. Depict the scene as a realistic photograph.",
            ),
            (
                "Three ceramic cups sit in a drying rack beside a stack of folded towels and a basin. Use Picasso's visual style throughout.",
                "Three ceramic cups sit in a drying rack beside a stack of folded towels and a basin. Depict the scene as a realistic photograph.",
            ),
            (
                "A student walks through a school corridor lined with upright lockers and white tiles. Interpret the scene through the visual language of Picasso.",
                "A student walks through a school corridor lined with upright lockers and white tiles. Depict the scene as a realistic photograph.",
            ),
            (
                "An empty rehearsal room contains two low stools and a wall of acoustic panels. Render the scene in the visual style of Picasso.",
                "An empty rehearsal room contains two low stools and a wall of acoustic panels. Depict the scene as a realistic photograph.",
            ),
            (
                "A square dining table holds a glass jug and two place settings beneath a white fabric shade. Use Picasso's visual style throughout.",
                "A square dining table holds a glass jug and two place settings beneath a white fabric shade. Depict the scene as a realistic photograph.",
            ),
            (
                "At an airport baggage carousel, a person waits beside a short stack of travel cases. Interpret the scene through the visual language of Picasso.",
                "At an airport baggage carousel, a person waits beside a short stack of travel cases. Depict the scene as a realistic photograph.",
            ),
            (
                "Two volunteers fold fabric banners in a sports hall with markings on the wooden floor. Render the scene in the visual style of Picasso.",
                "Two volunteers fold fabric banners in a sports hall with markings on the wooden floor. Depict the scene as a realistic photograph.",
            ),
            (
                "A technician stands at a lab bench with three glass containers arranged by size. Use Picasso's visual style throughout.",
                "A technician stands at a lab bench with three glass containers arranged by size. Depict the scene as a realistic photograph.",
            ),
            (
                "A green workbench holds a shallow tray of fasteners and a folded sheet of instructions. Interpret the scene through the visual language of Picasso.",
                "A green workbench holds a shallow tray of fasteners and a folded sheet of instructions. Depict the scene as a realistic photograph.",
            ),
            (
                "A delivery worker carries a cardboard box past a row of sheltered vending machines. Render the scene in the visual style of Picasso.",
                "A delivery worker carries a cardboard box past a row of sheltered vending machines. Depict the scene as a realistic photograph.",
            ),
            (
                "An attendant arranges a stack of towels at a reception desk beside a glass partition. Use Picasso's visual style throughout.",
                "An attendant arranges a stack of towels at a reception desk beside a glass partition. Depict the scene as a realistic photograph.",
            ),
            (
                "A reading desk in an archive has a padded book cradle and a small square reading lamp. Interpret the scene through the visual language of Picasso.",
                "A reading desk in an archive has a padded book cradle and a small square reading lamp. Depict the scene as a realistic photograph.",
            ),
            (
                "A bookstore worker rolls a short ladder past three shelves of upright books. Render the scene in the visual style of Picasso.",
                "A bookstore worker rolls a short ladder past three shelves of upright books. Depict the scene as a realistic photograph.",
            ),
            (
                "A diner booth holds a stainless steel napkin holder, a water glass, and two paper menus. Use Picasso's visual style throughout.",
                "A diner booth holds a stainless steel napkin holder, a water glass, and two paper menus. Depict the scene as a realistic photograph.",
            ),
            (
                "An empty concert hall surrounds a circular platform with a ring of tall wooden seats. Interpret the scene through the visual language of Picasso.",
                "An empty concert hall surrounds a circular platform with a ring of tall wooden seats. Depict the scene as a realistic photograph.",
            ),
        ],
    },
    "object": {
        "church": [
            (
                "A gravel footpath runs between rows of terraced vines. A church rises at the far end of the path. Use a wide composition with the surroundings visible on both sides.",
                "A gravel footpath runs between rows of terraced vines. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A gravel footpath runs between rows of terraced vines. A church rises at the far end of the path. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A gravel footpath runs between rows of terraced vines. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A gravel footpath runs between rows of terraced vines. A church rises at the far end of the path. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A gravel footpath runs between rows of terraced vines. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A gravel footpath runs between rows of terraced vines. A church rises at the far end of the path. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A gravel footpath runs between rows of terraced vines. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A gravel footpath runs between rows of terraced vines. A church rises at the far end of the path. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A gravel footpath runs between rows of terraced vines. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "Three houses and a line of power poles border a level canal bank. An old brick church stands beyond the last house. Use a wide composition with the surroundings visible on both sides.",
                "Three houses and a line of power poles border a level canal bank. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "Three houses and a line of power poles border a level canal bank. An old brick church stands beyond the last house. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "Three houses and a line of power poles border a level canal bank. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "Three houses and a line of power poles border a level canal bank. An old brick church stands beyond the last house. Compose a vertical frame that keeps both the near ground and the background visible.",
                "Three houses and a line of power poles border a level canal bank. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "Three houses and a line of power poles border a level canal bank. An old brick church stands beyond the last house. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "Three houses and a line of power poles border a level canal bank. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "Three houses and a line of power poles border a level canal bank. An old brick church stands beyond the last house. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "Three houses and a line of power poles border a level canal bank. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A short wooden bridge crosses a drainage ditch in a broad meadow. A small white church stands on the other side of the ditch. Use a wide composition with the surroundings visible on both sides.",
                "A short wooden bridge crosses a drainage ditch in a broad meadow. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A short wooden bridge crosses a drainage ditch in a broad meadow. A small white church stands on the other side of the ditch. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A short wooden bridge crosses a drainage ditch in a broad meadow. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A short wooden bridge crosses a drainage ditch in a broad meadow. A small white church stands on the other side of the ditch. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A short wooden bridge crosses a drainage ditch in a broad meadow. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A short wooden bridge crosses a drainage ditch in a broad meadow. A small white church stands on the other side of the ditch. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A short wooden bridge crosses a drainage ditch in a broad meadow. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A short wooden bridge crosses a drainage ditch in a broad meadow. A small white church stands on the other side of the ditch. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A short wooden bridge crosses a drainage ditch in a broad meadow. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "An industrial lane bends past a low warehouse and a loading ramp. A stone church occupies the open plot past the bend. Use a wide composition with the surroundings visible on both sides.",
                "An industrial lane bends past a low warehouse and a loading ramp. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "An industrial lane bends past a low warehouse and a loading ramp. A stone church occupies the open plot past the bend. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "An industrial lane bends past a low warehouse and a loading ramp. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "An industrial lane bends past a low warehouse and a loading ramp. A stone church occupies the open plot past the bend. Compose a vertical frame that keeps both the near ground and the background visible.",
                "An industrial lane bends past a low warehouse and a loading ramp. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "An industrial lane bends past a low warehouse and a loading ramp. A stone church occupies the open plot past the bend. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "An industrial lane bends past a low warehouse and a loading ramp. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "An industrial lane bends past a low warehouse and a loading ramp. A stone church occupies the open plot past the bend. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "An industrial lane bends past a low warehouse and a loading ramp. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A narrow access road curves between dry-stone walls and grazing land. A timber church stands near the top of the curve. Use a wide composition with the surroundings visible on both sides.",
                "A narrow access road curves between dry-stone walls and grazing land. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A narrow access road curves between dry-stone walls and grazing land. A timber church stands near the top of the curve. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A narrow access road curves between dry-stone walls and grazing land. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A narrow access road curves between dry-stone walls and grazing land. A timber church stands near the top of the curve. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A narrow access road curves between dry-stone walls and grazing land. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A narrow access road curves between dry-stone walls and grazing land. A timber church stands near the top of the curve. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A narrow access road curves between dry-stone walls and grazing land. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A narrow access road curves between dry-stone walls and grazing land. A timber church stands near the top of the curve. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A narrow access road curves between dry-stone walls and grazing land. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A ferry ramp opens onto an island quay lined with fishermen's huts. A modest church stands uphill from the huts. Use a wide composition with the surroundings visible on both sides.",
                "A ferry ramp opens onto an island quay lined with fishermen's huts. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A ferry ramp opens onto an island quay lined with fishermen's huts. A modest church stands uphill from the huts. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A ferry ramp opens onto an island quay lined with fishermen's huts. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A ferry ramp opens onto an island quay lined with fishermen's huts. A modest church stands uphill from the huts. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A ferry ramp opens onto an island quay lined with fishermen's huts. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A ferry ramp opens onto an island quay lined with fishermen's huts. A modest church stands uphill from the huts. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A ferry ramp opens onto an island quay lined with fishermen's huts. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A ferry ramp opens onto an island quay lined with fishermen's huts. A modest church stands uphill from the huts. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A ferry ramp opens onto an island quay lined with fishermen's huts. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "Fresh tire tracks cross the gravel apron of a rural bus stop. A brick church stands behind the bus stop. Use a wide composition with the surroundings visible on both sides.",
                "Fresh tire tracks cross the gravel apron of a rural bus stop. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "Fresh tire tracks cross the gravel apron of a rural bus stop. A brick church stands behind the bus stop. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "Fresh tire tracks cross the gravel apron of a rural bus stop. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "Fresh tire tracks cross the gravel apron of a rural bus stop. A brick church stands behind the bus stop. Compose a vertical frame that keeps both the near ground and the background visible.",
                "Fresh tire tracks cross the gravel apron of a rural bus stop. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "Fresh tire tracks cross the gravel apron of a rural bus stop. A brick church stands behind the bus stop. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "Fresh tire tracks cross the gravel apron of a rural bus stop. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "Fresh tire tracks cross the gravel apron of a rural bus stop. A brick church stands behind the bus stop. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "Fresh tire tracks cross the gravel apron of a rural bus stop. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A paved road passes between a chestnut grove and a low hedge. A church stands beyond a break in the hedge. Use a wide composition with the surroundings visible on both sides.",
                "A paved road passes between a chestnut grove and a low hedge. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A paved road passes between a chestnut grove and a low hedge. A church stands beyond a break in the hedge. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A paved road passes between a chestnut grove and a low hedge. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A paved road passes between a chestnut grove and a low hedge. A church stands beyond a break in the hedge. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A paved road passes between a chestnut grove and a low hedge. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A paved road passes between a chestnut grove and a low hedge. A church stands beyond a break in the hedge. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A paved road passes between a chestnut grove and a low hedge. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A paved road passes between a chestnut grove and a low hedge. A church stands beyond a break in the hedge. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A paved road passes between a chestnut grove and a low hedge. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A dockside street has pale warehouses on its left and rope bollards on its right. A small church stands at the inland end of the street. Use a wide composition with the surroundings visible on both sides.",
                "A dockside street has pale warehouses on its left and rope bollards on its right. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A dockside street has pale warehouses on its left and rope bollards on its right. A small church stands at the inland end of the street. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A dockside street has pale warehouses on its left and rope bollards on its right. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A dockside street has pale warehouses on its left and rope bollards on its right. A small church stands at the inland end of the street. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A dockside street has pale warehouses on its left and rope bollards on its right. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A dockside street has pale warehouses on its left and rope bollards on its right. A small church stands at the inland end of the street. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A dockside street has pale warehouses on its left and rope bollards on its right. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A dockside street has pale warehouses on its left and rope bollards on its right. A small church stands at the inland end of the street. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A dockside street has pale warehouses on its left and rope bollards on its right. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A hillside allotment has rectangular vegetable plots and a brick retaining wall. A church sits above the retaining wall. Use a wide composition with the surroundings visible on both sides.",
                "A hillside allotment has rectangular vegetable plots and a brick retaining wall. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A hillside allotment has rectangular vegetable plots and a brick retaining wall. A church sits above the retaining wall. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A hillside allotment has rectangular vegetable plots and a brick retaining wall. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A hillside allotment has rectangular vegetable plots and a brick retaining wall. A church sits above the retaining wall. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A hillside allotment has rectangular vegetable plots and a brick retaining wall. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A hillside allotment has rectangular vegetable plots and a brick retaining wall. A church sits above the retaining wall. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A hillside allotment has rectangular vegetable plots and a brick retaining wall. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A hillside allotment has rectangular vegetable plots and a brick retaining wall. A church sits above the retaining wall. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A hillside allotment has rectangular vegetable plots and a brick retaining wall. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "An unpaved lane divides a wheat stubble field from a row of poplars. A church is visible through a gap in the poplars. Use a wide composition with the surroundings visible on both sides.",
                "An unpaved lane divides a wheat stubble field from a row of poplars. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "An unpaved lane divides a wheat stubble field from a row of poplars. A church is visible through a gap in the poplars. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "An unpaved lane divides a wheat stubble field from a row of poplars. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "An unpaved lane divides a wheat stubble field from a row of poplars. A church is visible through a gap in the poplars. Compose a vertical frame that keeps both the near ground and the background visible.",
                "An unpaved lane divides a wheat stubble field from a row of poplars. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "An unpaved lane divides a wheat stubble field from a row of poplars. A church is visible through a gap in the poplars. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "An unpaved lane divides a wheat stubble field from a row of poplars. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "An unpaved lane divides a wheat stubble field from a row of poplars. A church is visible through a gap in the poplars. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "An unpaved lane divides a wheat stubble field from a row of poplars. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A row of stone cottages overlooks a paved square with a dry fountain. A church occupies one side of the square. Use a wide composition with the surroundings visible on both sides.",
                "A row of stone cottages overlooks a paved square with a dry fountain. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A row of stone cottages overlooks a paved square with a dry fountain. A church occupies one side of the square. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A row of stone cottages overlooks a paved square with a dry fountain. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A row of stone cottages overlooks a paved square with a dry fountain. A church occupies one side of the square. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A row of stone cottages overlooks a paved square with a dry fountain. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A row of stone cottages overlooks a paved square with a dry fountain. A church occupies one side of the square. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A row of stone cottages overlooks a paved square with a dry fountain. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A row of stone cottages overlooks a paved square with a dry fountain. A church occupies one side of the square. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A row of stone cottages overlooks a paved square with a dry fountain. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A farm road runs parallel to a row of white fence posts. A clapboard church stands across the road from the fence. Use a wide composition with the surroundings visible on both sides.",
                "A farm road runs parallel to a row of white fence posts. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A farm road runs parallel to a row of white fence posts. A clapboard church stands across the road from the fence. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A farm road runs parallel to a row of white fence posts. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A farm road runs parallel to a row of white fence posts. A clapboard church stands across the road from the fence. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A farm road runs parallel to a row of white fence posts. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A farm road runs parallel to a row of white fence posts. A clapboard church stands across the road from the fence. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A farm road runs parallel to a row of white fence posts. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A farm road runs parallel to a row of white fence posts. A clapboard church stands across the road from the fence. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A farm road runs parallel to a row of white fence posts. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A riverside embankment carries a cycle path past a long flood barrier. A church stands on higher ground behind the barrier. Use a wide composition with the surroundings visible on both sides.",
                "A riverside embankment carries a cycle path past a long flood barrier. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A riverside embankment carries a cycle path past a long flood barrier. A church stands on higher ground behind the barrier. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A riverside embankment carries a cycle path past a long flood barrier. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A riverside embankment carries a cycle path past a long flood barrier. A church stands on higher ground behind the barrier. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A riverside embankment carries a cycle path past a long flood barrier. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A riverside embankment carries a cycle path past a long flood barrier. A church stands on higher ground behind the barrier. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A riverside embankment carries a cycle path past a long flood barrier. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A riverside embankment carries a cycle path past a long flood barrier. A church stands on higher ground behind the barrier. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A riverside embankment carries a cycle path past a long flood barrier. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "Two stone gateposts mark the entrance to a lane through a valley settlement. A church stands opposite the entrance. Use a wide composition with the surroundings visible on both sides.",
                "Two stone gateposts mark the entrance to a lane through a valley settlement. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "Two stone gateposts mark the entrance to a lane through a valley settlement. A church stands opposite the entrance. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "Two stone gateposts mark the entrance to a lane through a valley settlement. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "Two stone gateposts mark the entrance to a lane through a valley settlement. A church stands opposite the entrance. Compose a vertical frame that keeps both the near ground and the background visible.",
                "Two stone gateposts mark the entrance to a lane through a valley settlement. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "Two stone gateposts mark the entrance to a lane through a valley settlement. A church stands opposite the entrance. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "Two stone gateposts mark the entrance to a lane through a valley settlement. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "Two stone gateposts mark the entrance to a lane through a valley settlement. A church stands opposite the entrance. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "Two stone gateposts mark the entrance to a lane through a valley settlement. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A hillside road overlooks a row of old clay-tile roofs. A church rises just beyond the roofs. Use a wide composition with the surroundings visible on both sides.",
                "A hillside road overlooks a row of old clay-tile roofs. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A hillside road overlooks a row of old clay-tile roofs. A church rises just beyond the roofs. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A hillside road overlooks a row of old clay-tile roofs. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A hillside road overlooks a row of old clay-tile roofs. A church rises just beyond the roofs. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A hillside road overlooks a row of old clay-tile roofs. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A hillside road overlooks a row of old clay-tile roofs. A church rises just beyond the roofs. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A hillside road overlooks a row of old clay-tile roofs. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A hillside road overlooks a row of old clay-tile roofs. A church rises just beyond the roofs. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A hillside road overlooks a row of old clay-tile roofs. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
        ],
        "golf ball": [
            (
                "A tray of narrow wooden compartments rests on a felt-covered workbench. A white golf ball sits in the compartment nearest the viewer. Use a wide composition with the surroundings visible on both sides.",
                "A tray of narrow wooden compartments rests on a felt-covered workbench. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A tray of narrow wooden compartments rests on a felt-covered workbench. A white golf ball sits in the compartment nearest the viewer. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A tray of narrow wooden compartments rests on a felt-covered workbench. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A tray of narrow wooden compartments rests on a felt-covered workbench. A white golf ball sits in the compartment nearest the viewer. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A tray of narrow wooden compartments rests on a felt-covered workbench. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A tray of narrow wooden compartments rests on a felt-covered workbench. A white golf ball sits in the compartment nearest the viewer. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A tray of narrow wooden compartments rests on a felt-covered workbench. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A tray of narrow wooden compartments rests on a felt-covered workbench. A white golf ball sits in the compartment nearest the viewer. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A tray of narrow wooden compartments rests on a felt-covered workbench. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A strip of artificial turf ends at an orange practice marker. A golf ball rests just short of the marker. Use a wide composition with the surroundings visible on both sides.",
                "A strip of artificial turf ends at an orange practice marker. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A strip of artificial turf ends at an orange practice marker. A golf ball rests just short of the marker. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A strip of artificial turf ends at an orange practice marker. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A strip of artificial turf ends at an orange practice marker. A golf ball rests just short of the marker. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A strip of artificial turf ends at an orange practice marker. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A strip of artificial turf ends at an orange practice marker. A golf ball rests just short of the marker. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A strip of artificial turf ends at an orange practice marker. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A strip of artificial turf ends at an orange practice marker. A golf ball rests just short of the marker. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A strip of artificial turf ends at an orange practice marker. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "Several hexagonal paving stones border a patch of neatly trimmed grass. A golf ball lies between the grass and the nearest stone. Use a wide composition with the surroundings visible on both sides.",
                "Several hexagonal paving stones border a patch of neatly trimmed grass. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "Several hexagonal paving stones border a patch of neatly trimmed grass. A golf ball lies between the grass and the nearest stone. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "Several hexagonal paving stones border a patch of neatly trimmed grass. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "Several hexagonal paving stones border a patch of neatly trimmed grass. A golf ball lies between the grass and the nearest stone. Compose a vertical frame that keeps both the near ground and the background visible.",
                "Several hexagonal paving stones border a patch of neatly trimmed grass. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "Several hexagonal paving stones border a patch of neatly trimmed grass. A golf ball lies between the grass and the nearest stone. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "Several hexagonal paving stones border a patch of neatly trimmed grass. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "Several hexagonal paving stones border a patch of neatly trimmed grass. A golf ball lies between the grass and the nearest stone. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "Several hexagonal paving stones border a patch of neatly trimmed grass. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A folding table holds a measuring tape and a folded scorecard. A golf ball sits on the table beside the scorecard. Use a wide composition with the surroundings visible on both sides.",
                "A folding table holds a measuring tape and a folded scorecard. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A folding table holds a measuring tape and a folded scorecard. A golf ball sits on the table beside the scorecard. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A folding table holds a measuring tape and a folded scorecard. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A folding table holds a measuring tape and a folded scorecard. A golf ball sits on the table beside the scorecard. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A folding table holds a measuring tape and a folded scorecard. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A folding table holds a measuring tape and a folded scorecard. A golf ball sits on the table beside the scorecard. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A folding table holds a measuring tape and a folded scorecard. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A folding table holds a measuring tape and a folded scorecard. A golf ball sits on the table beside the scorecard. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A folding table holds a measuring tape and a folded scorecard. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A short wooden practice ramp ends in a shallow rubber channel. A golf ball rests near the lip of the channel. Use a wide composition with the surroundings visible on both sides.",
                "A short wooden practice ramp ends in a shallow rubber channel. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A short wooden practice ramp ends in a shallow rubber channel. A golf ball rests near the lip of the channel. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A short wooden practice ramp ends in a shallow rubber channel. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A short wooden practice ramp ends in a shallow rubber channel. A golf ball rests near the lip of the channel. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A short wooden practice ramp ends in a shallow rubber channel. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A short wooden practice ramp ends in a shallow rubber channel. A golf ball rests near the lip of the channel. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A short wooden practice ramp ends in a shallow rubber channel. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A short wooden practice ramp ends in a shallow rubber channel. A golf ball rests near the lip of the channel. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A short wooden practice ramp ends in a shallow rubber channel. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A transport crate has shallow padded recesses along its upper edge. A yellow golf ball rests in one of the recesses. Use a wide composition with the surroundings visible on both sides.",
                "A transport crate has shallow padded recesses along its upper edge. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A transport crate has shallow padded recesses along its upper edge. A yellow golf ball rests in one of the recesses. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A transport crate has shallow padded recesses along its upper edge. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A transport crate has shallow padded recesses along its upper edge. A yellow golf ball rests in one of the recesses. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A transport crate has shallow padded recesses along its upper edge. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A transport crate has shallow padded recesses along its upper edge. A yellow golf ball rests in one of the recesses. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A transport crate has shallow padded recesses along its upper edge. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A transport crate has shallow padded recesses along its upper edge. A yellow golf ball rests in one of the recesses. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A transport crate has shallow padded recesses along its upper edge. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A park bench stands beside a crushed-gravel path and a low wooden rail. A golf ball lies on the path beneath the end of the bench. Use a wide composition with the surroundings visible on both sides.",
                "A park bench stands beside a crushed-gravel path and a low wooden rail. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A park bench stands beside a crushed-gravel path and a low wooden rail. A golf ball lies on the path beneath the end of the bench. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A park bench stands beside a crushed-gravel path and a low wooden rail. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A park bench stands beside a crushed-gravel path and a low wooden rail. A golf ball lies on the path beneath the end of the bench. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A park bench stands beside a crushed-gravel path and a low wooden rail. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A park bench stands beside a crushed-gravel path and a low wooden rail. A golf ball lies on the path beneath the end of the bench. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A park bench stands beside a crushed-gravel path and a low wooden rail. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A park bench stands beside a crushed-gravel path and a low wooden rail. A golf ball lies on the path beneath the end of the bench. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A park bench stands beside a crushed-gravel path and a low wooden rail. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A metal workbench holds a roll of paper and a narrow wooden ruler. A golf ball sits in the open space beside the ruler. Use a wide composition with the surroundings visible on both sides.",
                "A metal workbench holds a roll of paper and a narrow wooden ruler. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A metal workbench holds a roll of paper and a narrow wooden ruler. A golf ball sits in the open space beside the ruler. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A metal workbench holds a roll of paper and a narrow wooden ruler. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A metal workbench holds a roll of paper and a narrow wooden ruler. A golf ball sits in the open space beside the ruler. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A metal workbench holds a roll of paper and a narrow wooden ruler. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A metal workbench holds a roll of paper and a narrow wooden ruler. A golf ball sits in the open space beside the ruler. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A metal workbench holds a roll of paper and a narrow wooden ruler. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A metal workbench holds a roll of paper and a narrow wooden ruler. A golf ball sits in the open space beside the ruler. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A metal workbench holds a roll of paper and a narrow wooden ruler. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "Short grass gives way to a sandy patch with shallow rake marks. A partly sunken golf ball lies near the upper edge of the sand. Use a wide composition with the surroundings visible on both sides.",
                "Short grass gives way to a sandy patch with shallow rake marks. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "Short grass gives way to a sandy patch with shallow rake marks. A partly sunken golf ball lies near the upper edge of the sand. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "Short grass gives way to a sandy patch with shallow rake marks. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "Short grass gives way to a sandy patch with shallow rake marks. A partly sunken golf ball lies near the upper edge of the sand. Compose a vertical frame that keeps both the near ground and the background visible.",
                "Short grass gives way to a sandy patch with shallow rake marks. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "Short grass gives way to a sandy patch with shallow rake marks. A partly sunken golf ball lies near the upper edge of the sand. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "Short grass gives way to a sandy patch with shallow rake marks. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "Short grass gives way to a sandy patch with shallow rake marks. A partly sunken golf ball lies near the upper edge of the sand. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "Short grass gives way to a sandy patch with shallow rake marks. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A practice mat is rolled halfway across a folded canvas sheet. A golf ball sits on the exposed corner of the canvas. Use a wide composition with the surroundings visible on both sides.",
                "A practice mat is rolled halfway across a folded canvas sheet. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A practice mat is rolled halfway across a folded canvas sheet. A golf ball sits on the exposed corner of the canvas. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A practice mat is rolled halfway across a folded canvas sheet. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A practice mat is rolled halfway across a folded canvas sheet. A golf ball sits on the exposed corner of the canvas. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A practice mat is rolled halfway across a folded canvas sheet. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A practice mat is rolled halfway across a folded canvas sheet. A golf ball sits on the exposed corner of the canvas. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A practice mat is rolled halfway across a folded canvas sheet. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A practice mat is rolled halfway across a folded canvas sheet. A golf ball sits on the exposed corner of the canvas. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A practice mat is rolled halfway across a folded canvas sheet. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A row of low plastic boundary markers divides a level training area. A golf ball rests midway between the first two markers. Use a wide composition with the surroundings visible on both sides.",
                "A row of low plastic boundary markers divides a level training area. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A row of low plastic boundary markers divides a level training area. A golf ball rests midway between the first two markers. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A row of low plastic boundary markers divides a level training area. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A row of low plastic boundary markers divides a level training area. A golf ball rests midway between the first two markers. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A row of low plastic boundary markers divides a level training area. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A row of low plastic boundary markers divides a level training area. A golf ball rests midway between the first two markers. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A row of low plastic boundary markers divides a level training area. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A row of low plastic boundary markers divides a level training area. A golf ball rests midway between the first two markers. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A row of low plastic boundary markers divides a level training area. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A row of shallow numbered compartments runs along a counter. A golf ball sits in the compartment marked seven. Use a wide composition with the surroundings visible on both sides.",
                "A row of shallow numbered compartments runs along a counter. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A row of shallow numbered compartments runs along a counter. A golf ball sits in the compartment marked seven. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A row of shallow numbered compartments runs along a counter. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A row of shallow numbered compartments runs along a counter. A golf ball sits in the compartment marked seven. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A row of shallow numbered compartments runs along a counter. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A row of shallow numbered compartments runs along a counter. A golf ball sits in the compartment marked seven. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A row of shallow numbered compartments runs along a counter. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A row of shallow numbered compartments runs along a counter. A golf ball sits in the compartment marked seven. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A row of shallow numbered compartments runs along a counter. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A narrow painted line runs across the floor of an indoor practice hall. A red golf ball sits just past the painted line. Use a wide composition with the surroundings visible on both sides.",
                "A narrow painted line runs across the floor of an indoor practice hall. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A narrow painted line runs across the floor of an indoor practice hall. A red golf ball sits just past the painted line. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A narrow painted line runs across the floor of an indoor practice hall. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A narrow painted line runs across the floor of an indoor practice hall. A red golf ball sits just past the painted line. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A narrow painted line runs across the floor of an indoor practice hall. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A narrow painted line runs across the floor of an indoor practice hall. A red golf ball sits just past the painted line. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A narrow painted line runs across the floor of an indoor practice hall. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A narrow painted line runs across the floor of an indoor practice hall. A red golf ball sits just past the painted line. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A narrow painted line runs across the floor of an indoor practice hall. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A clipboard rests next to a strip of foam on a long equipment shelf. A golf ball sits in front of the clipboard. Use a wide composition with the surroundings visible on both sides.",
                "A clipboard rests next to a strip of foam on a long equipment shelf. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A clipboard rests next to a strip of foam on a long equipment shelf. A golf ball sits in front of the clipboard. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A clipboard rests next to a strip of foam on a long equipment shelf. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A clipboard rests next to a strip of foam on a long equipment shelf. A golf ball sits in front of the clipboard. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A clipboard rests next to a strip of foam on a long equipment shelf. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A clipboard rests next to a strip of foam on a long equipment shelf. A golf ball sits in front of the clipboard. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A clipboard rests next to a strip of foam on a long equipment shelf. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A clipboard rests next to a strip of foam on a long equipment shelf. A golf ball sits in front of the clipboard. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A clipboard rests next to a strip of foam on a long equipment shelf. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A broad stretch of lawn ends where a shallow drainage grate crosses the turf. A golf ball sits near the last opening in the grate. Use a wide composition with the surroundings visible on both sides.",
                "A broad stretch of lawn ends where a shallow drainage grate crosses the turf. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A broad stretch of lawn ends where a shallow drainage grate crosses the turf. A golf ball sits near the last opening in the grate. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A broad stretch of lawn ends where a shallow drainage grate crosses the turf. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A broad stretch of lawn ends where a shallow drainage grate crosses the turf. A golf ball sits near the last opening in the grate. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A broad stretch of lawn ends where a shallow drainage grate crosses the turf. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A broad stretch of lawn ends where a shallow drainage grate crosses the turf. A golf ball sits near the last opening in the grate. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A broad stretch of lawn ends where a shallow drainage grate crosses the turf. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A broad stretch of lawn ends where a shallow drainage grate crosses the turf. A golf ball sits near the last opening in the grate. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A broad stretch of lawn ends where a shallow drainage grate crosses the turf. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A bare concrete threshold meets a section of rubber flooring. A golf ball rests against the step in the threshold. Use a wide composition with the surroundings visible on both sides.",
                "A bare concrete threshold meets a section of rubber flooring. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A bare concrete threshold meets a section of rubber flooring. A golf ball rests against the step in the threshold. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A bare concrete threshold meets a section of rubber flooring. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A bare concrete threshold meets a section of rubber flooring. A golf ball rests against the step in the threshold. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A bare concrete threshold meets a section of rubber flooring. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A bare concrete threshold meets a section of rubber flooring. A golf ball rests against the step in the threshold. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A bare concrete threshold meets a section of rubber flooring. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A bare concrete threshold meets a section of rubber flooring. A golf ball rests against the step in the threshold. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A bare concrete threshold meets a section of rubber flooring. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
        ],
        "parachute": [
            (
                "A mown airfield has white boundary markers and a low service shed. An open parachute descends over the far side of the airfield. Use a wide composition with the surroundings visible on both sides.",
                "A mown airfield has white boundary markers and a low service shed. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A mown airfield has white boundary markers and a low service shed. An open parachute descends over the far side of the airfield. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A mown airfield has white boundary markers and a low service shed. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A mown airfield has white boundary markers and a low service shed. An open parachute descends over the far side of the airfield. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A mown airfield has white boundary markers and a low service shed. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A mown airfield has white boundary markers and a low service shed. An open parachute descends over the far side of the airfield. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A mown airfield has white boundary markers and a low service shed. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A mown airfield has white boundary markers and a low service shed. An open parachute descends over the far side of the airfield. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A mown airfield has white boundary markers and a low service shed. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A broad salt flat stretches toward a dark ridge with a track across the foreground. An orange parachute is suspended above the ridge line. Use a wide composition with the surroundings visible on both sides.",
                "A broad salt flat stretches toward a dark ridge with a track across the foreground. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A broad salt flat stretches toward a dark ridge with a track across the foreground. An orange parachute is suspended above the ridge line. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A broad salt flat stretches toward a dark ridge with a track across the foreground. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A broad salt flat stretches toward a dark ridge with a track across the foreground. An orange parachute is suspended above the ridge line. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A broad salt flat stretches toward a dark ridge with a track across the foreground. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A broad salt flat stretches toward a dark ridge with a track across the foreground. An orange parachute is suspended above the ridge line. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A broad salt flat stretches toward a dark ridge with a track across the foreground. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A broad salt flat stretches toward a dark ridge with a track across the foreground. An orange parachute is suspended above the ridge line. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A broad salt flat stretches toward a dark ridge with a track across the foreground. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A broad strip of stubble extends between a gravel lane and a windbreak. A collapsed parachute lies on the stubble just beyond the lane. Use a wide composition with the surroundings visible on both sides.",
                "A broad strip of stubble extends between a gravel lane and a windbreak. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A broad strip of stubble extends between a gravel lane and a windbreak. A collapsed parachute lies on the stubble just beyond the lane. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A broad strip of stubble extends between a gravel lane and a windbreak. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A broad strip of stubble extends between a gravel lane and a windbreak. A collapsed parachute lies on the stubble just beyond the lane. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A broad strip of stubble extends between a gravel lane and a windbreak. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A broad strip of stubble extends between a gravel lane and a windbreak. A collapsed parachute lies on the stubble just beyond the lane. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A broad strip of stubble extends between a gravel lane and a windbreak. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A broad strip of stubble extends between a gravel lane and a windbreak. A collapsed parachute lies on the stubble just beyond the lane. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A broad strip of stubble extends between a gravel lane and a windbreak. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A hangar floor is marked with yellow lines beside a half-open door. A folded parachute lies just inside the marked area. Use a wide composition with the surroundings visible on both sides.",
                "A hangar floor is marked with yellow lines beside a half-open door. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A hangar floor is marked with yellow lines beside a half-open door. A folded parachute lies just inside the marked area. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A hangar floor is marked with yellow lines beside a half-open door. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A hangar floor is marked with yellow lines beside a half-open door. A folded parachute lies just inside the marked area. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A hangar floor is marked with yellow lines beside a half-open door. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A hangar floor is marked with yellow lines beside a half-open door. A folded parachute lies just inside the marked area. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A hangar floor is marked with yellow lines beside a half-open door. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A hangar floor is marked with yellow lines beside a half-open door. A folded parachute lies just inside the marked area. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A hangar floor is marked with yellow lines beside a half-open door. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A training yard has a tall practice frame and a row of low padded barriers. A parachute is spread flat beside the barriers. Use a wide composition with the surroundings visible on both sides.",
                "A training yard has a tall practice frame and a row of low padded barriers. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A training yard has a tall practice frame and a row of low padded barriers. A parachute is spread flat beside the barriers. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A training yard has a tall practice frame and a row of low padded barriers. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A training yard has a tall practice frame and a row of low padded barriers. A parachute is spread flat beside the barriers. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A training yard has a tall practice frame and a row of low padded barriers. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A training yard has a tall practice frame and a row of low padded barriers. A parachute is spread flat beside the barriers. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A training yard has a tall practice frame and a row of low padded barriers. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A training yard has a tall practice frame and a row of low padded barriers. A parachute is spread flat beside the barriers. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A training yard has a tall practice frame and a row of low padded barriers. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A footpath bisects a heath of low, scrubby plants. A tangled parachute lies across low brush to one side of the path. Use a wide composition with the surroundings visible on both sides.",
                "A footpath bisects a heath of low, scrubby plants. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A footpath bisects a heath of low, scrubby plants. A tangled parachute lies across low brush to one side of the path. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A footpath bisects a heath of low, scrubby plants. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A footpath bisects a heath of low, scrubby plants. A tangled parachute lies across low brush to one side of the path. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A footpath bisects a heath of low, scrubby plants. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A footpath bisects a heath of low, scrubby plants. A tangled parachute lies across low brush to one side of the path. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A footpath bisects a heath of low, scrubby plants. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A footpath bisects a heath of low, scrubby plants. A tangled parachute lies across low brush to one side of the path. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A footpath bisects a heath of low, scrubby plants. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A warehouse has a wide packing table, a concrete floor, and a row of steel cabinets. A partly folded parachute covers the center of the packing table. Use a wide composition with the surroundings visible on both sides.",
                "A warehouse has a wide packing table, a concrete floor, and a row of steel cabinets. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A warehouse has a wide packing table, a concrete floor, and a row of steel cabinets. A partly folded parachute covers the center of the packing table. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A warehouse has a wide packing table, a concrete floor, and a row of steel cabinets. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A warehouse has a wide packing table, a concrete floor, and a row of steel cabinets. A partly folded parachute covers the center of the packing table. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A warehouse has a wide packing table, a concrete floor, and a row of steel cabinets. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A warehouse has a wide packing table, a concrete floor, and a row of steel cabinets. A partly folded parachute covers the center of the packing table. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A warehouse has a wide packing table, a concrete floor, and a row of steel cabinets. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A warehouse has a wide packing table, a concrete floor, and a row of steel cabinets. A partly folded parachute covers the center of the packing table. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A warehouse has a wide packing table, a concrete floor, and a row of steel cabinets. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A gray loading bay has wheel stops and a painted equipment lane. A fabric parachute rests just inside the painted lane. Use a wide composition with the surroundings visible on both sides.",
                "A gray loading bay has wheel stops and a painted equipment lane. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A gray loading bay has wheel stops and a painted equipment lane. A fabric parachute rests just inside the painted lane. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A gray loading bay has wheel stops and a painted equipment lane. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A gray loading bay has wheel stops and a painted equipment lane. A fabric parachute rests just inside the painted lane. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A gray loading bay has wheel stops and a painted equipment lane. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A gray loading bay has wheel stops and a painted equipment lane. A fabric parachute rests just inside the painted lane. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A gray loading bay has wheel stops and a painted equipment lane. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A gray loading bay has wheel stops and a painted equipment lane. A fabric parachute rests just inside the painted lane. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A gray loading bay has wheel stops and a painted equipment lane. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "An empty runway has numbered distance markers and a dark band of tarmac in the foreground. A partially inflated parachute hovers above the grass beside the runway. Use a wide composition with the surroundings visible on both sides.",
                "An empty runway has numbered distance markers and a dark band of tarmac in the foreground. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "An empty runway has numbered distance markers and a dark band of tarmac in the foreground. A partially inflated parachute hovers above the grass beside the runway. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "An empty runway has numbered distance markers and a dark band of tarmac in the foreground. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "An empty runway has numbered distance markers and a dark band of tarmac in the foreground. A partially inflated parachute hovers above the grass beside the runway. Compose a vertical frame that keeps both the near ground and the background visible.",
                "An empty runway has numbered distance markers and a dark band of tarmac in the foreground. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "An empty runway has numbered distance markers and a dark band of tarmac in the foreground. A partially inflated parachute hovers above the grass beside the runway. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "An empty runway has numbered distance markers and a dark band of tarmac in the foreground. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "An empty runway has numbered distance markers and a dark band of tarmac in the foreground. A partially inflated parachute hovers above the grass beside the runway. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "An empty runway has numbered distance markers and a dark band of tarmac in the foreground. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A reservoir shore slopes up toward a line of dark conifers. An open parachute drifts above the shore and in front of the trees. Use a wide composition with the surroundings visible on both sides.",
                "A reservoir shore slopes up toward a line of dark conifers. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A reservoir shore slopes up toward a line of dark conifers. An open parachute drifts above the shore and in front of the trees. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A reservoir shore slopes up toward a line of dark conifers. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A reservoir shore slopes up toward a line of dark conifers. An open parachute drifts above the shore and in front of the trees. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A reservoir shore slopes up toward a line of dark conifers. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A reservoir shore slopes up toward a line of dark conifers. An open parachute drifts above the shore and in front of the trees. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A reservoir shore slopes up toward a line of dark conifers. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A reservoir shore slopes up toward a line of dark conifers. An open parachute drifts above the shore and in front of the trees. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A reservoir shore slopes up toward a line of dark conifers. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A dry riverbed has a bend of rounded stones and broad banks of unvegetated gravel. A red parachute lies on the gravel at the inside of the bend. Use a wide composition with the surroundings visible on both sides.",
                "A dry riverbed has a bend of rounded stones and broad banks of unvegetated gravel. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A dry riverbed has a bend of rounded stones and broad banks of unvegetated gravel. A red parachute lies on the gravel at the inside of the bend. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A dry riverbed has a bend of rounded stones and broad banks of unvegetated gravel. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A dry riverbed has a bend of rounded stones and broad banks of unvegetated gravel. A red parachute lies on the gravel at the inside of the bend. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A dry riverbed has a bend of rounded stones and broad banks of unvegetated gravel. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A dry riverbed has a bend of rounded stones and broad banks of unvegetated gravel. A red parachute lies on the gravel at the inside of the bend. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A dry riverbed has a bend of rounded stones and broad banks of unvegetated gravel. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A dry riverbed has a bend of rounded stones and broad banks of unvegetated gravel. A red parachute lies on the gravel at the inside of the bend. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A dry riverbed has a bend of rounded stones and broad banks of unvegetated gravel. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A field camp has a folding trestle and a square vinyl groundsheet. A white parachute is spread across one corner of the groundsheet. Use a wide composition with the surroundings visible on both sides.",
                "A field camp has a folding trestle and a square vinyl groundsheet. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A field camp has a folding trestle and a square vinyl groundsheet. A white parachute is spread across one corner of the groundsheet. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A field camp has a folding trestle and a square vinyl groundsheet. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A field camp has a folding trestle and a square vinyl groundsheet. A white parachute is spread across one corner of the groundsheet. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A field camp has a folding trestle and a square vinyl groundsheet. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A field camp has a folding trestle and a square vinyl groundsheet. A white parachute is spread across one corner of the groundsheet. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A field camp has a folding trestle and a square vinyl groundsheet. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A field camp has a folding trestle and a square vinyl groundsheet. A white parachute is spread across one corner of the groundsheet. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A field camp has a folding trestle and a square vinyl groundsheet. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A paved test yard contains two tall towers and markings for wind direction. An unfurled parachute hangs between the towers. Use a wide composition with the surroundings visible on both sides.",
                "A paved test yard contains two tall towers and markings for wind direction. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A paved test yard contains two tall towers and markings for wind direction. An unfurled parachute hangs between the towers. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A paved test yard contains two tall towers and markings for wind direction. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A paved test yard contains two tall towers and markings for wind direction. An unfurled parachute hangs between the towers. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A paved test yard contains two tall towers and markings for wind direction. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A paved test yard contains two tall towers and markings for wind direction. An unfurled parachute hangs between the towers. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A paved test yard contains two tall towers and markings for wind direction. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A paved test yard contains two tall towers and markings for wind direction. An unfurled parachute hangs between the towers. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A paved test yard contains two tall towers and markings for wind direction. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A gravel pullout sits beneath the switchbacks of a bare hillside. A collapsed parachute lies on the far side of the pullout. Use a wide composition with the surroundings visible on both sides.",
                "A gravel pullout sits beneath the switchbacks of a bare hillside. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A gravel pullout sits beneath the switchbacks of a bare hillside. A collapsed parachute lies on the far side of the pullout. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A gravel pullout sits beneath the switchbacks of a bare hillside. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A gravel pullout sits beneath the switchbacks of a bare hillside. A collapsed parachute lies on the far side of the pullout. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A gravel pullout sits beneath the switchbacks of a bare hillside. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A gravel pullout sits beneath the switchbacks of a bare hillside. A collapsed parachute lies on the far side of the pullout. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A gravel pullout sits beneath the switchbacks of a bare hillside. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A gravel pullout sits beneath the switchbacks of a bare hillside. A collapsed parachute lies on the far side of the pullout. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A gravel pullout sits beneath the switchbacks of a bare hillside. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A grassy clearing is bounded by a white fence and an empty flagpole. A parachute trails low above the clearing. Use a wide composition with the surroundings visible on both sides.",
                "A grassy clearing is bounded by a white fence and an empty flagpole. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A grassy clearing is bounded by a white fence and an empty flagpole. A parachute trails low above the clearing. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A grassy clearing is bounded by a white fence and an empty flagpole. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A grassy clearing is bounded by a white fence and an empty flagpole. A parachute trails low above the clearing. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A grassy clearing is bounded by a white fence and an empty flagpole. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A grassy clearing is bounded by a white fence and an empty flagpole. A parachute trails low above the clearing. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A grassy clearing is bounded by a white fence and an empty flagpole. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A grassy clearing is bounded by a white fence and an empty flagpole. A parachute trails low above the clearing. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A grassy clearing is bounded by a white fence and an empty flagpole. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A workshop bay contains a low worktable and a row of safety harnesses on hooks. A neatly gathered parachute rests on the worktable. Use a wide composition with the surroundings visible on both sides.",
                "A workshop bay contains a low worktable and a row of safety harnesses on hooks. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A workshop bay contains a low worktable and a row of safety harnesses on hooks. A neatly gathered parachute rests on the worktable. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A workshop bay contains a low worktable and a row of safety harnesses on hooks. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A workshop bay contains a low worktable and a row of safety harnesses on hooks. A neatly gathered parachute rests on the worktable. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A workshop bay contains a low worktable and a row of safety harnesses on hooks. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A workshop bay contains a low worktable and a row of safety harnesses on hooks. A neatly gathered parachute rests on the worktable. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A workshop bay contains a low worktable and a row of safety harnesses on hooks. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A workshop bay contains a low worktable and a row of safety harnesses on hooks. A neatly gathered parachute rests on the worktable. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A workshop bay contains a low worktable and a row of safety harnesses on hooks. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
        ],
        "cassette player": [
            (
                "A painted kitchen windowsill holds a folded recipe and a ceramic saucer. A cassette player rests on the sill beside the recipe. Use a wide composition with the surroundings visible on both sides.",
                "A painted kitchen windowsill holds a folded recipe and a ceramic saucer. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A painted kitchen windowsill holds a folded recipe and a ceramic saucer. A cassette player rests on the sill beside the recipe. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A painted kitchen windowsill holds a folded recipe and a ceramic saucer. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A painted kitchen windowsill holds a folded recipe and a ceramic saucer. A cassette player rests on the sill beside the recipe. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A painted kitchen windowsill holds a folded recipe and a ceramic saucer. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A painted kitchen windowsill holds a folded recipe and a ceramic saucer. A cassette player rests on the sill beside the recipe. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A painted kitchen windowsill holds a folded recipe and a ceramic saucer. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A painted kitchen windowsill holds a folded recipe and a ceramic saucer. A cassette player rests on the sill beside the recipe. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A painted kitchen windowsill holds a folded recipe and a ceramic saucer. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A worktable at a repair shop has labeled drawers and a cork mat. A cassette player sits on the cork mat. Use a wide composition with the surroundings visible on both sides.",
                "A worktable at a repair shop has labeled drawers and a cork mat. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A worktable at a repair shop has labeled drawers and a cork mat. A cassette player sits on the cork mat. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A worktable at a repair shop has labeled drawers and a cork mat. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A worktable at a repair shop has labeled drawers and a cork mat. A cassette player sits on the cork mat. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A worktable at a repair shop has labeled drawers and a cork mat. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A worktable at a repair shop has labeled drawers and a cork mat. A cassette player sits on the cork mat. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A worktable at a repair shop has labeled drawers and a cork mat. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A worktable at a repair shop has labeled drawers and a cork mat. A cassette player sits on the cork mat. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A worktable at a repair shop has labeled drawers and a cork mat. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A mostly empty metal locker has a canvas strap looped over an upper hook. A portable cassette player rests on the lowest shelf. Use a wide composition with the surroundings visible on both sides.",
                "A mostly empty metal locker has a canvas strap looped over an upper hook. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A mostly empty metal locker has a canvas strap looped over an upper hook. A portable cassette player rests on the lowest shelf. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A mostly empty metal locker has a canvas strap looped over an upper hook. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A mostly empty metal locker has a canvas strap looped over an upper hook. A portable cassette player rests on the lowest shelf. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A mostly empty metal locker has a canvas strap looped over an upper hook. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A mostly empty metal locker has a canvas strap looped over an upper hook. A portable cassette player rests on the lowest shelf. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A mostly empty metal locker has a canvas strap looped over an upper hook. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A mostly empty metal locker has a canvas strap looped over an upper hook. A portable cassette player rests on the lowest shelf. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A mostly empty metal locker has a canvas strap looped over an upper hook. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A training room contains a wooden lectern and two rows of folding tables. A cassette player sits on the small table beside the lectern. Use a wide composition with the surroundings visible on both sides.",
                "A training room contains a wooden lectern and two rows of folding tables. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A training room contains a wooden lectern and two rows of folding tables. A cassette player sits on the small table beside the lectern. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A training room contains a wooden lectern and two rows of folding tables. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A training room contains a wooden lectern and two rows of folding tables. A cassette player sits on the small table beside the lectern. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A training room contains a wooden lectern and two rows of folding tables. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A training room contains a wooden lectern and two rows of folding tables. A cassette player sits on the small table beside the lectern. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A training room contains a wooden lectern and two rows of folding tables. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A training room contains a wooden lectern and two rows of folding tables. A cassette player sits on the small table beside the lectern. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A training room contains a wooden lectern and two rows of folding tables. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A flea market stall has clean gray fabric laid over a folding table. A cassette player occupies the corner nearest the aisle. Use a wide composition with the surroundings visible on both sides.",
                "A flea market stall has clean gray fabric laid over a folding table. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A flea market stall has clean gray fabric laid over a folding table. A cassette player occupies the corner nearest the aisle. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A flea market stall has clean gray fabric laid over a folding table. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A flea market stall has clean gray fabric laid over a folding table. A cassette player occupies the corner nearest the aisle. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A flea market stall has clean gray fabric laid over a folding table. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A flea market stall has clean gray fabric laid over a folding table. A cassette player occupies the corner nearest the aisle. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A flea market stall has clean gray fabric laid over a folding table. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A flea market stall has clean gray fabric laid over a folding table. A cassette player occupies the corner nearest the aisle. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A flea market stall has clean gray fabric laid over a folding table. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A former ticket booth has a narrow glass ledge and a faded paper schedule. A cassette player sits on the ledge in front of the schedule. Use a wide composition with the surroundings visible on both sides.",
                "A former ticket booth has a narrow glass ledge and a faded paper schedule. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A former ticket booth has a narrow glass ledge and a faded paper schedule. A cassette player sits on the ledge in front of the schedule. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A former ticket booth has a narrow glass ledge and a faded paper schedule. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A former ticket booth has a narrow glass ledge and a faded paper schedule. A cassette player sits on the ledge in front of the schedule. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A former ticket booth has a narrow glass ledge and a faded paper schedule. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A former ticket booth has a narrow glass ledge and a faded paper schedule. A cassette player sits on the ledge in front of the schedule. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A former ticket booth has a narrow glass ledge and a faded paper schedule. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A former ticket booth has a narrow glass ledge and a faded paper schedule. A cassette player sits on the ledge in front of the schedule. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A former ticket booth has a narrow glass ledge and a faded paper schedule. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A desk in a listening room has a logbook and a coil of wire. A cassette player rests beside the logbook. Use a wide composition with the surroundings visible on both sides.",
                "A desk in a listening room has a logbook and a coil of wire. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A desk in a listening room has a logbook and a coil of wire. A cassette player rests beside the logbook. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A desk in a listening room has a logbook and a coil of wire. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A desk in a listening room has a logbook and a coil of wire. A cassette player rests beside the logbook. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A desk in a listening room has a logbook and a coil of wire. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A desk in a listening room has a logbook and a coil of wire. A cassette player rests beside the logbook. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A desk in a listening room has a logbook and a coil of wire. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A desk in a listening room has a logbook and a coil of wire. A cassette player rests beside the logbook. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A desk in a listening room has a logbook and a coil of wire. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A plane of honey-colored wood forms the top of a desk by a window. A cassette player sits at the far end of the desk. Use a wide composition with the surroundings visible on both sides.",
                "A plane of honey-colored wood forms the top of a desk by a window. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A plane of honey-colored wood forms the top of a desk by a window. A cassette player sits at the far end of the desk. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A plane of honey-colored wood forms the top of a desk by a window. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A plane of honey-colored wood forms the top of a desk by a window. A cassette player sits at the far end of the desk. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A plane of honey-colored wood forms the top of a desk by a window. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A plane of honey-colored wood forms the top of a desk by a window. A cassette player sits at the far end of the desk. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A plane of honey-colored wood forms the top of a desk by a window. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A plane of honey-colored wood forms the top of a desk by a window. A cassette player sits at the far end of the desk. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A plane of honey-colored wood forms the top of a desk by a window. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A changing room has a row of wooden benches beneath brass hooks. A cassette player sits on the end of the nearest bench. Use a wide composition with the surroundings visible on both sides.",
                "A changing room has a row of wooden benches beneath brass hooks. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A changing room has a row of wooden benches beneath brass hooks. A cassette player sits on the end of the nearest bench. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A changing room has a row of wooden benches beneath brass hooks. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A changing room has a row of wooden benches beneath brass hooks. A cassette player sits on the end of the nearest bench. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A changing room has a row of wooden benches beneath brass hooks. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A changing room has a row of wooden benches beneath brass hooks. A cassette player sits on the end of the nearest bench. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A changing room has a row of wooden benches beneath brass hooks. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A changing room has a row of wooden benches beneath brass hooks. A cassette player sits on the end of the nearest bench. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A changing room has a row of wooden benches beneath brass hooks. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A library archive has a narrow counter and rows of plain transfer boxes. A cassette player sits in front of the first box. Use a wide composition with the surroundings visible on both sides.",
                "A library archive has a narrow counter and rows of plain transfer boxes. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A library archive has a narrow counter and rows of plain transfer boxes. A cassette player sits in front of the first box. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A library archive has a narrow counter and rows of plain transfer boxes. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A library archive has a narrow counter and rows of plain transfer boxes. A cassette player sits in front of the first box. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A library archive has a narrow counter and rows of plain transfer boxes. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A library archive has a narrow counter and rows of plain transfer boxes. A cassette player sits in front of the first box. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A library archive has a narrow counter and rows of plain transfer boxes. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A library archive has a narrow counter and rows of plain transfer boxes. A cassette player sits in front of the first box. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A library archive has a narrow counter and rows of plain transfer boxes. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A garage shelf holds a stack of folded cloths and a container of spare screws. A cassette player rests between the cloths and the container. Use a wide composition with the surroundings visible on both sides.",
                "A garage shelf holds a stack of folded cloths and a container of spare screws. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A garage shelf holds a stack of folded cloths and a container of spare screws. A cassette player rests between the cloths and the container. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A garage shelf holds a stack of folded cloths and a container of spare screws. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A garage shelf holds a stack of folded cloths and a container of spare screws. A cassette player rests between the cloths and the container. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A garage shelf holds a stack of folded cloths and a container of spare screws. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A garage shelf holds a stack of folded cloths and a container of spare screws. A cassette player rests between the cloths and the container. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A garage shelf holds a stack of folded cloths and a container of spare screws. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A garage shelf holds a stack of folded cloths and a container of spare screws. A cassette player rests between the cloths and the container. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A garage shelf holds a stack of folded cloths and a container of spare screws. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A display cabinet has a low plywood riser and a sheet of gray backing paper. A cassette player rests on the riser. Use a wide composition with the surroundings visible on both sides.",
                "A display cabinet has a low plywood riser and a sheet of gray backing paper. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A display cabinet has a low plywood riser and a sheet of gray backing paper. A cassette player rests on the riser. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A display cabinet has a low plywood riser and a sheet of gray backing paper. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A display cabinet has a low plywood riser and a sheet of gray backing paper. A cassette player rests on the riser. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A display cabinet has a low plywood riser and a sheet of gray backing paper. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A display cabinet has a low plywood riser and a sheet of gray backing paper. A cassette player rests on the riser. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A display cabinet has a low plywood riser and a sheet of gray backing paper. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A display cabinet has a low plywood riser and a sheet of gray backing paper. A cassette player rests on the riser. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A display cabinet has a low plywood riser and a sheet of gray backing paper. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A lounge has a low side table, a woven floor mat, and a fabric sofa. A cassette player sits on the side table. Use a wide composition with the surroundings visible on both sides.",
                "A lounge has a low side table, a woven floor mat, and a fabric sofa. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A lounge has a low side table, a woven floor mat, and a fabric sofa. A cassette player sits on the side table. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A lounge has a low side table, a woven floor mat, and a fabric sofa. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A lounge has a low side table, a woven floor mat, and a fabric sofa. A cassette player sits on the side table. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A lounge has a low side table, a woven floor mat, and a fabric sofa. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A lounge has a low side table, a woven floor mat, and a fabric sofa. A cassette player sits on the side table. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A lounge has a low side table, a woven floor mat, and a fabric sofa. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A lounge has a low side table, a woven floor mat, and a fabric sofa. A cassette player sits on the side table. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A lounge has a low side table, a woven floor mat, and a fabric sofa. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A break-room shelf sits above a wall of pale green tiles. A cassette player is set near the far end of the shelf. Use a wide composition with the surroundings visible on both sides.",
                "A break-room shelf sits above a wall of pale green tiles. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A break-room shelf sits above a wall of pale green tiles. A cassette player is set near the far end of the shelf. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A break-room shelf sits above a wall of pale green tiles. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A break-room shelf sits above a wall of pale green tiles. A cassette player is set near the far end of the shelf. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A break-room shelf sits above a wall of pale green tiles. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A break-room shelf sits above a wall of pale green tiles. A cassette player is set near the far end of the shelf. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A break-room shelf sits above a wall of pale green tiles. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A break-room shelf sits above a wall of pale green tiles. A cassette player is set near the far end of the shelf. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A break-room shelf sits above a wall of pale green tiles. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "An open storage drawer has blue dividers and a narrow compartment on the right. A compact cassette player occupies that compartment. Use a wide composition with the surroundings visible on both sides.",
                "An open storage drawer has blue dividers and a narrow compartment on the right. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "An open storage drawer has blue dividers and a narrow compartment on the right. A compact cassette player occupies that compartment. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "An open storage drawer has blue dividers and a narrow compartment on the right. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "An open storage drawer has blue dividers and a narrow compartment on the right. A compact cassette player occupies that compartment. Compose a vertical frame that keeps both the near ground and the background visible.",
                "An open storage drawer has blue dividers and a narrow compartment on the right. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "An open storage drawer has blue dividers and a narrow compartment on the right. A compact cassette player occupies that compartment. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "An open storage drawer has blue dividers and a narrow compartment on the right. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "An open storage drawer has blue dividers and a narrow compartment on the right. A compact cassette player occupies that compartment. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "An open storage drawer has blue dividers and a narrow compartment on the right. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A small equipment booth has a plywood shelf beneath an observation window. A cassette player rests on the shelf beneath the window. Use a wide composition with the surroundings visible on both sides.",
                "A small equipment booth has a plywood shelf beneath an observation window. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A small equipment booth has a plywood shelf beneath an observation window. A cassette player rests on the shelf beneath the window. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A small equipment booth has a plywood shelf beneath an observation window. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A small equipment booth has a plywood shelf beneath an observation window. A cassette player rests on the shelf beneath the window. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A small equipment booth has a plywood shelf beneath an observation window. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A small equipment booth has a plywood shelf beneath an observation window. A cassette player rests on the shelf beneath the window. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A small equipment booth has a plywood shelf beneath an observation window. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A small equipment booth has a plywood shelf beneath an observation window. A cassette player rests on the shelf beneath the window. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A small equipment booth has a plywood shelf beneath an observation window. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
        ],
        "banana": [
            (
                "A walnut table holds a white ceramic bowl and a folded blue napkin. A banana lies on the table next to the bowl. Use a wide composition with the surroundings visible on both sides.",
                "A walnut table holds a white ceramic bowl and a folded blue napkin. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A walnut table holds a white ceramic bowl and a folded blue napkin. A banana lies on the table next to the bowl. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A walnut table holds a white ceramic bowl and a folded blue napkin. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A walnut table holds a white ceramic bowl and a folded blue napkin. A banana lies on the table next to the bowl. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A walnut table holds a white ceramic bowl and a folded blue napkin. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A walnut table holds a white ceramic bowl and a folded blue napkin. A banana lies on the table next to the bowl. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A walnut table holds a white ceramic bowl and a folded blue napkin. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A walnut table holds a white ceramic bowl and a folded blue napkin. A banana lies on the table next to the bowl. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A walnut table holds a white ceramic bowl and a folded blue napkin. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A grocer's counter has shallow green trays and a roll of brown paper. A banana lies across the front of one tray. Use a wide composition with the surroundings visible on both sides.",
                "A grocer's counter has shallow green trays and a roll of brown paper. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A grocer's counter has shallow green trays and a roll of brown paper. A banana lies across the front of one tray. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A grocer's counter has shallow green trays and a roll of brown paper. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A grocer's counter has shallow green trays and a roll of brown paper. A banana lies across the front of one tray. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A grocer's counter has shallow green trays and a roll of brown paper. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A grocer's counter has shallow green trays and a roll of brown paper. A banana lies across the front of one tray. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A grocer's counter has shallow green trays and a roll of brown paper. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A grocer's counter has shallow green trays and a roll of brown paper. A banana lies across the front of one tray. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A grocer's counter has shallow green trays and a roll of brown paper. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A metal lunch tray sits on a bench beside a closed lunch box. A banana rests in the large compartment of the tray. Use a wide composition with the surroundings visible on both sides.",
                "A metal lunch tray sits on a bench beside a closed lunch box. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A metal lunch tray sits on a bench beside a closed lunch box. A banana rests in the large compartment of the tray. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A metal lunch tray sits on a bench beside a closed lunch box. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A metal lunch tray sits on a bench beside a closed lunch box. A banana rests in the large compartment of the tray. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A metal lunch tray sits on a bench beside a closed lunch box. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A metal lunch tray sits on a bench beside a closed lunch box. A banana rests in the large compartment of the tray. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A metal lunch tray sits on a bench beside a closed lunch box. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A metal lunch tray sits on a bench beside a closed lunch box. A banana rests in the large compartment of the tray. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A metal lunch tray sits on a bench beside a closed lunch box. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A cutting board and a closed jar sit on pale butcher-block counters. A banana lies along the near edge of the cutting board. Use a wide composition with the surroundings visible on both sides.",
                "A cutting board and a closed jar sit on pale butcher-block counters. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A cutting board and a closed jar sit on pale butcher-block counters. A banana lies along the near edge of the cutting board. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A cutting board and a closed jar sit on pale butcher-block counters. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A cutting board and a closed jar sit on pale butcher-block counters. A banana lies along the near edge of the cutting board. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A cutting board and a closed jar sit on pale butcher-block counters. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A cutting board and a closed jar sit on pale butcher-block counters. A banana lies along the near edge of the cutting board. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A cutting board and a closed jar sit on pale butcher-block counters. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A cutting board and a closed jar sit on pale butcher-block counters. A banana lies along the near edge of the cutting board. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A cutting board and a closed jar sit on pale butcher-block counters. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A paper shopping bag lies open near a set of house keys on a sideboard. A banana sits across the mouth of the bag. Use a wide composition with the surroundings visible on both sides.",
                "A paper shopping bag lies open near a set of house keys on a sideboard. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A paper shopping bag lies open near a set of house keys on a sideboard. A banana sits across the mouth of the bag. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A paper shopping bag lies open near a set of house keys on a sideboard. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A paper shopping bag lies open near a set of house keys on a sideboard. A banana sits across the mouth of the bag. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A paper shopping bag lies open near a set of house keys on a sideboard. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A paper shopping bag lies open near a set of house keys on a sideboard. A banana sits across the mouth of the bag. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A paper shopping bag lies open near a set of house keys on a sideboard. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A paper shopping bag lies open near a set of house keys on a sideboard. A banana sits across the mouth of the bag. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A paper shopping bag lies open near a set of house keys on a sideboard. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A dark slate plate sits on a table made of alternating light and dark wood strips. A single banana rests at the edge of the plate. Use a wide composition with the surroundings visible on both sides.",
                "A dark slate plate sits on a table made of alternating light and dark wood strips. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A dark slate plate sits on a table made of alternating light and dark wood strips. A single banana rests at the edge of the plate. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A dark slate plate sits on a table made of alternating light and dark wood strips. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A dark slate plate sits on a table made of alternating light and dark wood strips. A single banana rests at the edge of the plate. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A dark slate plate sits on a table made of alternating light and dark wood strips. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A dark slate plate sits on a table made of alternating light and dark wood strips. A single banana rests at the edge of the plate. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A dark slate plate sits on a table made of alternating light and dark wood strips. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A dark slate plate sits on a table made of alternating light and dark wood strips. A single banana rests at the edge of the plate. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A dark slate plate sits on a table made of alternating light and dark wood strips. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A checked picnic cloth is spread across the level top of a stone wall. A banana lies near the front corner of the cloth. Use a wide composition with the surroundings visible on both sides.",
                "A checked picnic cloth is spread across the level top of a stone wall. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A checked picnic cloth is spread across the level top of a stone wall. A banana lies near the front corner of the cloth. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A checked picnic cloth is spread across the level top of a stone wall. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A checked picnic cloth is spread across the level top of a stone wall. A banana lies near the front corner of the cloth. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A checked picnic cloth is spread across the level top of a stone wall. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A checked picnic cloth is spread across the level top of a stone wall. A banana lies near the front corner of the cloth. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A checked picnic cloth is spread across the level top of a stone wall. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A checked picnic cloth is spread across the level top of a stone wall. A banana lies near the front corner of the cloth. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A checked picnic cloth is spread across the level top of a stone wall. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A broad produce scale stands on a grocer's work surface next to a stack of paper bags. A banana rests on the weighing tray. Use a wide composition with the surroundings visible on both sides.",
                "A broad produce scale stands on a grocer's work surface next to a stack of paper bags. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A broad produce scale stands on a grocer's work surface next to a stack of paper bags. A banana rests on the weighing tray. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A broad produce scale stands on a grocer's work surface next to a stack of paper bags. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A broad produce scale stands on a grocer's work surface next to a stack of paper bags. A banana rests on the weighing tray. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A broad produce scale stands on a grocer's work surface next to a stack of paper bags. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A broad produce scale stands on a grocer's work surface next to a stack of paper bags. A banana rests on the weighing tray. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A broad produce scale stands on a grocer's work surface next to a stack of paper bags. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A broad produce scale stands on a grocer's work surface next to a stack of paper bags. A banana rests on the weighing tray. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A broad produce scale stands on a grocer's work surface next to a stack of paper bags. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A kitchen shelf holds a woven basket and three neatly stacked cups. A banana rests in the basket. Use a wide composition with the surroundings visible on both sides.",
                "A kitchen shelf holds a woven basket and three neatly stacked cups. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A kitchen shelf holds a woven basket and three neatly stacked cups. A banana rests in the basket. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A kitchen shelf holds a woven basket and three neatly stacked cups. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A kitchen shelf holds a woven basket and three neatly stacked cups. A banana rests in the basket. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A kitchen shelf holds a woven basket and three neatly stacked cups. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A kitchen shelf holds a woven basket and three neatly stacked cups. A banana rests in the basket. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A kitchen shelf holds a woven basket and three neatly stacked cups. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A kitchen shelf holds a woven basket and three neatly stacked cups. A banana rests in the basket. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A kitchen shelf holds a woven basket and three neatly stacked cups. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A narrow countertop has a stainless steel backsplash and a pale porcelain spoon rest. A ripe banana lies beside the spoon rest. Use a wide composition with the surroundings visible on both sides.",
                "A narrow countertop has a stainless steel backsplash and a pale porcelain spoon rest. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A narrow countertop has a stainless steel backsplash and a pale porcelain spoon rest. A ripe banana lies beside the spoon rest. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A narrow countertop has a stainless steel backsplash and a pale porcelain spoon rest. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A narrow countertop has a stainless steel backsplash and a pale porcelain spoon rest. A ripe banana lies beside the spoon rest. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A narrow countertop has a stainless steel backsplash and a pale porcelain spoon rest. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A narrow countertop has a stainless steel backsplash and a pale porcelain spoon rest. A ripe banana lies beside the spoon rest. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A narrow countertop has a stainless steel backsplash and a pale porcelain spoon rest. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A narrow countertop has a stainless steel backsplash and a pale porcelain spoon rest. A ripe banana lies beside the spoon rest. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A narrow countertop has a stainless steel backsplash and a pale porcelain spoon rest. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A plain wooden stool sits beside an empty coat stand in a tiled entryway. A banana rests on the seat of the stool. Use a wide composition with the surroundings visible on both sides.",
                "A plain wooden stool sits beside an empty coat stand in a tiled entryway. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A plain wooden stool sits beside an empty coat stand in a tiled entryway. A banana rests on the seat of the stool. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A plain wooden stool sits beside an empty coat stand in a tiled entryway. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A plain wooden stool sits beside an empty coat stand in a tiled entryway. A banana rests on the seat of the stool. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A plain wooden stool sits beside an empty coat stand in a tiled entryway. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A plain wooden stool sits beside an empty coat stand in a tiled entryway. A banana rests on the seat of the stool. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A plain wooden stool sits beside an empty coat stand in a tiled entryway. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A plain wooden stool sits beside an empty coat stand in a tiled entryway. A banana rests on the seat of the stool. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A plain wooden stool sits beside an empty coat stand in a tiled entryway. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A breakfast bar has a linen placemat and a glass jug of water. A banana sits on the placemat beside the jug. Use a wide composition with the surroundings visible on both sides.",
                "A breakfast bar has a linen placemat and a glass jug of water. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A breakfast bar has a linen placemat and a glass jug of water. A banana sits on the placemat beside the jug. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A breakfast bar has a linen placemat and a glass jug of water. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A breakfast bar has a linen placemat and a glass jug of water. A banana sits on the placemat beside the jug. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A breakfast bar has a linen placemat and a glass jug of water. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A breakfast bar has a linen placemat and a glass jug of water. A banana sits on the placemat beside the jug. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A breakfast bar has a linen placemat and a glass jug of water. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A breakfast bar has a linen placemat and a glass jug of water. A banana sits on the placemat beside the jug. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A breakfast bar has a linen placemat and a glass jug of water. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A glass-fronted market case has a row of shallow white serving dishes. An unpeeled banana rests in the dish nearest the glass. Use a wide composition with the surroundings visible on both sides.",
                "A glass-fronted market case has a row of shallow white serving dishes. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A glass-fronted market case has a row of shallow white serving dishes. An unpeeled banana rests in the dish nearest the glass. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A glass-fronted market case has a row of shallow white serving dishes. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A glass-fronted market case has a row of shallow white serving dishes. An unpeeled banana rests in the dish nearest the glass. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A glass-fronted market case has a row of shallow white serving dishes. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A glass-fronted market case has a row of shallow white serving dishes. An unpeeled banana rests in the dish nearest the glass. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A glass-fronted market case has a row of shallow white serving dishes. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A glass-fronted market case has a row of shallow white serving dishes. An unpeeled banana rests in the dish nearest the glass. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A glass-fronted market case has a row of shallow white serving dishes. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A striped tablecloth is spread beneath an empty wire fruit rack. A banana lies on the cloth beside the rack. Use a wide composition with the surroundings visible on both sides.",
                "A striped tablecloth is spread beneath an empty wire fruit rack. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A striped tablecloth is spread beneath an empty wire fruit rack. A banana lies on the cloth beside the rack. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A striped tablecloth is spread beneath an empty wire fruit rack. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A striped tablecloth is spread beneath an empty wire fruit rack. A banana lies on the cloth beside the rack. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A striped tablecloth is spread beneath an empty wire fruit rack. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A striped tablecloth is spread beneath an empty wire fruit rack. A banana lies on the cloth beside the rack. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A striped tablecloth is spread beneath an empty wire fruit rack. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A striped tablecloth is spread beneath an empty wire fruit rack. A banana lies on the cloth beside the rack. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A striped tablecloth is spread beneath an empty wire fruit rack. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "A square of parchment lies on a dark kitchen counter beside an oven mitt. A banana rests in the center of the parchment. Use a wide composition with the surroundings visible on both sides.",
                "A square of parchment lies on a dark kitchen counter beside an oven mitt. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "A square of parchment lies on a dark kitchen counter beside an oven mitt. A banana rests in the center of the parchment. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "A square of parchment lies on a dark kitchen counter beside an oven mitt. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "A square of parchment lies on a dark kitchen counter beside an oven mitt. A banana rests in the center of the parchment. Compose a vertical frame that keeps both the near ground and the background visible.",
                "A square of parchment lies on a dark kitchen counter beside an oven mitt. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "A square of parchment lies on a dark kitchen counter beside an oven mitt. A banana rests in the center of the parchment. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "A square of parchment lies on a dark kitchen counter beside an oven mitt. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "A square of parchment lies on a dark kitchen counter beside an oven mitt. A banana rests in the center of the parchment. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "A square of parchment lies on a dark kitchen counter beside an oven mitt. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
            (
                "An open wooden produce crate lies on a tiled floor beside a low loading sill. A banana lies just inside the opening of the crate. Use a wide composition with the surroundings visible on both sides.",
                "An open wooden produce crate lies on a tiled floor beside a low loading sill. Use a wide composition with the surroundings visible on both sides.",
            ),
            (
                "An open wooden produce crate lies on a tiled floor beside a low loading sill. A banana lies just inside the opening of the crate. Use soft, even illumination and retain visible texture on the nearby surfaces.",
                "An open wooden produce crate lies on a tiled floor beside a low loading sill. Use soft, even illumination and retain visible texture on the nearby surfaces.",
            ),
            (
                "An open wooden produce crate lies on a tiled floor beside a low loading sill. A banana lies just inside the opening of the crate. Compose a vertical frame that keeps both the near ground and the background visible.",
                "An open wooden produce crate lies on a tiled floor beside a low loading sill. Compose a vertical frame that keeps both the near ground and the background visible.",
            ),
            (
                "An open wooden produce crate lies on a tiled floor beside a low loading sill. A banana lies just inside the opening of the crate. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
                "An open wooden produce crate lies on a tiled floor beside a low loading sill. Use an off-center viewpoint so that the foreground lines lead diagonally into the frame.",
            ),
            (
                "An open wooden produce crate lies on a tiled floor beside a low loading sill. A banana lies just inside the opening of the crate. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
                "An open wooden produce crate lies on a tiled floor beside a low loading sill. Keep the main surfaces and background in focus, with natural colors and restrained contrast.",
            ),
        ],
    },
}
