"""Zero-shot video label + per-category 'excitement' prompts for SigLIP. Pure text definitions and
numpy maths; the text embeddings are computed once at feature-extraction time."""
import numpy as np

P = "a first-person GoPro video of "
CLASS_PROMPTS = {
    "bike": [P + "riding a bicycle", P + "mountain biking down a trail", "POV of a cyclist's handlebars on a ride"],
    "bikepolo": [P + "playing bike polo", "POV of hitting a ball with a mallet while riding a bicycle on a court"],
    "boxing": [P + "boxing sparring", "POV of a boxer throwing punches at an opponent in a ring"],
    "horse": [P + "horse riding", "POV from a rider on horseback, the horse's head and mane in front"],
    "jumping": [P + "base jumping or skydiving", P + "jumping off a cliff or bridge", "POV of a person in free fall"],
    "longboarding": [P + "longboarding downhill on a road", "POV of a longboard rider carving down a street"],
    "motor": [P + "riding a motorcycle", "POV from a motorbike rider with the handlebars and dashboard visible"],
    "parkour": [P + "parkour and freerunning", "POV of running and jumping across rooftops and walls"],
    "plane": [P + "flying an airplane", "POV from an aircraft cockpit or wing in flight"],
    "rockclimbing": [P + "rock climbing", "POV of a climber's hands gripping holds on a rock face"],
    "scuba": [P + "scuba diving underwater", "underwater POV of a diver with fish and coral"],
    "skate": [P + "skateboarding", "POV of a skateboarder doing tricks in a skate park"],
    "ski": [P + "skiing down a snowy slope", "POV of a skier with ski poles on snow"],
    "surfing": [P + "surfing a wave", "POV from a surfboard riding an ocean wave"],
    "cooking": [P + "cooking in a kitchen", "POV of hands preparing food at a kitchen counter or washing dishes"],
}
CLASSES = list(CLASS_PROMPTS)
SPORTS = [c for c in CLASSES if c != "cooking"]
_NAME = {"bike": "cycling", "bikepolo": "bike polo", "boxing": "boxing", "horse": "horse riding", "jumping": "jumping",
         "longboarding": "longboarding", "motor": "motorcycle riding", "parkour": "parkour", "plane": "flying",
         "rockclimbing": "rock climbing", "scuba": "scuba diving", "skate": "skateboarding", "ski": "skiing",
         "surfing": "surfing"}


def excitement_prompts(cat):
    x = _NAME[cat]
    pos = [f"the most exciting moment of a {x} video", f"a spectacular {x} trick, stunt or fast action",
           f"a thrilling highlight of {x}"]
    neg = ["a boring uneventful moment where nothing happens", f"standing around waiting or getting ready before {x}",
           "a static shot of the ground or equipment"]
    return pos, neg


def classify(label_logits):
    """label_logits: [frames x len(CLASSES)] (already scaled). Returns (name, score, probs)."""
    z = np.asarray(label_logits, np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    probs = p.mean(axis=0)
    k = int(np.argmax(probs))
    return CLASSES[k], float(probs[k]), probs
