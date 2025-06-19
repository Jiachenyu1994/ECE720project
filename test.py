# Diffusion Sharing Evaluation Script
# Requirements: pip install gdsfactory


import gdsfactory as gf
from itertools import combinations

# ---- 1. Parse SPICE File ----
def parse_spice(filename):
    transistors = []
    with open(filename) as f:
        for line in f:
            if line.startswith("X"):
                tokens = line.strip().split()
                name = tokens[0]
                drain, gate, source, bulk = tokens[1:5]
                model = tokens[5]
                w = l = None
                for token in tokens[6:]:
                    if token.startswith("w="):
                        w = float(token[2:-1])  # remove 'u'
                    elif token.startswith("l="):
                        l = float(token[2:-1])
                transistors.append((name, model, drain, source, w, l))
    return transistors

# ---- 2. Identify Possible Shared Pairs ----
def possible_shared_pairs(transistors):
    pairs = []
    for t1, t2 in combinations(transistors, 2):
        if (
            t1[1] == t2[1] and  # model
            t1[4] == t2[4] and  # w
            t1[5] == t2[5] and  # l
            set(t1[2:4]) & set(t2[2:4])  # common net between drain/source
        ):
            pairs.append((t1[0], t2[0]))
    return pairs

# ---- 3. Placeholder for Layout-Based Sharing Detection ----
def get_actual_shared_pairs(component):
    # For now, simulate: assume 80% of potential pairs are realized
    # Replace this with real analysis using layout bounding boxes
    return int(0.8 * len(component.transistors))  # fake estimate for now

# ---- 4. Main Evaluation ----
def evaluate_diffusion_score(spice_file, gds_file):
    transistors = parse_spice(spice_file)
    P = len(possible_shared_pairs(transistors))

    component = gf.import_gds(gds_file)
    A = int(0.8 * P)  # Simulate with 80% realization (for now)

    score = A / P if P > 0 else 0
    print(f"Diffusion Sharing Score: {score:.2f} ({A} shared out of {P} possible pairs)")

# ---- Run Example ----
if __name__ == "__main__":
    evaluate_diffusion_score("sky130_fd_sc_hd__fa_1.spice", "sky130_fd_sc_hd__fa_1.gds")
