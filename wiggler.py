import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.color
import sys

# Extra weight given to the L channel in Lab colorspace difference
LUMA_WEIGHT = 2

# Keep only TOP_K probabilities so that ugly colors should never be picked
TOP_K = 3

# Increase this to add more dithering
NOISE_SPREAD = 2

palette_path = "palette.png"
palette_img = Image.open(palette_path)

path = "images/tokyo1993_small.png"

if len(sys.argv) > 1:
    path = sys.argv[1]

input_img = Image.open(path)

# Load images and convert them to first to [0,1]^3 normalized sRGB and then to Lab.

palette_rgb_int = np.array(palette_img)[:, :, :3]
img_rgb_int = np.array(input_img)[:, :, :3]

palette_rgb = palette_rgb_int.copy().astype(float).reshape(-1, 3)
num_colors = palette_rgb.shape[0]

palette_rgb /= 255
img_rgb = img_rgb_int.astype(float) / 255

palette = skimage.color.rgb2lab(palette_rgb)
img = skimage.color.rgb2lab(img_rgb)

assert palette.shape[1] == img.shape[2]
shape = img.shape[:2]

# Compute per-pixel distance for each palette color.

distances = np.zeros((*shape, num_colors))  # Will be [width x height x 16]
weight_mask = np.array([[[LUMA_WEIGHT,1,1]]])

for i in range(num_colors):
    diff = img - palette[i]
    diff *= weight_mask
    diff = diff ** 2
    dist = np.sqrt(np.sum(diff, axis=2))
    distances[:, :, i] = dist


distances[distances == 0] = 1e-6

# We need some function to convert from euclidean Lab distances to unnormalized probabilities ("scores")
# and one way to do it is e^-x. See https://www.desmos.com/calculator/wtf0mprdeq
def distance2score(dist):
    return np.exp(-dist / NOISE_SPREAD)

scores = distance2score(distances)

# Only keep the TOP_K highest scores in the array to avoid ugly outlier pixels.

for i, j in np.ndindex(shape):
    inds = np.argsort(scores[i,j])
    worst = inds[:-TOP_K]
    scores[i,j][worst] = 0

score_sum = np.sum(scores, axis=2).reshape(*scores.shape[:2], 1)

# Normalize the scores into probabilities.
probas = scores / score_sum

rng = np.random.default_rng(seed=0)

nearest_choices = np.zeros(shape, np.int32)
random_choices = np.zeros(shape, np.int32)

total = shape[0] * shape[1]
pixels_processed = 0

# Pick a palette index for each pixel.
# We get two integer arrays 'nearest_choices' and 'random_choices' which
# are then mapped into actual colors below.

for i, j in np.ndindex(nearest_choices.shape):
    ps = probas[i,j]
    nearest_choices[i,j] = np.argmax(ps)
    random_choices[i,j] = rng.choice(num_colors, p=ps)

    # TODO Never pick an alternative color if the largest probability is very large?

    if pixels_processed % 1000 == 0:
        print(f"{pixels_processed/total*100} %")

    pixels_processed += 1

# Map palette indices into colors
nearest = palette_rgb[nearest_choices]
random = palette_rgb[random_choices]

# Save the dithered image

print('Saving result.png')
Image.fromarray(np.clip(random*255, 0, 255).astype(np.uint8)).save('result.png')

# Plot the results

fig, ax = plt.subplots(1,1)
xs = np.linspace(0.0, distances.max(), 100)
ax.plot(xs, distance2score(xs))
ax.set_xlabel('Distance')
ax.set_ylabel('Score')
fig.suptitle(f"LAB color distance to score mapping with spread={NOISE_SPREAD}")

fig, ax = plt.subplots(4,4,figsize=(10,13))
fig.suptitle('Palette color areas')

for i, a in enumerate(ax.flatten()):
    a.axis('off')
    a.imshow(probas[:, :, i])
    color = palette_rgb[i]
    rect = patches.Rectangle((0, 0), 16, 16, linewidth=1, edgecolor='black', facecolor=color, fill=True)
    a.add_patch(rect)

plt.tight_layout()

fig, ax = plt.subplots(1,3, figsize=(9,6))
for a in ax:
    a.axis('off')

ax[0].imshow(img_rgb)
ax[1].imshow(nearest)
ax[2].imshow(random)

ax[0].set_title('Input')
ax[1].set_title('Nearest color')
ax[2].set_title(f"Dithered color (K={TOP_K}, spread={NOISE_SPREAD})")

plt.show()

