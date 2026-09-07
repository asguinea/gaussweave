# Data and model policy

GaussWeave does not redistribute datasets, source images, pretrained Gaussian
representations, generated renders, checkpoints, or other third-party artifacts.
Those paths are ignored by Git by default.

Some real-scene helpers recognize the Truck scene distributed through the upstream
3D Gaussian Splatting dataset bundle, which incorporates Tanks and Temples data.
The code records expected archive identities, camera conventions, and attribution
metadata for validation; it does not grant rights to the underlying files.

Before acquiring or using external material:

1. Visit the authoritative upstream source.
2. Review and comply with its current license, attribution, and use restrictions.
3. Verify that your intended use is permitted.
4. Store the material only under the ignored `datasets/` directory.
5. Never commit source media, derived renders, weights, or credentials.

The currently recognized archive endpoint is recorded in
[`src/gaussweave/data/real_truck.py`](src/gaussweave/data/real_truck.py). Its
metadata is an integrity aid, not a license statement. If the upstream archive or
terms change, re-evaluate the workflow before use.

Synthetic configurations in `configs/scenes/` contain parameters and metadata
only. Generated outputs are also excluded from version control.
