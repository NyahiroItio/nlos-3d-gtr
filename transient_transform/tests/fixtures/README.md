# Frozen MATLAB numerical reference

`author_matlab_reference.npz` contains two deterministic 4×4×16 reference
calculations exported from MATLAB R2024b Update 7. Both are produced by the
original 2018 LCT arithmetic, with retroreflective and diffuse radiometric
branches. They are numerical test outputs, not redistributed author datasets
or source code.

Original author source SHA-256:
`824275cb1fbb5c33751781b7b7498d7c793a92dc07a456670ec169e38ea6b1f6`.

Fixture SHA-256 at generation:
`1f83d0c9376da278d38b7ead2f36b16c18637041e2a8b22bc41817d396572854`.

Regenerate using `python -m transient_transform.reference`, run its generated
MATLAB script, then `python -m transient_transform.reference --export-fixtures`.
See `../../ORIGINAL_ALIGNMENT.md` for source provenance and the single
MATLAB variable-declaration compatibility fix.
