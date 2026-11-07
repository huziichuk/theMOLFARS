# ml branch segmentation output

`objects.json` and `transform.json` are copied unchanged from `origin/ml` (commit b82ca99, `segment.py` + `prepare_scene.py` by Maks6666; Open3D: floor alignment, RANSAC wall/ceiling removal, DBSCAN, yaw boxes).
`pipeline/build_scene.py --boxes inputs/ml_segment` adds their boxes that do not overlap ours. Regenerate with the ml branch scripts if the scan changes.
