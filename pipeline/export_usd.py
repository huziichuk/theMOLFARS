"""Export the semantic layer (3D frames, Excel metadata, routes) as OpenUSD text (.usda): Builder Question Q-09,
"portable across web, OpenUSD/Omniverse, Unity and Unreal". No USD library needed; Omniverse, usdview, Blender,
and the Unity/Unreal USD importers read it.

    python pipeline/export_usd.py web/scene.json out.usda

Y-up, metres (the pipeline's world frame). Every frame is a unit Cube scaled to the box, rotated by its PCA yaw,
coloured by status, with valmet:* attributes carrying the object ID, vision label and the mapped Excel asset.
"""
import json
import math
import sys
from pathlib import Path

STATUS_RGB = {"target": (1.0, 0.3, 0.3), "warn": (0.95, 0.71, 0.21), "healthy": (0.24, 0.81, 0.43), "none": (0.55, 0.58, 0.64)}


def s(v) -> str:
    """A USD string literal (same escaping as JSON)."""
    return json.dumps(str(v), ensure_ascii=False)


def prim_name(oid: str) -> str:
    return oid.replace("-", "_")


def scene_to_usda(scene: dict) -> str:
    out = ['#usda 1.0', '(', '    defaultPrim = "BoilerRoom"', '    metersPerUnit = 1', '    upAxis = "Y"',
           f'    doc = {s("theMOLFARS semantic layer, scene_id " + scene.get("scene_id", ""))}', ')', '',
           'def Xform "BoilerRoom" (', '    kind = "assembly"', ')', '{', '    def Scope "Objects"', '    {']
    for o in scene["objects"]:
        b = o.get("obb") or {"center": o["center"], "size": o["size"], "yaw": 0.0}
        a = o.get("asset")
        status = "target" if o["id"] == scene.get("target_object") else ("none" if not a else "healthy" if a.get("condition_state") == "healthy" else "warn")
        attrs = [f'custom string valmet:objectId = {s(o["id"])}', f'custom string valmet:label = {s(o.get("label", ""))}',
                 f'custom string valmet:visionClass = {s(o.get("vision_id", o.get("cls", "")))}',
                 f'custom float valmet:visionScore = {float(o.get("vision_score", 0)):.3f}']
        if o.get("group"):
            attrs.append(f'custom string valmet:group = {s(o["group"])}')
        if a:
            attrs += [f'custom string valmet:assetId = {s(a.get("asset_id"))}', f'custom string valmet:tag = {s(a.get("tag"))}',
                      f'custom string valmet:assetClass = {s(a.get("asset_class"))}',
                      f'custom string valmet:condition = {s(a.get("condition_state"))}',
                      f'custom int valmet:health = {int(a.get("health_score") or 0)}']
        c, sz = b["center"], b["size"]
        out += [f'        def Cube "{prim_name(o["id"])}"', '        {', '            double size = 1',
                f'            double3 xformOp:translate = ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})',
                f'            float xformOp:rotateY = {-math.degrees(b.get("yaw", 0.0)):.3f}',
                f'            float3 xformOp:scale = ({sz[0]:.4f}, {sz[1]:.4f}, {sz[2]:.4f})',
                '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateY", "xformOp:scale"]',
                '            color3f[] primvars:displayColor = [({:.2f}, {:.2f}, {:.2f})]'.format(*STATUS_RGB[status]),
                '            float[] primvars:displayOpacity = [0.25]',
                *[f'            {x}' for x in attrs], '        }']
    out.append('    }')
    for name, path in (("MissionRoute", scene.get("path", [])), ("InspectionTour", (scene.get("tour") or {}).get("path", []))):
        if len(path) < 2:
            continue
        pts = ", ".join(f"({p[0]:.3f}, {p[1] + 0.02:.3f}, {p[2]:.3f})" for p in path)
        out += [f'    def BasisCurves "{name}"', '    {', '        uniform token type = "linear"',
                f'        int[] curveVertexCounts = [{len(path)}]', f'        point3f[] points = [{pts}]',
                '        float[] widths = [0.04] (interpolation = "constant")',
                '        color3f[] primvars:displayColor = [(1, 0.83, 0)]', '    }']
    out += ['}', '']
    return "\n".join(out)


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.write_text(scene_to_usda(json.loads(src.read_text())), encoding="utf-8")
    print(f"wrote {dst}")
