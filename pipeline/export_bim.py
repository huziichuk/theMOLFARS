"""Scan-to-BIM output of the semantic layer: IFC4 (open in any BIM tool: Revit, BlenderBIM/Bonsai, xeokit, Solibri)
and a 2D semantic floor plan + section A-A as SVG, the "bird's-eye view" Valmet asked for.

    python pipeline/export_bim.py web/scene.json out.ifc out_plan.svg

No IFC library needed: IFC4 STEP text is written directly. Frame -> IFC element by vision class (IfcPump, IfcValve,
IfcElectricMotor, IfcFlowInstrument, IfcSensor, IfcTank, IfcElectricDistributionBoard, IfcPipeSegment, IfcDuctSegment,
IfcLightFixture, IfcFurniture, else IfcBuildingElementProxy), a box solid at its (yaw-oriented) frame, and property
sets Pset_Semantic3D (object, vision class and score) and Pset_ValmetAsset (the mapped Excel record).
World frame: Y-up metres (x, y, z) -> IFC Z-up (x, -z, y - floor), so the storey sits at the floor.
"""
import json
import math
import sys
import uuid
from pathlib import Path
from xml.sax.saxutils import escape

IFC_CLASS = {"PUMP": "IFCPUMP", "VALVE": "IFCVALVE", "MOTOR": "IFCELECTRICMOTOR", "GAUGE": "IFCFLOWINSTRUMENT",
             "SENSOR": "IFCSENSOR", "TANK": "IFCTANK", "CABINET": "IFCELECTRICDISTRIBUTIONBOARD", "PIPE": "IFCPIPESEGMENT",
             "DUCT": "IFCDUCTSEGMENT", "LAMP": "IFCLIGHTFIXTURE", "CHAIR": "IFCFURNITURE", "BOX": "IFCFURNITURE",
             "CART": "IFCFURNITURE", "LADDER": "IFCFURNITURE", "APPLIANCE": "IFCFURNITURE"}
_B64 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"


def ifc_guid(key: str) -> str:
    """Stable 22-char IFC GlobalId from a key (same object -> same GUID on every export)."""
    n = uuid.uuid5(uuid.NAMESPACE_URL, "themolfars:" + key).int
    return _B64[n >> 126] + "".join(_B64[(n >> (6 * i)) & 63] for i in reversed(range(21)))  # 2 + 21x6 bits


def ifc_str(s) -> str:
    """STEP string literal: quotes doubled, backslash escaped, non-ASCII as \\X2\\hhhh\\X0\\."""
    out = []
    for ch in str(s):
        if ch == "'":
            out.append("''")
        elif ch == "\\":
            out.append("\\\\")
        elif 32 <= ord(ch) < 127:
            out.append(ch)
        elif ord(ch) < 0x10000:
            out.append(f"\\X2\\{ord(ch):04X}\\X0\\")
    return "'" + "".join(out) + "'"


def real(v) -> str:
    r = repr(float(round(v, 6)))
    return r if ("." in r or "E" in r or "e" in r) else r + "."


def scene_to_ifc(scene: dict, name: str = "boiler_room.ifc") -> str:
    lines, n = [], [0]

    def add(entity: str) -> int:
        n[0] += 1
        lines.append(f"#{n[0]}={entity};")
        return n[0]

    floor = scene["floor_y"]
    origin = add("IFCCARTESIANPOINT((0.,0.,0.))")
    zdir, xdir = add("IFCDIRECTION((0.,0.,1.))"), add("IFCDIRECTION((1.,0.,0.))")
    world = add(f"IFCAXIS2PLACEMENT3D(#{origin},#{zdir},#{xdir})")
    ctx = add(f"IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-05,#{world},$)")
    body = add(f"IFCGEOMETRICREPRESENTATIONSUBCONTEXT('Body','Model',*,*,*,*,#{ctx},$,.MODEL_VIEW.,$)")
    units = [add("IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.)"), add("IFCSIUNIT(*,.AREAUNIT.,$,.SQUARE_METRE.)"),
             add("IFCSIUNIT(*,.VOLUMEUNIT.,$,.CUBIC_METRE.)"), add("IFCSIUNIT(*,.PLANEANGLEUNIT.,$,.RADIAN.)")]
    ua = add(f"IFCUNITASSIGNMENT(({','.join('#%d' % u for u in units)}))")
    sid = scene.get("scene_id", "")
    project = add(f"IFCPROJECT('{ifc_guid('project' + sid)}',$,'Valmet boiler room digital twin',"
                  f"{ifc_str('theMOLFARS scan-to-BIM, scene ' + sid)},$,$,$,(#{ctx}),#{ua})")
    site_pl = add(f"IFCLOCALPLACEMENT($,#{world})")
    site = add(f"IFCSITE('{ifc_guid('site' + sid)}',$,'Site',$,$,#{site_pl},$,$,.ELEMENT.,$,$,$,$,$)")
    bld_pl = add(f"IFCLOCALPLACEMENT(#{site_pl},#{world})")
    building = add(f"IFCBUILDING('{ifc_guid('building' + sid)}',$,'Plant',$,$,#{bld_pl},$,$,.ELEMENT.,$,$,$)")
    st_pl = add(f"IFCLOCALPLACEMENT(#{bld_pl},#{world})")
    storey = add(f"IFCBUILDINGSTOREY('{ifc_guid('storey' + sid)}',$,'Boiler room',$,$,#{st_pl},$,$,.ELEMENT.,0.)")
    add(f"IFCRELAGGREGATES('{ifc_guid('agg1' + sid)}',$,$,$,#{project},(#{site}))")
    add(f"IFCRELAGGREGATES('{ifc_guid('agg2' + sid)}',$,$,$,#{site},(#{building}))")
    add(f"IFCRELAGGREGATES('{ifc_guid('agg3' + sid)}',$,$,$,#{building},(#{storey}))")

    elements = []
    for o in scene["objects"]:
        b = o.get("obb") or {"center": o["center"], "size": o["size"], "yaw": 0.0}
        cx, cy, cz = b["center"]
        su, sh, sv = b["size"]
        yaw = b.get("yaw", 0.0)
        base = cy - sh / 2 - floor
        p = add(f"IFCCARTESIANPOINT(({real(cx)},{real(-cz)},{real(base)}))")
        ref = add(f"IFCDIRECTION(({real(math.cos(yaw))},{real(-math.sin(yaw))},0.))")
        ax = add(f"IFCAXIS2PLACEMENT3D(#{p},#{zdir},#{ref})")
        place = add(f"IFCLOCALPLACEMENT(#{st_pl},#{ax})")
        c2 = add("IFCCARTESIANPOINT((0.,0.))")
        p2 = add(f"IFCAXIS2PLACEMENT2D(#{c2},$)")
        prof = add(f"IFCRECTANGLEPROFILEDEF(.AREA.,$,#{p2},{real(max(su, 0.01))},{real(max(sv, 0.01))})")
        solid = add(f"IFCEXTRUDEDAREASOLID(#{prof},#{world},#{zdir},{real(max(sh, 0.01))})")
        rep = add(f"IFCSHAPEREPRESENTATION(#{body},'Body','SweptSolid',(#{solid}))")
        shape = add(f"IFCPRODUCTDEFINITIONSHAPE($,$,(#{rep}))")
        a = o.get("asset")
        vid = o.get("vision_id", o.get("cls", ""))
        cls = IFC_CLASS.get(vid, "IFCBUILDINGELEMENTPROXY")
        label = (f"{a['tag']} " if a else "") + str(o.get("label", vid))
        el = add(f"{cls}('{ifc_guid(sid + o['id'])}',$,{ifc_str(label)},{ifc_str(o.get('reason', ''))},{ifc_str(vid)},"
                 f"#{place},#{shape},{ifc_str(a['tag'] if a else o['id'])},.NOTDEFINED.)")
        elements.append(el)
        props = [("ObjectId", f"IFCIDENTIFIER({ifc_str(o['id'])})"), ("VisionClass", f"IFCLABEL({ifc_str(vid)})"),
                 ("VisionScore", f"IFCREAL({real(o.get('vision_score', 0))})"), ("VisionSource", f"IFCLABEL({ifc_str(o.get('vision_source', ''))})")]
        if o.get("group"):
            props.append(("Group", f"IFCIDENTIFIER({ifc_str(o['group'])})"))
        psets = [("Pset_Semantic3D", props)]
        if a:
            psets.append(("Pset_ValmetAsset", [
                ("AssetId", f"IFCIDENTIFIER({ifc_str(a.get('asset_id'))})"), ("Tag", f"IFCIDENTIFIER({ifc_str(a.get('tag'))})"),
                ("AssetClass", f"IFCLABEL({ifc_str(a.get('asset_class'))})"), ("Criticality", f"IFCLABEL({ifc_str(a.get('criticality'))})"),
                ("ConditionState", f"IFCLABEL({ifc_str(a.get('condition_state'))})"),
                ("HealthScore", f"IFCINTEGER({int(a.get('health_score') or 0)})"), ("Urgency", f"IFCLABEL({ifc_str(a.get('urgency'))})"),
                ("IssueCode", f"IFCLABEL({ifc_str(a.get('issue_code'))})"),
                ("IsMissionTarget", f"IFCBOOLEAN({'.T.' if o['id'] == scene.get('target_object') else '.F.'})")]))
        for pname, plist in psets:
            pids = [add(f"IFCPROPERTYSINGLEVALUE({ifc_str(k)},$,{v},$)") for k, v in plist]
            ps = add(f"IFCPROPERTYSET('{ifc_guid(sid + o['id'] + pname)}',$,{ifc_str(pname)},$,({','.join('#%d' % i for i in pids)}))")
            add(f"IFCRELDEFINESBYPROPERTIES('{ifc_guid(sid + o['id'] + pname + 'rel')}',$,$,$,(#{el}),#{ps})")
    add(f"IFCRELCONTAINEDINSPATIALSTRUCTURE('{ifc_guid('contain' + sid)}',$,'Boiler room content',$,"
        f"({','.join('#%d' % e for e in elements)}),#{storey})")
    head = ["ISO-10303-21;", "HEADER;", "FILE_DESCRIPTION(('ViewDefinition [ReferenceView]'),'2;1');",
            f"FILE_NAME({ifc_str(name)},'2026-11-08T00:00:00',('theMOLFARS'),('theMOLFARS'),'','theMOLFARS export_bim.py','');",
            "FILE_SCHEMA(('IFC4'));", "ENDSEC;", "DATA;"]
    return "\n".join(head + lines + ["ENDSEC;", "END-ISO-10303-21;", ""])


# ---------------------------------------------------------------- 2D semantic plan + section
CLASS_COLOR = {"PUMP": "#38d996", "MOTOR": "#50c8b4", "VALVE": "#ffb43c", "GAUGE": "#be8cff", "SENSOR": "#be8cff",
               "TANK": "#4cafff", "CABINET": "#ff78b4", "PIPE": "#6e829b", "DUCT": "#64788c", "APPLIANCE": "#d2d7e1",
               "CART": "#f0965a", "LADDER": "#dcbe5a", "BOX": "#a59178", "CHAIR": "#a59178", "LAMP": "#e6e68c"}


def scene_to_plan_svg(scene: dict, px: float = 90) -> str:
    """Top view (walkable floor, frames by class, Excel tags, route, tour stops) above section A-A, a vertical cut
    through the mission target along x."""
    g = scene["grid"]
    x0, z0, cell, W, H = g["origin"][0], g["origin"][1], g["cell"], g["w"], g["h"]
    floor, ceil = scene["floor_y"], scene["ceiling_y"]
    pw, ph = W * cell * px, H * cell * px
    sec_h = (ceil - floor + 0.4) * px
    pad, title = 20, 34
    sx = lambda x: pad + (x - x0) * px  # noqa: E731
    sz = lambda z: title + pad + (z - z0) * px  # noqa: E731
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{pw + 2 * pad:.0f}" height="{title + ph + sec_h + 4 * pad + 30:.0f}" '
           f'font-family="Arial, sans-serif" font-size="11">',
           '<rect width="100%" height="100%" fill="#ffffff"/>', '<g shape-rendering="crispEdges">',
           f'<text x="{pad}" y="22" font-size="16" font-weight="bold">Boiler room — semantic floor plan (top) and section A-A</text>']
    walk = g["walkable"]
    for j in range(H):  # run-length rows of walkable cells keep the file small
        i = 0
        while i < W:
            if walk[j * W + i] == "1":
                k = i
                while k < W and walk[j * W + k] == "1":
                    k += 1
                out.append(f'<rect x="{sx(x0 + i * cell):.1f}" y="{sz(z0 + j * cell):.1f}" width="{(k - i) * cell * px:.1f}" '
                           f'height="{cell * px:.1f}" fill="#e3f4e6"/>')
                i = k
            else:
                i += 1
    out.append(f'</g><rect x="{pad}" y="{title + pad}" width="{pw:.0f}" height="{ph:.0f}" fill="none" stroke="#9aa4b0"/>')
    target = next(o for o in scene["objects"] if o["id"] == scene["target_object"])
    for o in scene["objects"]:
        b = o.get("obb") or {"center": o["center"], "size": o["size"], "yaw": 0.0}
        vid = o.get("vision_id", "")
        if vid == "STRUCTURE" and not o.get("asset"):
            continue
        col = "#e5484d" if o is target else CLASS_COLOR.get(vid, "#8b93a3")
        cx, cz = sx(b["center"][0]), sz(b["center"][2])
        w, d = b["size"][0] * px, b["size"][2] * px
        out.append(f'<rect x="{cx - w / 2:.1f}" y="{cz - d / 2:.1f}" width="{w:.1f}" height="{d:.1f}" fill="{col}" fill-opacity="0.35" '
                   f'stroke="{col}" stroke-width="{2 if o.get("asset") else 1}" transform="rotate({math.degrees(b.get("yaw", 0)):.1f} {cx:.1f} {cz:.1f})">'
                   f'<title>{o["id"]} {escape(str(o.get("label", "")))}</title></rect>')
        if o.get("asset"):
            out.append(f'<text x="{cx:.1f}" y="{cz - d / 2 - 3:.1f}" text-anchor="middle" font-weight="bold" '
                       f'fill="{"#c62828" if o is target else "#1f2933"}">{escape(str(o["asset"]["tag"]))}</text>')
    for name, path, col, dash in (("route", scene.get("path", []), "#e0a800", ""), ("tour", (scene.get("tour") or {}).get("path", []), "#3a6ea5", ' stroke-dasharray="6 4"')):
        if len(path) > 1:
            out.append(f'<polyline points="{" ".join(f"{sx(p[0]):.1f},{sz(p[2]):.1f}" for p in path)}" fill="none" stroke="{col}" stroke-width="2.5"{dash}/>')
    for k, st in enumerate((scene.get("tour") or {}).get("stops", []), 1):
        p = scene["tour"]["path"][st["path_index"]]
        out.append(f'<circle cx="{sx(p[0]):.1f}" cy="{sz(p[2]):.1f}" r="7" fill="#3a6ea5"/><text x="{sx(p[0]):.1f}" y="{sz(p[2]) + 4:.1f}" '
                   f'text-anchor="middle" fill="#fff" font-size="9">{k}</text>')
    za = target["center"][2]  # section line A-A
    out.append(f'<line x1="{pad}" y1="{sz(za):.1f}" x2="{pad + pw:.0f}" y2="{sz(za):.1f}" stroke="#c62828" stroke-dasharray="8 4"/>'
               f'<text x="{pad + 4}" y="{sz(za) - 4:.1f}" fill="#c62828" font-weight="bold">A</text>'
               f'<text x="{pad + pw - 12:.0f}" y="{sz(za) - 4:.1f}" fill="#c62828" font-weight="bold">A</text>')
    # section A-A: every frame crossing the cut, x horizontal, height vertical
    top = title + ph + 3 * pad + 20
    sy = lambda y: top + (ceil + 0.2 - y) * px  # noqa: E731
    out.append(f'<text x="{pad}" y="{top - 8}" font-size="13" font-weight="bold">Section A-A (z = {za:.2f} m)</text>')
    out.append(f'<line x1="{pad}" y1="{sy(floor):.1f}" x2="{pad + pw:.0f}" y2="{sy(floor):.1f}" stroke="#333" stroke-width="2"/>'
               f'<line x1="{pad}" y1="{sy(ceil):.1f}" x2="{pad + pw:.0f}" y2="{sy(ceil):.1f}" stroke="#999"/>'
               f'<line x1="{pad}" y1="{sy(floor + 1.7):.1f}" x2="{pad + pw:.0f}" y2="{sy(floor + 1.7):.1f}" stroke="#bbb" stroke-dasharray="3 3"/>'
               f'<text x="{pad + pw - 4:.0f}" y="{sy(floor + 1.7) - 3:.1f}" text-anchor="end" fill="#888">head height 1.7 m</text>')
    for o in scene["objects"]:
        c, s = o["center"], o["size"]
        if abs(c[2] - za) > s[2] / 2 or (o.get("vision_id") == "STRUCTURE" and not o.get("asset")):
            continue
        col = "#e5484d" if o is target else CLASS_COLOR.get(o.get("vision_id", ""), "#8b93a3")
        out.append(f'<rect x="{sx(c[0] - s[0] / 2):.1f}" y="{sy(c[1] + s[1] / 2):.1f}" width="{s[0] * px:.1f}" height="{s[1] * px:.1f}" '
                   f'fill="{col}" fill-opacity="0.4" stroke="{col}"><title>{o["id"]}</title></rect>')
        if o.get("asset"):
            out.append(f'<text x="{sx(c[0]):.1f}" y="{sy(c[1] + s[1] / 2) - 3:.1f}" text-anchor="middle" font-weight="bold">{escape(str(o["asset"]["tag"]))}</text>')
    legend_y = top + sec_h + 16
    items = [("walkable floor", "#e3f4e6"), ("fault target", "#e5484d"), ("pump", "#38d996"), ("valve", "#ffb43c"),
             ("gauge / sensor", "#be8cff"), ("tank", "#4cafff"), ("cabinet", "#ff78b4"), ("pipe", "#6e829b")]
    x = pad
    for name, col in items:
        out.append(f'<rect x="{x}" y="{legend_y - 9}" width="12" height="12" fill="{col}" stroke="#666" stroke-width="0.5"/>'
                   f'<text x="{x + 16}" y="{legend_y + 1}">{name}</text>')
        x += 16 + 7 * len(name) + 14
    out.append(f'<text x="{x}" y="{legend_y + 1}" fill="#e0a800">— mission route</text><text x="{x + 110}" y="{legend_y + 1}" fill="#3a6ea5">- - inspection tour</text>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    scene = json.loads(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(scene_to_ifc(scene, Path(sys.argv[2]).name), encoding="ascii")
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(scene_to_plan_svg(scene), encoding="utf-8")
    print("written")
