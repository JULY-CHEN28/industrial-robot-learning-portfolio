"""Extract pinned official ZU5 assets from a local Git object database."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import math

ROOT = Path(__file__).resolve().parent
REVISION = "f6b28c6be93f289f101c6cc3907caf7804fb1a47"


def quaternion(rpy):
    r, p, y = [float(x) / 2 for x in rpy.split()]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return " ".join(map(str, (cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                              cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy)))


def origin(element):
    value = element.find("origin")
    return {"pos": value.get("xyz", "0 0 0"), "quat": quaternion(value.get("rpy", "0 0 0"))}


def build():
    assets = ROOT / "assets"
    urdf = ET.parse(assets / "jaka_zu5.urdf").getroot()
    links = {link.get("name"): link for link in urdf.findall("link")}
    root = ET.Element("mujoco", model="jaka_zu5_reach_provisional")
    ET.SubElement(root, "compiler", angle="radian", meshdir="meshes", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="720")
    defaults = ET.SubElement(root, "default")
    ET.SubElement(defaults, "joint", damping="1", armature="0.05")
    ET.SubElement(defaults, "geom", friction="0.7 0.005 0.0001", solref="0.008 1")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1")
    ET.SubElement(world, "camera", name="overview", pos="1.5 -1.8 1.3", xyaxes="0.768 0.640 0 -0.232 0.278 0.932")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 0.05", pos="0 0 -0.002", rgba="0.22 0.26 0.30 1")
    target = ET.SubElement(world, "body", name="target", mocap="true", pos="0.4 0 0.4")
    ET.SubElement(target, "geom", name="target_marker", type="sphere", size="0.025", rgba="1 0.15 0.1 0.65", contype="0", conaffinity="0")
    bodies = {"world": world}
    contact = ET.SubElement(root, "contact")
    actuator = ET.SubElement(root, "actuator")
    for joint in urdf.findall("joint"):
        parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
        body = ET.SubElement(bodies[parent], "body", name=child, **origin(joint))
        bodies[child] = body
        link = links[child]
        inertia = link.find("inertial")
        if inertia is not None:
            # Official inertial frames have zero rotation; full tensor is preserved.
            if inertia.find("origin").get("rpy") != "0 0 0":
                raise ValueError("Nonzero inertial rotation requires explicit tensor rotation")
            tensor = inertia.find("inertia")
            ET.SubElement(body, "inertial", pos=inertia.find("origin").get("xyz"),
                          mass=inertia.find("mass").get("value"),
                          fullinertia=" ".join(tensor.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")))
        mesh_name = child + "_mesh"
        mesh_file = link.find("visual/geometry/mesh").get("filename").split("/")[-1]
        ET.SubElement(asset, "mesh", name=mesh_name, file=mesh_file)
        ET.SubElement(body, "geom", name=child+"_visual", type="mesh", mesh=mesh_name,
                      rgba="0.8 0.82 0.84 1", contype="0", conaffinity="0", group="2")
        ET.SubElement(body, "geom", name=child+"_collision", type="mesh", mesh=mesh_name,
                      group="3", rgba="0.4 0.7 0.4 0.25")
        if parent != "world":
            ET.SubElement(contact, "exclude", body1=parent, body2=child)
        if joint.get("type") == "revolute":
            limit = joint.find("limit")
            ET.SubElement(body, "joint", name=joint.get("name"), type="hinge",
                          axis=joint.find("axis").get("xyz"), range=limit.get("lower")+" "+limit.get("upper"))
            index = int(joint.get("name").split("_")[-1]) - 1
            # Simulation commissioning values, NOT vendor hardware specifications.
            kp = [800, 800, 600, 150, 100, 80][index]
            kv = [80, 80, 60, 15, 10, 8][index]
            cap = [150, 150, 100, 40, 30, 20][index]
            ET.SubElement(actuator, "position", name=joint.get("name"), joint=joint.get("name"),
                          kp=str(kp), kv=str(kv), forcerange=f"{-cap} {cap}")
    ET.SubElement(bodies["Link6"], "site", name="flange", pos="0 0 0", size="0.008", rgba="0 1 0 1")
    # The fixed link0 is coplanar with the mount; no floor/base collision is useful.
    # Robot-to-floor and all non-adjacent self contacts remain enabled.
    ET.indent(root)
    ET.ElementTree(root).write(assets / "scene.xml", encoding="utf-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    args = parser.parse_args()
    files = ["src/jaka_description/urdf/jaka_zu5.urdf", "src/jaka_description/package.xml"]
    files += ["src/jaka_description/meshes/jaka_zu5_meshes/"+name for name in
              ["link0.STL"] + [f"Link{i}.STL" for i in range(1, 7)]]
    manifest = {"repository": "https://github.com/JAKARobotics/jaka_ros2", "commit": REVISION, "files": {}}
    for name in files:
        data = subprocess.check_output(["git", "-C", str(args.source_repo), "show", REVISION+":"+name])
        dest = ROOT / "assets" / ("meshes" if name.endswith("STL") else "") / Path(name).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        manifest["files"][name] = hashlib.sha256(data).hexdigest()
    (ROOT / "assets" / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    build()
    print("Official assets extracted; provisional MJCF generated.")


if __name__ == "__main__":
    main()
