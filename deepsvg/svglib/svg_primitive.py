from __future__ import annotations
from .geom import *
import torch
import re
from typing import List, Union
from xml.dom import minidom
from .svg_path import SVGPath, Filling
from .svg_command import SVGCommandLine, SVGCommandArc, SVGCommandBezier, SVGCommandClose
import shapely
import shapely.ops
import shapely.geometry
import networkx as nx

from .utils import CSS_COLORS

FLOAT_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")

def extract_args(args):
    return list(map(float, FLOAT_RE.findall(args)))

class SVGPrimitive:
    """
    Reference: https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Basic_Shapes
    """
    def __init__(self, color="black", stroke_color="none", fill=False, stroke=False, dasharray=None, stroke_width=".3", opacity="1.0", stroke_opacity="1.0", all_attrs={}, *args, **kwargs):
        self.color = color
        if not stroke_color == "none":
            self.stroke_color = stroke_color # 追加
        else :
            self.stroke_color = color
        self.dasharray = dasharray
        self.stroke_width = stroke_width
        self.opacity = opacity
        self.stroke_opacity = stroke_opacity # 追加

        self.all_attrs = all_attrs # その他の属性。辞書(実際にはxmlで指定されるすべての属性が入る)
        # その他の属性が含まれる場合にto_pathでパスへの近似を行うと元の図形と異なる図形が出力される

        self.fill = fill
        self.stroke = stroke
        # stroke/fillにより、stroke/fillの有効化が決定される。


    def _get_fill_attr(self):
        # 色関係の属性strを返す        
        fill_attr = ""

        if self.fill:
            fill_attr = f'fill="{self.color}" fill-opacity="{self.opacity}" '
            if self.stroke:
                fill_attr += f'stroke="{self.stroke_color}" stroke-width="{self.stroke_width}" stroke-opacity="{self.stroke_opacity}" '
        else:
            fill_attr = f'fill="none" '
            if self.stroke:
                fill_attr += f'stroke="{self.stroke_color}" stroke-width="{self.stroke_width}" stroke-opcity="{self.stroke_opacity}" '
                
        if self.dasharray is not None:
            fill_attr += f'stroke-dasharray="{self.dasharray}" '

        return fill_attr
    
    def _get_other_attr(self, excludes=[]):
        # その他の属性strを返す.excludeは無視する属性
        exclude_attrs = ["fill", "fill-opacity", "stroke", "stroke-opacity", "stroke-width", "opacity", "stroke-dasharray"] # opacityは入れてもいいかも
        for item in excludes:
            if item not in exclude_attrs:
                exclude_attrs.append(item)

        attr = ""
        for name, value in self.all_attrs.items():
            if name not in exclude_attrs:
                attr += f'{name}="{value}" '
        return attr

    @classmethod
    def from_xml(cls, x: minidom.Element):
        raise NotImplementedError

    def draw(self, viewbox=Bbox(24), *args, **kwargs):
        from .svg import SVG
        return SVG([self], viewbox=viewbox).draw(*args, **kwargs)

    def _get_viz_elements(self, with_points=False, with_handles=False, with_bboxes=False, color_firstlast=True, with_moves=True):
        return []

    def to_path(self):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def bbox(self):
        raise NotImplementedError

    def fill_(self, fill=True):
        self.fill = fill
        return self

    def set_all_attrs_from_xml(self, x: minidom.Element):
        # 全ての属性を取得する
        # fillmode による分岐からfill/strokeによる分岐に書き換え
        attrs = x.attributes
        all_attrs = {}
        for name,  value in attrs.items():
            all_attrs[name] = value
            if name == "fill":
                self.color = value
            if name == "stroke":
                self.stroke_color = value
            if name == "stroke-dasharray":
                self.dasharray = value
            if name == "opacity":
                self.opacity = value
                self.stroke_opacity = value
            if name == "fill-opacity":
                self.opacity = value
            if name == "stroke-opacity":
                self.stroke_opacity = value
            if name == "stroke-width":
                self.stroke_width = value
        self.all_attrs = all_attrs
        
        if "fill" in self.all_attrs or not self.all_attrs["fill"] == "none":
            self.fill = True
        else: 
            self.fill = False
        if "stroke" in self.all_attrs or not self.all_attrs["stroke"] ==  "none":
            self.stroke = True
        else:
            self.stroke = False

        return self.all_attrs

    def color_rgba(self, fill=True):
        # 色をrgba値のリストで返す.fillの場合fill、そうでない場合stroke
        color = self.color if fill else self.stroke_color
        opacity = self.opacity if fill else self.stroke_opacity
        if color[0] == "#": # 16進表記
            r = float.fromhex(color[1:3])
            g = float.fromhex(color[3:5])
            b = float.fromhex(color[5:7])
        elif color[0:3] == "rgb":
            pat = r"\s*rgb\(([1-9, ]+)\)"
            rgb = re.findall(pat,color)[0].replace(" ","")
            rgb = rgb.split(",")
            r = float(rgb[0][:-1])*256 if rgb[0][-1] == "%" else float(rgb[0])
            g = float(rgb[1][:-1])*256 if rgb[1][-1] == "%" else float(rgb[1])
            b = float(rgb[2][:-1])*256 if rgb[2][-1] == "%" else float(rgb[2])
        elif color.lower() in CSS_COLORS.keys():
            r = float.fromhex(CSS_COLORS[color.lower()][1:3])
            g = float.fromhex(CSS_COLORS[color.lower()][3:5])
            b = float.fromhex(CSS_COLORS[color.lower()][5:7])
        elif re.match(r"\s*(\S+)\(.+\)", color) in ["hsl","hwb","lch","oklch",";lab","oklab"]:
            raise NotImplementedError

        a = float(opacity[:-1])*0.01 if opacity[-1]=="%" else float(opacity)
        return [r,g,b,a]


class SVGEllipse(SVGPrimitive):
    #SVGPrimitiveのコンストラクタの変更に伴いいくつか変更する必要がある。
    def __init__(self, center: Point, radius: Radius, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.center = center
        self.radius = radius

    def __repr__(self):
        return f'SVGEllipse(c={self.center} r={self.radius})'

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        other_attr = self._get_other_attr(excludes= ["cx","cy","rx","ry"])
        return f'<ellipse {fill_attr} cx="{self.center.x}" cy="{self.center.y}" rx="{self.radius.x}" ry="{self.radius.y}" {other_attr}/>'

    @classmethod
    def from_xml(_, x: minidom.Element):
        fill = not x.hasAttribute("fill") or not x.getAttribute("fill") == "none"

        center = Point(float(x.getAttribute("cx")), float(x.getAttribute("cy")))
        radius = Radius(float(x.getAttribute("rx")), float(x.getAttribute("ry")))
        return SVGEllipse(center, radius, fill=fill)

    def to_path(self):
        p0, p1 = self.center + self.radius.xproj(), self.center + self.radius.yproj()
        p2, p3 = self.center - self.radius.xproj(), self.center - self.radius.yproj()
        commands = [
            SVGCommandArc(p0, self.radius, Angle(0.), Flag(0.), Flag(1.), p1),
            SVGCommandArc(p1, self.radius, Angle(0.), Flag(0.), Flag(1.), p2),
            SVGCommandArc(p2, self.radius, Angle(0.), Flag(0.), Flag(1.), p3),
            SVGCommandArc(p3, self.radius, Angle(0.), Flag(0.), Flag(1.), p0),
        ]
        filling = Filling.FILL if self.fill else Filling.OUTLINE
        return SVGPath(commands, closed=True, filling=filling).to_group(*vars(self))


class SVGCircle(SVGEllipse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f'SVGCircle(c={self.center} r={self.radius})'

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        other_attr = self._get_other_attr(excludes= ["cx","cy","r"])
        return f'<circle {fill_attr} cx="{self.center.x}" cy="{self.center.y}" r="{self.radius.x}" {other_attr}/>'

    @classmethod
    def from_xml(_, x: minidom.Element):
        fill = not x.hasAttribute("fill") or not x.getAttribute("fill") == "none"

        center = Point(float(x.getAttribute("cx")), float(x.getAttribute("cy")))
        radius = Radius(float(x.getAttribute("r")))
        return SVGCircle(center, radius, fill=fill)


class SVGRectangle(SVGPrimitive):
    def __init__(self, xy: Point, wh: Size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.xy = xy
        self.wh = wh

    def __repr__(self):
        return f'SVGRectangle(xy={self.xy} wh={self.wh})'

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        other_attr = self._get_other_attr(excludes= ["x","y","width","height"])
        return f'<rect {fill_attr} x="{self.xy.x}" y="{self.xy.y}" width="{self.wh.x}" height="{self.wh.y}" {other_attr}/>'

    @classmethod
    def from_xml(_, x: minidom.Element):
        fill = not x.hasAttribute("fill") or not x.getAttribute("fill") == "none"

        xy = Point(0.)
        if x.hasAttribute("x"):
            xy.pos[0] = float(x.getAttribute("x"))
        if x.hasAttribute("y"):
            xy.pos[1] = float(x.getAttribute("y"))
        wh = Size(float(x.getAttribute("width")), float(x.getAttribute("height")))
        return SVGRectangle(xy, wh, fill=fill)

    def to_path(self):
        p0, p1, p2, p3 = self.xy, self.xy + self.wh.xproj(), self.xy + self.wh, self.xy + self.wh.yproj()
        commands = [
            SVGCommandLine(p0, p1),
            SVGCommandLine(p1, p2),
            SVGCommandLine(p2, p3),
            SVGCommandLine(p3, p0)
        ]
        filling = Filling.FILL if self.fill else Filling.OUTLINE

        return SVGPath(commands, closed=True, filling=filling).to_group(*vars(self))


class SVGLine(SVGPrimitive):
    def __init__(self, start_pos: Point, end_pos: Point, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.start_pos = start_pos
        self.end_pos = end_pos

    def __repr__(self):
        return f'SVGLine(xy1={self.start_pos} xy2={self.end_pos})'

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        other_attr = self._get_other_attr(excludes= ["x1","1y","x2","y2"])
        return f'<line {fill_attr} x1="{self.start_pos.x}" y1="{self.start_pos.y}" x2="{self.end_pos.x}" y2="{self.end_pos.y}" {other_attr}/>'

    @classmethod
    def from_xml(_, x: minidom.Element):
        fill = not x.hasAttribute("fill") or not x.getAttribute("fill") == "none"

        start_pos = Point(float(x.getAttribute("x1") or 0.), float(x.getAttribute("y1") or 0.))
        end_pos = Point(float(x.getAttribute("x2") or 0.), float(x.getAttribute("y2") or 0.))
        return SVGLine(start_pos, end_pos, fill=fill)

    def to_path(self):
        filling = Filling.FILL if self.fill else Filling.OUTLINE
        return SVGPath([SVGCommandLine(self.start_pos, self.end_pos)], filling=filling).to_group(*vars(self))


class SVGPolyline(SVGPrimitive):
    def __init__(self, points: List[Point], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.points = points

    def __repr__(self):
        return f'SVGPolyline(points={self.points})'

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        other_attr = self._get_other_attr(excludes= ["points"])
        return '<polyline {} points="{}" {}/>'.format(fill_attr, ' '.join([p.to_str() for p in self.points], other_attr))

    @classmethod
    def from_xml(cls, x: minidom.Element):
        fill = not x.hasAttribute("fill") or not x.getAttribute("fill") == "none"

        args = extract_args(x.getAttribute("points"))
        assert len(args) % 2 == 0, f"Expected even number of arguments for SVGPolyline: {len(args)} given"
        points = [Point(x, args[2*i+1]) for i, x in enumerate(args[::2])]
        return cls(points, fill=fill)

    def to_path(self):
        commands = [SVGCommandLine(p1, p2) for p1, p2 in zip(self.points[:-1], self.points[1:])]
        is_closed = self.__class__.__name__ == "SVGPolygon"
        filling = Filling.FILL if self.fill else Filling.OUTLINE
        return SVGPath(commands, closed=is_closed, filling=filling).to_group(*vars(self))


class SVGPolygon(SVGPolyline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f'SVGPolygon(points={self.points})'

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        other_attr = self._get_other_attr(excludes= ["points"])
        return '<polygon {} points="{}" {}/>'.format(fill_attr, ' '.join([p.to_str() for p in self.points], other_attr))


class SVGPathGroup(SVGPrimitive):
    def __init__(self, svg_paths: List[SVGPath] = None, origin=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.svg_paths = svg_paths

        if origin is None:
            origin = Point(0.)
        self.origin = origin

    # Alias
    @property
    def paths(self):
        return self.svg_paths

    @property
    def path(self):
        return self.svg_paths[0]

    def __getitem__(self, idx):
        return self.svg_paths[idx]

    def __len__(self):
        return len(self.paths)

    def total_len(self):
        # 全てのパスのコマンド数の合計
        return sum([len(path) for path in self.svg_paths])

    @property
    def start_pos(self):
        return self.svg_paths[0].start_pos

    @property
    def end_pos(self):
        last_path = self.svg_paths[-1]
        if last_path.closed:
            return last_path.start_pos
        return last_path.end_pos

    def set_origin(self, origin: Point):
        self.origin = origin
        if self.svg_paths:
            self.svg_paths[0].origin = origin
        self.recompute_origins()

    def append(self, path: SVGPath):
        self.svg_paths.append(path)

    def copy(self):
        #colorとopacityが同じ変数による定義なのでOUTLINE_modeの時に表示がおかしくなる
        paths = [svg_path.copy() for svg_path in self.svg_paths]
        ori = self.origin.copy()
        attrs = vars(self).copy()
        del attrs["origin"]
        del attrs["svg_paths"]

        return SVGPathGroup(paths, ori,
                            **attrs)

    def __repr__(self):
        return "SVGPathGroup({})".format(", ".join(svg_path.__repr__() for svg_path in self.svg_paths))

    def _get_viz_elements(self, with_points=False, with_handles=False, with_bboxes=False, color_firstlast=True, with_moves=True):
        viz_elements = []
        for svg_path in self.svg_paths:
            viz_elements.extend(svg_path._get_viz_elements(with_points, with_handles, with_bboxes, color_firstlast, with_moves))

        if with_bboxes:
            viz_elements.append(self._get_bbox_viz())

        return viz_elements

    def _get_bbox_viz(self):
        color = "red" if self.color == "black" else self.color
        bbox = self.bbox().to_rectangle(color=color)
        return bbox

    def to_path(self):
        return self

    def to_str(self, with_markers=False, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        other_attr = self._get_other_attr(excludes= ["filling", "d"])
        marker_attr = 'marker-start="url(#arrow)"' if with_markers else ''
        return '<path {} {} filling="{}" {} d="{}"></path>'.format(fill_attr, marker_attr, self.path.filling, other_attr,
                                                   " ".join(svg_path.to_str() for svg_path in self.svg_paths))
        # 最初のpathのfillingで全ての部分に関するfillingが決まるならばSVGPathクラスでfillingを決定するのは多分無駄。というか他のプリミティブに適応的ないのでSVGPrimitive.fillをFillingクラスオブジェクトにするべきなような

    def to_tensor(self, PAD_VAL=-1):
        cmd_tensor = torch.cat([p.to_tensor(PAD_VAL=PAD_VAL) for p in self.svg_paths], dim=0)
        # color_tensor = torch.tensor(self.color_rgba)
        print(self.color_rgba(self.fill)) ##ここから
        return cmd_tensor

    def _apply_to_paths(self, method, *args, **kwargs):
        for path in self.svg_paths:
            getattr(path, method)(*args, **kwargs)
        return self

    def translate(self, vec):
        return self._apply_to_paths("translate", vec)

    def rotate(self, angle: Angle):
        return self._apply_to_paths("rotate", angle)

    def scale(self, factor):
        return self._apply_to_paths("scale", factor)

    def numericalize(self, n=256):
        return self._apply_to_paths("numericalize", n)

    def drop_z(self):
        return self._apply_to_paths("set_closed", False)

    def recompute_origins(self):
        origin = self.origin
        for path in self.svg_paths:
            path.origin = origin.copy()
            origin = path.end_pos
        return self

    def reorder(self):
        self._apply_to_paths("reorder")
        self.recompute_origins()
        return self

    def filter_empty(self):
        self.svg_paths = [path for path in self.svg_paths if path.path_commands]
        return self

    def canonicalize(self):
        self.svg_paths = sorted(self.svg_paths, key=lambda x: x.start_pos.tolist()[::-1])
        if not self.svg_paths[0].is_clockwise():
            self._apply_to_paths("reverse")

        self.recompute_origins()
        return self

    def reverse(self):
        self._apply_to_paths("reverse")

        self.recompute_origins()
        return self

    def duplicate_extremities(self):
        self._apply_to_paths("duplicate_extremities")
        return self

    def reverse_non_closed(self):
        self._apply_to_paths("reverse_non_closed")

        self.recompute_origins()
        return self

    def simplify(self, tolerance=0.1, epsilon=0.1, angle_threshold=179., force_smooth=False):
        self._apply_to_paths("simplify", tolerance=tolerance, epsilon=epsilon, angle_threshold=angle_threshold,
                             force_smooth=force_smooth)
        self.recompute_origins()
        return self

    def split_paths(self):
        return [SVGPathGroup([svg_path], self.origin,
                             self.color, self.stroke_color,self.fill, self.stroke, self.dasharray, self.stroke_width, self.opacity, self.stroke_opacity, self.all_attrs)
                for svg_path in self.svg_paths]

    def split(self, n=None, max_dist=None, include_lines=True):
        return self._apply_to_paths("split", n=n, max_dist=max_dist, include_lines=include_lines)

    def simplify_arcs(self):
        return self._apply_to_paths("simplify_arcs")

    def filter_consecutives(self):
        return self._apply_to_paths("filter_consecutives")

    def filter_duplicates(self):
        return self._apply_to_paths("filter_duplicates")

    def bbox(self):
        return union_bbox([path.bbox() for path in self.svg_paths])

    def to_shapely(self):
        return shapely.ops.unary_union([path.to_shapely() for path in self.svg_paths])

    def compute_filling(self):
        if self.fill:
            G = self.overlap_graph()

            root_nodes = [i for i, d in G.in_degree() if d == 0]

            for root in root_nodes:
                if not self.svg_paths[root].closed:
                    continue

                current = [(1, root)]

                while current:
                    visited = set()
                    neighbors = set()
                    for d, n in current:
                        self.svg_paths[n].set_filling(d != 0)

                        for n2 in G.neighbors(n):
                            if not n2 in visited:
                                d2 = d + (self.svg_paths[n2].is_clockwise() == self.svg_paths[n].is_clockwise()) * 2 - 1
                                visited.add(n2)
                                neighbors.add((d2, n2))

                    G.remove_nodes_from([n for d, n in current])

                    current = [(d, n) for d, n in neighbors if G.in_degree(n) == 0]

        return self

    def overlap_graph(self, threshold=0.9, draw=False):
        G = nx.DiGraph()
        shapes = [path.to_shapely() for path in self.svg_paths]

        for i, path1 in enumerate(shapes):
            G.add_node(i)

            if self.svg_paths[i].closed:
                for j, path2 in enumerate(shapes):
                    if i != j and self.svg_paths[j].closed:
                        overlap = path1.intersection(path2).area / path1.area
                        if overlap > threshold:
                            G.add_edge(j, i, weight=overlap)

        if draw:
            pos = nx.spring_layout(G)
            nx.draw_networkx(G, pos, with_labels=True)
            labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        return G

    def bbox_overlap(self, other: SVGPathGroup):
        return self.bbox().overlap(other.bbox())

    def to_points(self):
        return np.concatenate([path.to_points() for path in self.svg_paths])
