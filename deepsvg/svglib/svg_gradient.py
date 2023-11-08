"""
グラデーションを実装
https://developer.mozilla.org/ja/docs/Web/SVG/Tutorial/Gradients
"""

from __future__ import annotations
from .geom import *
from .svg_defs import *
import re
import torch
from typing import List, Union
from xml.dom import minidom
import math
import numpy as np

class SVGLinearGradient:
    def __init__(self, grad_id, stops:List[GradientStop], x1=0, y1=0, x2=1, y2=0, spreadMethod="pad"):
        self.id = grad_id
        self.stops = stops
        self.direction = [Point(x1,y1), Point(x2,y2)] # p1→p2 の方向(y軸方向注意)に 0%→100%
        if spreadMethod in ["pad", "reflect", "repeat"]:
            self.spreadMethod = spreadMethod
        else:
            raise ValueError()
        # self.gradientUnits = None 
        # self.gradientTransform = None
    
    @property
    def p1(self):
        return self.direction[0]

    @property
    def p2(self):
        return self.direction[1]

    def print_stops(self):
        for stop in self.stops:
            stop.print_attrs()

    def xlink(self):
        raise NotImplementedError

    def to_str(self, *args, **kwargs) -> str:
        string = f'<linearGradient id="{self.id}"'
        string += f' x1="{self.p1.x}"' 
        string += f' y1="{self.p1.y}"'
        string += f' x2="{self.p2.x}"'
        string += f' y2="{self.p2.y}"'
        if not self.spreadMethod == "pad":
            string += f' spreadMethod="{self.spreadMethod}"'
        string += ">\n"
        for stop in self.stops:
            string += "\t"+stop.to_str()
        string += '</linearGradient>\n'
        return string

    @classmethod
    def from_xml(cls, x: minidom.Element, style: SVGStyle=None):
        # xlinkでstopを指定する方法はそのうち
        if not x.tagName == "linearGradient":
            raise ValueError("input is not a linearGradient tag")

        grad_id = x.getAttribute("id")
        x1 = x.getAttribute("x1") if x.hasAttribute("x1") else "0"
        y1 = x.getAttribute("y1") if x.hasAttribute("y1") else "0"
        x2 = x.getAttribute("x2") if x.hasAttribute("x2") else "1"
        y2 = x.getAttribute("y2") if x.hasAttribute("y2") else "0"
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)
        spreadMethod = x.getAttribute("spreadMethod") if x.hasAttribute("spreadMethod") else "pad"

        stops: List[GradientStop] = []
        for stop in x.childNodes:
            if isinstance(stop, minidom.Element):
                # stopタグ以外は考えない
                s = GradientStop.from_xml(stop, style=style)
                stops.append(s)

        return SVGLinearGradient(grad_id=grad_id, stops=stops, x1=x1, y1=y1, x2=x2, y2=y2, spreadMethod=spreadMethod)

    def set_direction():
        """arg or dig から Pointsに変換してセットとかできたらいいな"""
        NotImplementedError


class SVGRadialGradient:
    def __init__(self, grad_id, stops:List[GradientStop], cx=0, cy=0, r=1, fx=0, fy=0, spreadMethod="pad"):
        self.id = grad_id
        self.stops = stops
        self.center = Point(cx,cy)
        self.radius = Radius(r) # ←怪しい
        self.focus = Point(fx,fy) # 焦点
        if spreadMethod in ["pad", "reflect", "repeat"]:
            self.spreadMethod = spreadMethod
        else:
            raise ValueError()
        # self.gradientUnits = None 
        # self.gradientTransform = None
    
    @property
    def c(self):
        return self.center

    @property
    def f(self):
        return self.focus

    @property
    def r(self):
        return self.radius.x

    def xlink(self):
        raise NotImplementedError

    def print_stops(self):
        for stop in self.stops:
            stop.print_attrs()

    def to_str(self,*args, **kwargs) -> str:
        string = f'<radialGradient id="{self.id}"'
        string += f' cx="{self.c.x}"' 
        string += f' cy="{self.c.y}"'
        string += f' r="{self.r.x}"'
        string += f' fx="{self.f.x}"'
        string += f' fy="{self.f.y}"'
        if not self.spreadMethod == "pad":
            string += f' spreadMethod="{self.spreadMethod}"'
        string += ">\n"
        for stop in self.stops:
            string += "\t" + stop.to_str()
        string += '</radialGradient>\n'
        return string

    @classmethod
    def from_xml(cls, x: minidom.Element, style: SVGStyle=None):
        # xlinkでstopを指定する方法はそのうち
        if not x.tagName == "linearGradient":
            raise ValueError("input is not a linearGradient tag")

        grad_id = x.getAttribute("id")
        cx = x.getAttribute("cx") if x.hasAttribute("cx") else "0"
        cy = x.getAttribute("cy") if x.hasAttribute("cy") else "0"
        r = x.getAttribute("r") if x.hasAttribute("r") else "1"
        fx = x.getAttribute("fx") if x.hasAttribute("fx") else cx
        fy = x.getAttribute("fy") if x.hasAttribute("fy") else cy
        cx = float(cx)
        cy = float(cy)
        r= float(r)
        fx = float(fx)
        fy = float(fy)
        spreadMethod = x.getAttribute("spreadMethod") if x.hasAttribute("spreadMethod") else "pad"

        stops: List[GradientStop] = []
        for stop in x.childNodes:
            if isinstance(stop, minidom.Element):
                # stopタグ以外は考えない
                s = GradientStop.from_xml(stop, style=style)
                stops.append(s)

        return SVGLinearGradient(grad_id=grad_id, stops=stops, x1=x1, y1=y1, x2=x2, y2=y2, spreadMethod=spreadMethod)


class GradientStop:
    """ グラデーションの色分布ノード """
    def __init__(self, offset="0", color="rgb(0,0,0)", opacity="1.0"):
        self.offset = offset # 0.~1. or x%
        self.color = color # rgb or #ffffff or str
        self.opacity = opacity # 0.~1.
    
    def to_str(self,*args, **kwargs) -> str:
        # str化するときはclass表記は特になし
        string = f'<stop offset="{self.offset}" stop-color="{self.color}" stop-opacity="{self.opacity}" />\n' 
        return string

    @classmethod
    def from_xml(cls, x: minidom.Element, style=None):
        # classと個別の属性がそれぞれ定義される場合は、個別の定義でclassを上書きする。要はclass属性よりもstop-color属性のほうが優先される
        offset = "0"
        color = "rgb(0,0,0)"
        opacity = "1.0"
        attrs = {}

        if x.hasAttribute("class"):
            attrs = cls.get_class_attrs(x.getAttribute("class"), style=style)
        if x.hasAttribute("id"):
            attrs = cls.get_id_attrs(x.getAttribute("id"), style=style)
        
        if attrs.get("offset") is not None:
            offset = attrs.get("offset")
        if attrs.get("stop-color") is not None:
            color = attrs.get("stop-color")
        if attrs.get("stop-opacity") is not None:
            opacity = attrs.get("stop-opacity")

        if x.hasAttribute("offset"):
            offset = x.getAttribute("offset")
        if x.hasAttribute("stop-color"):
            color = x.getAttribute("stop-color") 
        if x.hasAttribute("stop-opacity"):
            opacity = x.getAttribute("stop-opacity")

        return GradientStop(offset=offset, color=color, opacity=opacity)

    @staticmethod
    def get_class_attrs(class_name, style: SVGStyle):
        if style is None:
            return {}
        for label, attrs in style.contents_dict.items():
            if label[0] == ".":
                label = label[1:]
            if label == class_name:
                return attrs
        print("No such class exists in the stylesheet")

    @staticmethod
    def get_id_attrs(id_name, style: SVGStyle):
        if style is None:
            return {}
        for label, attrs in style.contents_dict.items():
            if label[0] == ".":
                label = label[1:]
            if label == id_name:
                return attrs
        print("No such class exists in the stylesheet")

    def print_attrs(self):
        print(f"offset: '{self.offset}', color: '{self.color}', opacity: '{self.opacity}'")

    def set_(self, offset=None, color=None, opacity=None):
        if offset is not None:
            self.offset = offset
        if color is not None:
            self.color = color
        if opacity is not None:
            self.opacity = opacity
    
