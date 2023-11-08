"""
<defs>タグを実装
"""

from __future__ import annotations
from .geom import *
from .svg_gradient import *
import re
import torch
from typing import List, Union
from xml.dom import minidom
import math
import numpy as np

class SVGDefs:
    """<defs>の子要素として定義されている諸々のコンポーネント"""
    def __init__(self, defs_list={}):
        self.defs_list = defs_list # from_xmlをした場合、[0]は必ずstyle.ない場合はNone

    @property
    def style(self):
        style = [elem for elem in self.defs_list if isinstance(elem, SVGStyle)]
        if len(style)>0:
            return style[0]
        else:
            return None

    @classmethod
    def from_xml(cls, defs:minidom.Element):
        # defsで定義してuseで再利用する用法は無視
        if defs is None: 
            return SVGDefs()

        if not defs.tagName == "defs":
            raise ValueError("input is not a defs tag")
        
        defs_contents = []
        for child in defs.childNodes:
            if isinstance(child, minidom.Element):
                defs_contents.append(child)

        defs_tags = {
            "linearGradient": SVGLinearGradient,
            "radialGradient": SVGRadialGradient,
            # "pattern": pattern,
            # "marker": marker,
            # "clipPath": clipPath,
            # "filter": filter,
        }

        defs_list = []
        style = None
        for x in defs_contents:
            if x.tagName == "style":
                e = SVGStyle.from_xml(x)
                defs_list.append(e) 
                style = e

        for x in defs_contents:
            for tag, elem in defs_tags.items():
                if x.tagName == tag:
                    e =  elem.from_xml(x,style=style)
                    defs_list.append(e)
        
        return SVGDefs(defs_list=defs_list)

    def _markers(self):
        # 別のクラスに分離しそうな気がする
        return ('<marker id="arrow" viewBox="0 0 10 10" markerWidth="4" markerHeight="4" refX="0" refY="3" orient="auto" markerUnits="strokeWidth">'
                '<path d="M0,0 L0,6 L9,3 z" fill="#f00" />'
                '</marker>\n')

    def to_str(self, with_markers=False, *args, **kwargs,) -> str:
        """ 線形グラデーションのsvgプロンプトを出力する.
        出力プロンプトは<defs>内に置かれる.
        """
        string = "<defs>\n"
        for d in self.defs_list:
            string += d.to_str()
        if with_markers==True:
            string += self._markers() 
        string += "</defs>\n"
        return string

    def copy(self):
        defs = SVGDefs()
        defs.defs_list = self.defs_list.copy()
        return defs


class SVGStyle:
    """styleタグの解析"""
    def __init__(self, classes_dict: dict[str, dict[str,str]]=None, ids_dict: dict[str, dict[str,str]]=None, styleSheetType="text/css"):
        self.type = styleSheetType
        self.classes_dict = classes_dict # {label: {attr_key: value}}の辞書in辞書
        self.ids_dict = ids_dict
        # self.selector_dict = selector_dict # <p>とかのhtml要素。svcなのでたぶん要らない。

    @property
    def contents_dict(self):
        return {**self.classes_dict, **self.ids_dict}

    @classmethod
    def from_xml(cls, x:minidom.Element):
        if not x.tagName == "style":
            raise ValueError("input is not a style tag")
        
        contents = [item for item in x.childNodes if isinstance(item, minidom.CDATASection)][0] # CDATAの中身

        pat = r'(\S+)\s*{\s*([^}]+)\s*}' # 正規表現で解析用の文字列パターン: <任意長の非空白文字列> <連続な空白文字> <{> <連続な空白文字> <}を除く任意長の文字列> <}>

        matches = re.findall(pat, contents.wholeText)
        classes_dict = {}
        ids_dict = {}
        for match in matches:
            name, attributes = match
            attribute_pairs = [attr.strip() for attr in attributes.split(";")]
            attribute_dict = {}
            for pair in attribute_pairs:
                if ":" in pair:
                    key, value = pair.split(":", 1)
                    attribute_dict[key.strip()] = value.strip()
            if name.startswith("."):
                classes_dict[name.strip()] = attribute_dict
            elif name.startswith("#"):
                ids_dict[name.strip()] = attribute_dict

        return SVGStyle(classes_dict=classes_dict, ids_dict=ids_dict, styleSheetType="text/css")

    def to_str(self, *args, **kwargs) -> str:
        # タブキーはなんか色々と標記が崩れそうな気がするので後で完成系strを整形するような関数を作るべき
        string = f'<style type="{self.type}">\n'
        string += '<![CDATA[\n'
        for label, attrs in self.contents_dict.items():
            string += f'\t{label} '
            string += '{'
            for attr, value in attrs.items():
                string += f' {attr}: {value}; '
            string += '}\n'
        string += ']]>\n'
        string += '</style>\n'
        return string
    
    def copy(self):
        raise NotImplementedError
