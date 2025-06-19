import re
import gdsfactory as gf
from shapely.geometry import Polygon
from shapely.ops import unary_union
import pya

import gdstk

import numpy as np




role_map = {'input': 0, 'output': 1, 'internal': 2,'power': 3}
bulk_map = {'VNB': 0, 'VPB': 1}
type_map = {'nfet_01v8': 0, 
            'pfet_01v8': 1, 
            'pfet_01v8_hvt': 2, 
            'nfet_01v8_hvt': 3, 
            'pfet_01v8_lvt': 4, 
            'nfet_01v8_lvt': 5, 
            'nfet_g5v0d10v5': 6, 
            'pfet_g5v0d10v5': 7}


def extract_pin_info(lef_file):
    input_pins = []
    output_pins = []
    inout_pins = []

    pin_name = None
    dir = None
    with open(lef_file,"r") as f:
        lef_data=f.read()
    
    lines=lef_data.splitlines()
    for line in lines:
        line=line.strip()
        if line.startswith("PIN"):
            pin_name=line.split()[1]
        elif line.startswith("DIRECTION"):
            dir=line.split()[1]
            match dir:
                case "INPUT":
                    input_pins.append(pin_name)
                case "OUTPUT":
                    output_pins.append(pin_name)
                case "INOUT" :
                    inout_pins.append(pin_name)
    return input_pins,output_pins,inout_pins

def extract_spice_for_nodes(sprice_file,input_pins,output_pins,inout_pins):
    nodes=[]
    with open(sprice_file,"r") as f:
        spice_data=f.read()
    
    lines=spice_data.splitlines()
    for line in lines:
        node=[]
        line=line.strip()
        if line.startswith("X"):
            features=line.split(" ")
            if features[2] in input_pins:
                node.append("input")
            elif features[2] in output_pins:
                node.append("output")
            elif features[2] in inout_pins:
                node.append("power")
            else:
                node.append("internal")
            node.append(features[4])
            node.append(features[5].split("__")[1])
            node.append(float(features[6].split("=")[1].replace('u', '')) * 1e-6)
            node.append(float(features[7].split("=")[1].replace('u', '')) * 1e-6)
            nodes.append(node)
    return nodes

def extract_spice_for_edges(spice_file):
    with open(spice_file,"r") as f:
        spice_data=f.read()
    nodes=[]
    edges=[]
    source=[]
    target=[]
    lines=spice_data.splitlines()
    for line in lines:
        line=line.strip()
        if line.startswith("X"):
            features=line.split(" ")
            node=features[1:5]
            nodes.append(node)
    for i in range(len(nodes)):
        for j in range(i+1,len(nodes)):
            if set(nodes[i]) & set(nodes[j]):
                source.append(i)
                target.append(j)
    edges.append(source)
    edges.append(target)

    return edges

def one_hot(index,size):
    vec=[0]*size
    vec[index]=1
    return vec

def encode_nodes(nodes):
    encode_nodes=[]
    for node in nodes:
        role_onehot=one_hot(role_map[node[0]],len(role_map))
        bulk_onehot=one_hot(bulk_map[node[1]],len(bulk_map))
        type_onehot=one_hot(type_map[node[2]],len(type_map))
        encoded_node=role_onehot+bulk_onehot+type_onehot+[node[3],node[4]]
        encode_nodes.append(encoded_node)
    return encode_nodes


if __name__ == "__main__":
    i,o,p=extract_pin_info("sky130_fd_sc_hd__fa_1.lef")
    # print(i,o,p)
    nodes=extract_spice_for_nodes("sky130_fd_sc_hd__fa_1.spice",i,o,p)
    print("\n".join(f"{node}" for node in nodes))
    edges=extract_spice_for_edges("sky130_fd_sc_hd__fa_1.spice")
    print(edges[0])
    print(edges[1])
    ed_nodes=encode_nodes(nodes)
    print("\n".join(f"{ed_node}" for ed_node in ed_nodes))
