import sys
import math
import ROOT
import yaml
import os
import argparse
from array import array
from podio import root_io
import edm4hep
import logging

logger_io = logging.getLogger('io')

# I'm sure this exists already 
def dRAngle(p1,p2):
    """ Calculate the angle between two particles in the eta-phi plane
    Args:
    p1 (TLorentzVector): 4-momentum vector of the first particle
    p2 (TLorentzVector): 4-momentum vector of the second particle
    Returns:
    float: angle between the two particles in the eta-phi plane
    """
    dphi=p1.Phi()-p2.Phi()
    if (dphi>math.pi) : dphi=2*math.pi-dphi
    dtheta=p1.Theta()-p2.Theta()
    dR=math.sqrt(dtheta*dtheta+dphi*dphi)
    return dR

# trick to prevent broken files (should not be a problem at CIEMAT)
def open_root_file(file_path):
    try:
        # Suppress ROOT's default error messages to the terminal
        ROOT.gErrorIgnoreLevel = ROOT.kError

        # Attempt to open the ROOT file in "READ" mode without auto-recovery
        root_file = ROOT.TFile.Open(file_path, "READ")

        # Check if the file is a zombie
        if not root_file or root_file.IsZombie():
            logger_io.error(f"Error: '{file_path}' is a zombie or could not be opened.")
            raise IOError(f"Error: '{file_path}' is a zombie or could not be opened.")
        
        # Check if file is recoverable (potentially corrupted)
        if root_file.TestBit(ROOT.TFile.kRecovered):
            logger_io.error(f"Warning: '{file_path}' is corrupted and has been recovered.")
            raise IOError(f"Error: '{file_path}' is corrupted and has been recovered.")
        
        #print(f"'{file_path}' opened successfully.")
        logger_io.debug("File '%s' opened successfully.", file_path)
        return root_file

    except Exception as e:
        logger_io.error("Error opening file '%s': %s", file_path, e)
        # print(f"Error: {e}")
        return None

# Fuction to sort by tau P
def sort_by_P(Tau):
    tau_with_P = []

    for i in range(0,len(Tau)):
        tau_with_P.append((Tau[i], Tau[i][0].P()))
    
    # Sort the list based on the P() value in descending order
    sorted_tau_with_P = sorted(tau_with_P, key=lambda x: x[1], reverse=True)
    
    # Extract only the sorted Tau[i] objects from the tuples
    sortedTau = [tau for tau, _ in sorted_tau_with_P]
   
    return sortedTau

def load_yaml_config(config_file, default_config):
    """Load the YAML configuration file if it exists.
    Args:
    args (argparse.Namespace): command line arguments
    config_file (str): path to the YAML configuration file
    Returns:
    dict: configuration parameters
    """
    if config_file is not None and os.path.exists(config_file):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
            # print(f"Loaded configuration parameters from '{config_file}'.")
    elif default_config:
        if not os.path.exists(default_config):
            raise FileNotFoundError(f"Error: '{default_config}' does not exist. A valid default configuration file is required.")
        with open(default_config, "r") as file:
            config = yaml.safe_load(file)
            # print(f"Loaded default configuration parameters from '{default_config}'.")
    else:
        raise FileNotFoundError(f"Error: A valid default configuration file is required.")
    return config

def load_yaml_config(config_file, default_config):
    """Load the YAML configuration file if it exists.
    Args:
    args (argparse.Namespace): command line arguments
    config_file (str): path to the YAML configuration file
    Returns:
    dict: configuration parameters
    """
    if config_file is not None and os.path.exists(config_file):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
            # print(f"Loaded configuration parameters from '{config_file}'.")
    elif default_config:
        if not os.path.exists(default_config):
            raise FileNotFoundError(f"Error: '{default_config}' does not exist. A valid default configuration file is required.")
        with open(default_config, "r") as file:
            config = yaml.safe_load(file)
            # print(f"Loaded default configuration parameters from '{default_config}'.")
    else:
        raise FileNotFoundError(f"Error: A valid default configuration file is required.")
    return config                