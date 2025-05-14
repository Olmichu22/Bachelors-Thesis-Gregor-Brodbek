import ROOT

class Particle():
  """ Standard particle class to store the information of a particle.
  Attributes:
      p4 (TLorentzVector): 4-momentum of the particle.
      PDGID (int): PDG ID of the particle.
      ID (int): Decay ID of the particle.
      charge (int): Charge of the particle.
  """
  
  def __init__(self, p4=None, PDGID=-1, ID=-1, charge=0):
    """ Constructor of the Particle class."""
    self.p4 = p4 if p4 is not None else ROOT.TLorentzVector(0, 0, 0, 0)
    self.PDGID = PDGID
    self.ID = ID
    self.charge = charge
  
  def getPDG(self):
    return self.PDGID
  
  def getID(self):
    return self.ID
  
  def getCharge(self):
    return self.charge
  
  def getMomentum(self):
    return self.p4
  
  def getMass(self):
    return self.p4.M()
  
  def setPDG(self, PDGID):
    self.PDGID = PDGID
    
  def setID(self, ID):
    self.ID = ID
  
  def setCharge(self, charge):
    self.charge = charge
  
  def setMomentum(self, p4):
    self.p4 = p4
  
  def ShowInfo(self):
    """ Print the information of the particle."""
    print("Particle PDG: %d, ID: %d, Charge: %d, Mass: %f" % (self.PDGID, self.ID, self.charge, self.p4.M()))
    
  def __str__(self):
    return "PDG: %d, ID: %d, Charge: %d, Mass: %f" % (self.PDGID, self.ID, self.charge, self.p4.M())

class RecoParticle(Particle):
  """ Particle class to store the information of a reco level particle.
  Attributes:
      p4 (TLorentzVector): 4-momentum of the particle.
      ID (int): Decay ID of the particle.
      charge (int): Charge of the particle.
      maxCone (float): Maximum angle between the guide particle and the rest of the constituents.
      nConst (int): Number of constituents of the particle.
      const (dict): Dictionary with the constituents of the particle.
      PDGID (int):  PDG ID of the particle.
  """
  
  def __init__(self, p4=None, ID=-1, charge=0, maxCone=0, nConst=0, const=None, PDGID=-1):
    """ Constructor of the Particle class."""
    # Initialize the parent class
    super(RecoParticle, self).__init__(p4, PDGID, ID, charge)
    
    self.maxCone = maxCone
    self.nConst = nConst
    self.const = const if const is not None else {}
  
  def getMomentum(self):
    return self.p4
  
  def getMass(self):
    return self.p4.M()
  
  def getID(self):
    return self.ID
  
  def getPDG(self):
    return self.PDGID
  
  def getCharge(self):
    return self.charge  
  
  def getMaxCone(self):
    return self.maxCone

  def getnConst(self):
    return self.nConst
  
  def getDaughters(self):
    return self.const
  
  def setMomentum(self, p4):
    self.p4 = p4
  
  def setID(self, ID):
    self.ID = ID
  
  def setPDG(self, PDGID):
    self.PDGID = PDGID
  
  def setCharge(self, charge):
    self.charge = charge
  
  def setMaxCone(self, maxCone):
    self.maxCone = maxCone
  
  def setDaughters(self, const):
    self.const = const
    self.nConst = len(const)
  
  def addDaughter(self, daughter):
    self.const[self.nConst] = daughter
    self.nConst += 1
  
  def ShowInfo(self):
    """ Print the information of the particle and its constituents."""
    print("RecoParticle ID: %d, Charge: %d, Mass: %f, nConst: %d, Angle: %f" % (self.ID, self.charge, self.p4.M(), self.nConst, self.maxCone))
    for i in range(self.nConst):
      print("  Constituent %d:" % i)
      print(self.const[i])
  
  def __str__(self):
    return "ID: %d, Charge: %d, Mass: %f, nConst: %d, Angle: %f" % (self.ID, self.charge, self.p4.M(), self.nConst, self.maxCone)
  

class GenParticle(Particle):
  """ Particle class to store the information of a gen level particle.
  Attributes:
      visp4 (TLorentzVector): 4-momentum of the visible part of the particle.
      ID (int): Decay ID of the particle.
      charge (int): Charge of the particle.
      visCharge (int): Charge of the visible part of the particle.
      genP4 (TLorentzVector): 4-momentum of the particle.
      maxAngleConsts (float): Maximum angle between the constituents of the particle.
      nConst (int): Number of constituents of the particle.
      const (dict): Dictionary with the constituents of the particle.
      PDGID (int):  PDG ID of the particle.
  """
  
  def __init__(self, visP4=None, ID=-1, charge=0, genP4=None, maxAngleConsts=0, nConsts=0, const=None, PDGID=-1):
    """ Constructor of the Particle class."""
    # Initialize the parent class
    super(GenParticle, self).__init__(genP4, PDGID, ID, charge)
    
    self.visp4 = visP4 if visP4 is not None else ROOT.TLorentzVector(0, 0, 0, 0)
    self.maxAngle = maxAngleConsts
    self.nConst = nConsts
    self.const = const if const is not None else {}
    
  def getPDG(self):
    return self.PDGID
    
  def getID(self):
    return self.ID

  def getCharge(self):
    return self.charge
  
  def getMomentum(self):
    return self.p4
  
  def getvisMomentum(self):
    return self.visp4
  
  def getMass(self):
    return self.p4.M()
  
  def getVisMass(self):
    return self.visp4.M()
  
  def getDaughters(self):
    return self.const
  
  def getnConst(self):
    return self.nConst
  
  def getMaxAngle(self):
    return self.maxAngle
  
  def setID(self, ID):
    self.ID = ID
  
  def setPDG(self, PDGID):
    self.PDGID = PDGID
  
  def setCharge(self, charge):
    self.charge = charge
  
  
  def setMomentum(self, candP):
    self.p4.SetXYZM(candP.getMomentum().x,
                    candP.getMomentum().y,
                    candP.getMomentum().z,
                    candP.getMass())
  
  def setvisMomentum(self, visP4):
    self.visp4 = visP4
  
  def setDaughters(self, const):
    self.const = const
    self.nConst = len(const)
  
  def addDaughter(self, daughter):
    self.const[self.nConst] = daughter
    self.nConst += 1
  
  def setnConst(self, nConst):
    self.nConst = nConst
  
  def setMaxAngle(self, maxAngle):
    self.maxAngle = maxAngle
    
  def ShowInfo(self):
    """ Print the information of the particle and its constituents."""
    print("GenParticle ID: %d, Charge: %d, Mass: %f, VisMass: %f, nConst: %d, Angle: %f" % (self.ID, self.charge, self.p4.M(), self.visp4.M(), self.nConst, self.maxAngle))
    for i in range(self.nConst):
      print("  Constituent %d:" % i)
      print(self.const[i])
  
  def __str__(self):
    return "ID: %d, Charge: %d, Mass: %f, VisMass: %f, nConst: %d, Angle: %f" % (self.ID, self.charge, self.p4.M(), self.visp4.M(), self.nConst, self.maxAngle)