import pubchempy as pcp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem , Draw
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
import os
os.environ["CHROMADB_DEFAULT_DATABASE"] = "duckdb"
#from crewai import Agent, Task, Crew, LLM
#from crewai_tools import SerperDevTool
import base64
from io import BytesIO
import datetime




# Suppress warnings
warnings.filterwarnings("ignore")

#os.environ["MISTRAL_API_KEY"] = "Vb3UFTvPq3sBIR2kJDwgDn59iFoYcZ7Q"
#os.environ["SERPER_API_KEY"] = "9b218beb8557de0554ba4710556e6978139bdc12"

#llm = LLM(
   # model="gemini-1.5-flash",   
    #api_base="https://generativelanguage.googleapis.com/v1beta/openai",  
   # api_key=os.getenv("GEMINI_API_KEY"),
    #temperature=0.2
#)
#tool = SerperDevTool()
#from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
#from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
import streamlit as st
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski, Draw, QED
import matplotlib.pyplot as plt
import math
from PIL import Image
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def compare_fingerprints_sy(smiles, radius=6, nBits=150):
    morgan_gen = GetMorganGenerator(radius=radius, fpSize=nBits)
    mol = Chem.MolFromSmiles(smiles)
    fp = morgan_gen.GetFingerprint(mol)
    return fp.ToBitString()

def dose_response(x, IC50):
    A1, A2, B = 0, 100, 1
    return A1 + (A2 - A1) / (1 + (x / IC50)**(-B))

def compare_fingerprints(smiles1, radius=6, nBits=150):
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    mol1 = Chem.MolFromSmiles(smiles1)
    
    if mol1 is None:
        # Return zero vector if SMILES is invalid
        return '0' * nBits
    
    fp1 = morgan_gen.GetFingerprint(mol1)
    bit_str1 = fp1.ToBitString()
    
    # Ensure exactly nBits length
    if len(bit_str1) != nBits:
        bit_str1 = bit_str1[:nBits] + '0' * (nBits - len(bit_str1))
    
    return bit_str1

class ClassificationModel(nn.Module):
    def __init__(self, input_dim=150, num_classes=3):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)  # No activation for logits
        return x
          
class CNNRegressionModel(nn.Module):
    def __init__(self, input_dim=150):
        super(CNNRegressionModel, self).__init__()
        # Assumes input is (batch_size, channels, input_dim), here channels=1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()

        # Fully connected layers
        self.fc1 = nn.Linear(64 * (input_dim // 8), 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#chnanged 29 july
def bit_string_to_tensor(bit_string, n_bits=150, device='cpu'):
    bit_list = [int(bit) for bit in bit_string]
    # Ensure we have exactly n_bits
    bit_list = bit_list[:n_bits] + [0] * (n_bits - len(bit_list))
    bit_tensor = torch.tensor(bit_list, dtype=torch.float32, device=device).unsqueeze(0)
    return bit_tensor


# Define the function to predict using the trained model
def predict_y(model, smiles1, n_bits=150):
    bit_string = compare_fingerprints(smiles1, nBits=n_bits)
    bit_tensor = bit_string_to_tensor(bit_string, n_bits)
    
    model.eval()
    with torch.no_grad():
        prediction = model(bit_tensor)
    
    return prediction.item()

#changed : 29-07-2025 :
def predict_cyp(model, smiles1, n_bits):
    device = next(model.parameters()).device  # detect model's device
    bit_string = compare_fingerprints(smiles1, nBits=n_bits)
    bit_tensor = bit_string_to_tensor(bit_string, n_bits).to(device)
    model.eval()
    with torch.no_grad():
        output = model(bit_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class




def bit_string_to_tensor_sy(bit_string1, bit_string2, n_bits=150):
    bit_list1 = [int(bit) for bit in bit_string1]
    bit_list2 = [int(bit) for bit in bit_string2]
    combined_bits = bit_list1 + bit_list2
    bit_tensor = torch.tensor(combined_bits, dtype=torch.float32).unsqueeze(0)
    return bit_tensor

def predict_sy(model, smiles1, smiles2, n_bits=150):
    bit_string1 = compare_fingerprints_sy(smiles1, nBits=n_bits)
    bit_string2 = compare_fingerprints_sy(smiles2, nBits=n_bits)
    bit_tensor = bit_string_to_tensor_sy(bit_string1, bit_string2, n_bits * 2)
    model_sy.eval()
    with torch.no_grad():
        prediction = model_sy(bit_tensor)
    return prediction.item()

model_sy= CNNRegressionModel(input_dim=300)
model = CNNRegressionModel(input_dim=150)
model_cyp = ClassificationModel(input_dim=150)
def lipinski_rule_of_five(mol):
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    h_donors = Lipinski.NumHDonors(mol)
    h_acceptors = Lipinski.NumHAcceptors(mol)
    violations = sum([mw > 500, logp > 5, h_donors > 5, h_acceptors > 10])
    return violations, violations <= 1

def water_solubility(mol):
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    mw = Descriptors.MolWt(mol)
    logS = -0.74 * logp - 0.006 * mw + 0.003 * tpsa + 0.63
    return logS

def synthetic_accessibility(mol):
    return 1 - QED.qed(mol)  # Lower means easier synthesis

def bioavailability_score(mol):
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    return int(tpsa <= 140 and rot_bonds <= 10)

def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        return img
    return None

def predict_adme(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {"Error": "Invalid SMILES"}

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    h_donors = Lipinski.NumHDonors(mol)
    h_acceptors = Lipinski.NumHAcceptors(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)

    lipinski_violations, lipinski_pass = lipinski_rule_of_five(mol)
    logS = water_solubility(mol)
    sa_score = synthetic_accessibility(mol)
    bio_score = bioavailability_score(mol)
    
    bbb_permeability = -0.3 < logp < 6 and mw < 450 and h_donors <= 3 and h_acceptors <= 7

    result = [
        ("Molecular Weight", mw, "Should be < 500 for good permeability"),
        ("logP", logp, "Measures hydrophobicity; affects absorption"),
        ("TPSA", tpsa, "Below 140 Å² favors permeability"),
        ("H-bond Donors", h_donors, "Should be ≤ 5 for drug-likeness"),
        ("H-bond Acceptors", h_acceptors, "Should be ≤ 10 for permeability"),
        ("Rotatable Bonds", rot_bonds, "Flexibility affects oral bioavailability"),
        ("Lipinski Violations", lipinski_violations, "≤ 1 violation preferred"),
        ("Lipinski Rule of Five Pass", lipinski_pass, "Indicates drug-likeness"),
        ("Water Solubility (LogS)", logS, "Lower LogS = better solubility"),
        ("Synthetic Accessibility", sa_score, "Lower value = easier synthesis"),
        ("Bioavailability Score", bio_score, "1 indicates good oral bioavailability"),
        ("BBB Permeability (Heuristic)", bbb_permeability, "Predicts CNS drug potential")
    ]
    return result


userchoice=["Name","SMILES"]
userchoice2=["Name","SMILES"]
userchoice3=["Name","SMILES"]
cell_line=["MCF7","7860","A549","DU145","HCT116","K562","OVCAR3","SNB75"]
st.set_page_config(page_title="💊DrugXplorer",page_icon=":pill")
st.markdown("# Drug:blue[X]plorer")


#modified
tab = st.sidebar.radio(
    "**Navigation**",
    ["🏠 Home", "🔬 ADME Analysis", "⚛️ Binding Affinity", "🧪 Drug Synergy", "📊 Generate Report"]
)
protein_groups = {
    "Nuclear Receptors": ["PPARD","PPARG", "AR", "ESR1", "NR3C1"],
    "Kinases & Cell Signaling": ["ABL1", "JAK2", "AKT1", "MAPK1", "PLK1","EGFR"],
    "Enzymes and Metabolic Targets": ["HMGCR", "PTGS2", "CYP3A4", "DPP4"],
    "Neurotransmitter and Neurological Targets": ["ADRB1", "ADORA2A", "DRD2", "ACHE", "BACE1"],
    "Cancer Therapeutic Targets": ["CASP3", "PARP1", "ROCK1","KDR"]
}
#newly added
def generate_report_data():
    """Store report data in session state"""
    if 'report_data' not in st.session_state:
        st.session_state.report_data = {
            'adme': None,
            'binding': None,
            'synergy': None,
            'timestamp': None
        }

if tab =="🏠 Home":
    st.markdown("#### :blue[*Welcome to DrugXplorer, your AI-powered companion for drug discovery*]")

    st.markdown(
    """
    <div style="text-align: justify;">
    
    :blue[**DrugXplorer**] is an advanced web application designed to streamline the drug discovery process by providing insights into key molecular properties. \n
    Whether you're a researcher, chemist, or biotechnology enthusiast, **DrugXplorer** enables you to predict crucial pharmacokinetic properties, assess drug-protein interactions, and analyze potential drug synergies—all in one platform.
    
    ### :blue[Features:]
    🔬 **ADME Prediction** - :gray[Evaluate the Absorption, Distribution, Metabolism, and Excretion (ADME) properties of drug-like molecules.]  
    🧬 **Binding Affinity Analysis** - :gray[Predict the binding strength between a drug molecule and various target proteins.]  
    💊 **Drug Synergy Prediction** - :gray[Analyze potential synergistic effects between drug combinations.]  
    
    :gray[Navigate through the different tabs to perform specific analyses:]
    - **ADME Properties**: :gray[Input your molecule's name or SMILES representation and obtain detailed ADME predictions.]
    - **Binding Affinity**: :gray[Select a target protein and provide a drug molecule to predict binding affinity values.]
    - **Drug Synergy**: :gray[Explore drug pair interactions and their potential for combination therapy.]
    
    **Harness the power of AI-driven drug discovery with DrugXplorer and accelerate your research with data-driven insights!** 
    
    </div>
    """, 
    unsafe_allow_html=True
)
if tab =="🔬 ADME Analysis":
    generate_report_data() #add this line
    st.subheader("ADME Analysis")
    st.markdown(
    """
    - ### **🔬Molecular Properties:**
        - **Molecular Weight (MW):** Measures the size of the molecule.  
        - **logP (Hydrophobicity):** Indicates lipid solubility; affects absorption.  
        - **Topological Polar Surface Area (TPSA):** Predicts permeability & solubility.  
        - **H-bond Donors:** Number of hydrogen bond donors in the molecule.  
        - **H-bond Acceptors:** Number of hydrogen bond acceptors in the molecule.  
        - **Rotatable Bonds:** Determines molecule flexibility; impacts oral bioavailability.
        - **Lipinski Violations:** Rules for drug-likeness (≤1 violation is preferred).
        - **Lipinski Rule of Five Pass:** Whether the molecule meets Lipinski’s criteria.    
        - **Water Solubility (LogS):** Predicts solubility; lower LogS = better solubility.  
        - **Synthetic Accessibility Score:** Estimates ease of synthesis (lower is better).
        - **Bioavailability Score:** Probability of good oral bioavailability.
        - **Blood-Brain Barrier (BBB) Permeability:** Predicts CNS drug potential.
    --- 
    - ### **💊Drug-Likeness & CYP Inhibition:**
        - **Lipinski Rule of Five Pass:** Evaluates overall drug-likeness.  
        - **Bioavailability Score:** Assesses potential for oral absorption.  
        - **Blood-Brain Barrier (BBB) Permeability:** Predicts CNS drug capability.
        - **CYP Inhibition:** Predicts whether molecule will inhibit CYP1A2, CYP2C9, CYP2C19, CYP2D6, and CYP34.
    --- 
    - ### **⚡How to Use This App**
        1. Enter drug name or its SMILES representation
        2. Click **Analyze** to get the ADME properties.
    """
    )
    user_ch=st.selectbox("Do you want to enter Name or SMILES representation",options=userchoice)
    if user_ch=="Name":
        drug_name= st.text_input("Enter molecule's name")
        def get_smiles(drug_name):
            try:
                compound = pcp.get_compounds(drug_name, 'name')[0]
                return compound.isomeric_smiles
            except:
                return 0
        smiles_ip=get_smiles(drug_name)
       
    elif user_ch=="SMILES":
        smiles_ip=st.text_input("Enter molecule's SMILES representation in capital letters")

    if st.button("Analyze"):
        if smiles_ip:
            result = predict_adme(smiles_ip)
            st.subheader("Molecular Structure")
            mol_img = draw_molecule(smiles_ip)
            if mol_img:
                st.image(mol_img, caption="Generated from SMILES", use_container_width=False)
            else:
                st.warning("Invalid SMILES provided. Please enter a valid SMILES string.")
            st.subheader("ADME Properties")
            df = pd.DataFrame(result, columns=["Property", "Value","Interpretation"])
            st.table(df)
            lipinski_pass = result[7][1]  # Extracts the boolean value (True/False)
            bioavailability_score = result[10][1]  # Extracts the score (1 or 0)
            bbb_permeability = result[11][1]  # Extracts the boolean value (True/False)

        # Additional Summary
       
            if lipinski_pass:
                st.success("✅ This molecule **passes** Lipinski's Rule of Five (drug-like).")
            else:
                st.warning("⚠️ This molecule **violates** Lipinski's Rule of Five.")

            if bioavailability_score:
                st.success("✅ This molecule **meets** Veber's bioavailability criteria.")
            else:
                st.warning("⚠️ This molecule **may have poor oral bioavailability**.")

            if bbb_permeability:
                st.success("✅ This molecule has **good potential** for Blood-Brain Barrier (BBB) permeability.")
            else:
                st.warning("⚠️ This molecule **may have limited** BBB permeability.")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            try:
                model_sy.load_state_dict(
                   torch.load(f"{cell}_MODEL.pth", map_location=torch.device("cpu"))
                )
                model_cyp.to(device)
                model_cyp.eval()
                predicted_1A2 = predict_cyp(model_cyp, smiles_ip, 150)  # Make sure smiles_ip is on device if tensor
            except Exception as e:
                st.error(f"Error predicting CYP1A2: {str(e)}")
                predicted_1A2 = 0

            try:
                model_cyp.load_state_dict(torch.load('CYP2C9_model_2.pth', map_location=device))
                model_cyp.to(device)
                model_cyp.eval()
                predicted_2C9 = predict_cyp(model_cyp, smiles_ip, 150)
            except Exception as e:
                st.error(f"Error predicting CYP2C9: {str(e)}")
                predicted_2C9 = 0

            try:
                model_cyp.load_state_dict(torch.load('CYP2C19_model_2.pth', map_location=device))
                model_cyp.to(device)
                model_cyp.eval()
                predicted_2C19 = predict_cyp(model_cyp, smiles_ip, 150)
            except Exception as e:
                st.error(f"Error predicting CYP2C19: {str(e)}")
                predicted_2C19 = 0

            try:
                model_cyp.load_state_dict(torch.load('CYP2D6_model_2.pth', map_location=device))
                model_cyp.to(device)
                model_cyp.eval()
                predicted_2D6 = predict_cyp(model_cyp, smiles_ip, 150)
            except Exception as e:
                st.error(f"Error predicting CYP1A2: {str(e)}")
                predicted_2D6 = 0

            try:
                model_cyp.load_state_dict(torch.load('CYP3A4_model_2.pth', map_location=device))
                model_cyp.to(device)
                model_cyp.eval()
                predicted_3A4 = predict_cyp(model_cyp, smiles_ip, 150)
            except Exception as e:
                st.error(f"Error predicting CYP3A4: {str(e)}")
                predicted_3A4 = 0

            if predicted_1A2 == 1:
                st.warning("⚠️ This molecule is an inhibitor of CYP1A2.")
            else:
                st.success("✅ This molecule is not an inhibitor of CYP1A2.")
            if predicted_2C9 == 1:
                st.warning("⚠️ This molecule is an inhibitor of CYP2C9.")
            else:
                st.success("✅ This molecule is not an inhibitor of CYP2C9.")
            if predicted_2C19 == 1:
                st.warning("⚠️ This molecule is an inhibitor of CYP2C19.")
            else:
                st.success("✅ This molecule is not an inhibitor of CYP2C19.")
            if predicted_2D6 == 1:
                st.warning("⚠️ This molecule is an inhibitor of CYP2D6.")
            else:
                st.success("✅ This molecule is not an inhibitor of CYP2D6.")
            if predicted_3A4 == 1:
                st.warning("⚠️ This molecule is an inhibitor of CYP3A4.")
            else:
                st.success("✅ This molecule is not an inhibitor of CYP3A4.")
            
            # ADD THIS BLOCK HERE ⬇️
            st.session_state.report_data['adme'] = {
                'smiles': smiles_ip,
                'drug_name': drug_name if user_ch == "Name" else "SMILES Input",
                'properties': result,
                'mol_img': mol_img,
                'lipinski_pass': lipinski_pass,
                'bioavailability_score': bioavailability_score,
                'bbb_permeability': bbb_permeability,
                'cyp_predictions': {
                    'CYP1A2': predicted_1A2,
                    'CYP2C9': predicted_2C9,
                    'CYP2C19': predicted_2C19,
                    'CYP2D6': predicted_2D6,
                    'CYP3A4': predicted_3A4
                },
                'inference': None,  # Will be updated after AI inference
                'timestamp': datetime.datetime.now()
            }
            # ADD ABOVE THIS BLOCK ⬆️
            '''
            with st.spinner("Processing... Please wait."):
                inferencer = Agent(
                    role="Drug Discovery Scientist",
                    goal="From a given output of a prediction or experiment, provide valuable insights and inferences",
                    backstory = "You give inferences and context from a given molecular property prediction",
                    verbose = True,
                    llm = llm,
                    #tool = [tool]
                )
                give_insight = Task(
                    description="""Write inference using bullet points and subheaders for Absorption, Distribution, Metabolism and Excretion for 
                    the following result: 
                    Molecular Weight: {mw}
                    Hydrophobicity: {logp}
                    Topological Polar Surface Area (TPSA):{tpsa}
                    Number of H-Donors: {hd}
                    Number of H-Acceptors: {ha}
                    Number of Rotatable bonds: {rb}
                    Number of Lipinski Violation: {lv}
                    Solubility(Log S): {sol}
                    Synthetic Accessibility: {sas}
                    Bioavailability  is true/false: {bio}
                    Blood - Brain Barrier Permeability is true/false: {bbb}
                    CYP1A2 inhibhitor (True:1, False:0):{1a2}
                    CYP2C9 inhibhitor (True:1, False:0):{2c9}
                    CYP2C19 inhibhitor (True:1, False:0):{2c19}
                    CYP2D6 inhibhitor (True:1, False:0):{2d6}
                    CYP3A4 inhibhitor (True:1, False:0):{3a4}
                    Provide inferences and context only, no supporting references""",
                    expected_output="Well-written, fact-based paragraph in bullet points from credible resources providing context to the given inputs",
                    agent=inferencer,
                )
                st.session_state.report_data['adme'] = { #newly added
                'smiles': smiles_ip,
                'drug_name': drug_name if user_ch == "Name" else "SMILES Input",
                'properties': result,
                'mol_img': mol_img,
                'lipinski_pass': lipinski_pass,
                'bioavailability_score': bioavailability_score,
                #'bbb_permeability': bbb_permeability,
                'cyp_predictions': {
                    'CYP1A2': predicted_1A2,
                    'CYP2C9': predicted_2C9,
                    'CYP2C19': predicted_2C19,
                    'CYP2D6': predicted_2D6,
                    'CYP3A4': predicted_3A4
                },
                'inference': give_insight.output if 'give_insight' in locals() else None,
                'timestamp': datetime.datetime.now()
            }

            
                crew = Crew(
                    agents=[inferencer],
                    tasks=[give_insight],
                    verbose=False
                )
                result = crew.kickoff(inputs={"mw":result[0][1], "logp":result[1][1],"tpsa":result[2][1],"hd":result[3][1],"ha":result[4][1],"rb":result[5][1], "lv":result[6][1],"sol":result[8][1],"sas":result[9][1],
                                              "bio":result[10][1],"bbb":result[11][1],
                                              "1a2":predicted_1A2,"2c9":predicted_2C9,"2c19":predicted_2C19,"2d6":predicted_2D6,"3a4":predicted_3A4})
                st.subheader("Inference")
                st.markdown(give_insight.output)
                # Update inference in report data
                if 'report_data' in st.session_state and st.session_state.report_data['adme']:
                    st.session_state.report_data['adme']['inference'] = give_insight.output
'''

if tab=="⚛️ Binding Affinity":
    generate_report_data()  # ADD THIS LINE
    st.subheader("Binding Affinity")
    st.markdown("""

- ### **🧬Protein Groups**

    **Nuclear Receptors (Hormone-Responsive)**
    - **PPARD** – Peroxisome Proliferator-Activated Receptor Delta.
    - **PPARG** – Peroxisome Proliferator-Activated Receptor Gamma.
    - **AR** – Androgen receptor
    - **ESR1** – Estrogen Receptor Alpha
    - **NR3C1** – Glucocorticoid receptor

    **Kinases & Cell Signaling**
    - **ABL1** – Abelson Murine Leukemia Viral Oncogene Homolog 1
    - **JAK2** – Janus Kinase 2
    - **AKT1** – AKT Serine/Threonine Kinase 1
    - **MAPK1** – Mitogen-Activated Protein Kinase 1
    - **PLK1** – Polo-Like Kinase 1
    - **EGFR** – Epidermal Growth Factor Receptor
                
    **Enzymes and Metabolic Targets**
    - **HMGCR** – 3-Hydroxy-3-Methylglutaryl-CoA Reductase
    - **PTGS2** – Prostaglandin-Endoperoxide Synthase 2 (COX-2)
    - **CYP3A4** – Cytochrome P450 3A4
    - **DPP4** – Dipeptidyl Peptidase 4

    **Neurotransmitter and Neurological Targets**
    - **ADRB1** – Beta-1 Adrenergic Receptor
    - **ADORA2A** – Adenosine A2A Receptor
    - **DRD2** – Dopamine Receptor D2
    - **ACHE** – Acetylcholinesterase
    - **BACE1** – Beta-Site Amyloid Precursor Protein-Cleaving Enzyme 1

    **Cancer Therapeutic Targets**
    - **CASP3** – Caspase-3
    - **PARP1** – Poly (ADP-Ribose) Polymerase 1
    - **ROCK1** – Rho-Associated Protein Kinase 1
    - **KDR** – Kinase Insert Domain Receptor (VEGFR-2)
---

- ### **📊How to interpret results**
        
    **Binding Affinity (ΔG)**
    Binding affinity (**ΔG**, in kcal/mol) represents how strongly a drug binds to a target protein.

    | **Binding Affinity (ΔG)** | **Binding Strength** |
    |--------------------------|--------------------|
    | **ΔG ≤ -10 kcal/mol** | Very strong binding |
    | **-10 < ΔG ≤ -8 kcal/mol** | Strong binding |
    | **-8 < ΔG ≤ -6 kcal/mol** | Moderate binding |
    | **-6 < ΔG ≤ -4 kcal/mol** | Weak binding |
    | **ΔG > -4 kcal/mol** | Very weak/no binding |

    **Equilibrium Constant (K)**
    The equilibrium constant (**K** in µM) represents the ratio of bound and unbound states of a drug-protein interaction:

    | **Kd Value (µM)** | **Biological Interpretation** |
    |------------------|------------------------------------|
    | **< 0.001 µM** | Likely irreversible inhibition or very strong target modulation |
    | **0.001 – 0.1 µM** | Highly potent modulator, strong effect at low concentrations |
    | **0.1 – 1 µM** | Effective modulation, commonly seen in drug candidates |
    | **1 – 10 µM** | Moderate effect, may require optimization for potency |
    | **10 – 100 µM** | Weak modulation, may be non-specific or require high doses |
    | **> 100 µM** | Very weak or no modulation, likely not effective |

---                
- ### **⚡How to Use This App**
    1. Enter drug name or its SMILES representation
    2. Select a protein group from the dropdown menu.
    3. Choose a target protein within the selected group.
    4. Click **Predict** to get the predicted interaction strength.

""")

    user_ch_2=st.selectbox("Do you want to enter Name or SMILES representation of a molecule",options=userchoice2)
    if user_ch_2=="Name":
        drug_name_2= st.text_input("Enter name of the molecule")
        def get_smiles(drug_name_2):
            try:
                compound = pcp.get_compounds(drug_name_2, 'name')[0]
                return compound.isomeric_smiles
            except:
                return 0
        smiles_ip_2=get_smiles(drug_name_2)
       
    elif user_ch_2=="SMILES":
        smiles_ip_2=st.text_input("Enter molecule's SMILES representation in capital")
    
    if smiles_ip_2:
        group = st.selectbox("Select Protein Group", list(protein_groups.keys()))
        if group:
            protein = st.selectbox("Select Target Protein", protein_groups[group])

    if smiles_ip_2:
        if st.button("Predict"):
            model.load_state_dict(torch.load(f"{protein}_model_best.pth"))
            model.eval()
            predicted_y = predict_y(model, smiles_ip_2)
            integer_value = round(predicted_y,2)
            del_g=str(integer_value)
            K=integer_value*4184
            K=K/(298*8.314)
            K=np.exp(K)
            K=K * 10**6
            K=round(K,4)
            eqb=str(K)
            L=" µM"
            cal= " kcal/mol"
            del_g=del_g+cal
            eqb=eqb+L
            st.write("Binding affinity of your drug molecule with selected protein is")
            st.write(del_g)
            st.write("Amount of drug in micromolar needed to modulate selected protein is")
            st.write(eqb)
            concentration = np.logspace(-2, 2, 1000)
            inhibition = dose_response(concentration, K)
            st.title('Dose-Response Curve')
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.semilogx(concentration, inhibition, 'g-', linewidth=2, label=f'Compound Kd = {K:.3f} µM)')
            y_K = dose_response(K, K)
            ax.plot(K, y_K, 'mo', markersize=8, label='_nolegend_')
            ax.axvline(K, color='m', linestyle='--', linewidth=1)
            ax.set_xlabel('Drug Concentration (µM)')
            ax.set_ylabel('% Inhibition')
            ax.set_title('Dose-Response Curve')
            ax.legend(loc='best')
            ax.grid(True)
            ax.set_xlim([min(concentration), max(concentration)])
            ax.set_ylim([0, 100])
            st.pyplot(fig)
            st.session_state.report_data['binding'] = {
                'smiles': smiles_ip_2,
                'drug_name': drug_name_2 if user_ch_2 == "Name" else "SMILES Input",
                'protein_group': group,
                'protein': protein,
                'binding_affinity': del_g,
                'equilibrium_constant': eqb,
                'kd_value': K,
                #'inference': give_insight.output if 'give_insight' in locals() else None,
                'inference': None,  # No AI inference needed
                'timestamp': datetime.datetime.now()
            }
            '''
            with st.spinner("Processing... Please wait."):
                inferencer = Agent(
                    role="Drug Discovery Scientist",
                    goal="From a given output of a prediction or experiment, provide valuable insights and inferences relating to the experiment",
                    backstory = "You give inferences and context from a given molecular property prediction",
                    verbose = True,
                    llm = llm,
                    #tool = [tool]
                )
                give_insight = Task(
                    description="""Write inference using bullet ponts and subheaders for the following result: A Molecule with SMILES: {smiles_ip_2} 
                    when docked with a protein {protein} is predicted to generate a binding affinity of {del_g} and Equillibrium constant value of {eqb}
                    Provide inferences and context only, no supporting references""",
                    expected_output="Well-written, fact-based paragraph in bullet points from credible resources providing context to the given inputs",
                    agent=inferencer,
                )
                crew = Crew(
                    agents=[inferencer],
                    tasks=[give_insight],
                    verbose=False
                )
                result = crew.kickoff(inputs={"smiles_ip_2":smiles_ip_2 , "protein": protein,"del_g": del_g, "eqb":eqb})
                st.subheader("Inference")
                st.markdown(give_insight.output)
                # Update inference in report data
                if 'report_data' in st.session_state and st.session_state.report_data['binding']:
                    st.session_state.report_data['binding']['inference'] = give_insight.output
'''
if tab=="🧪 Drug Synergy":
    generate_report_data()  # ADD THIS LINE
    st.subheader("Drug Synergy Prediction")
    st.markdown("""  

- ### **🧪Cell Lines**  

    **Cell Lines & Cancer Types**  
    - **MCF7** – Breast cancer (ER⁺, Luminal A).  
    - **A549** – Lung adenocarcinoma (NSCLC).  
    - **HCT116** – Colorectal carcinoma.  
    - **DU145** – Prostate cancer (androgen-independent).  
    - **K562** – Chronic myelogenous leukemia (CML).  
    - **OVCAR3** – Ovarian adenocarcinoma.  
    - **SNB75** – Glioblastoma (brain tumor).   
    - **786-O** – Renal cell carcinoma (RCC, kidney cancer).  

---  

- ### **📊 How to Interpret Bliss Synergy Score**  

    The **Bliss Synergy Score** quantifies the interaction between two drugs compared to their expected independent effects.  

    | **Bliss Synergy Score** | **Interpretation** |  
    |-------------------------|--------------------|  
    | **> 1**  | Synergistic – Drugs work significantly better together than expected. |   
    | **1 to -1**  | Additive – Drugs work as expected without interaction. |  
    | **< -1**  | Antagonistic – Drug combination reduces effectiveness. |  

---  

- ### **⚡ How to Use This App**  

    1. Select two drugs from the input list or enter SMILES.  
    2. Choose a cancer cell line from the dropdown menu.  
    3. Click **Predict** to calculate the **Bliss Synergy Score**.    

""")
    user_ch_3=st.selectbox("Do you want to enter the Name or SMILES representation",options=userchoice3)
    if user_ch_3=="Name":
        drug_name_sy1= st.text_input("Enter name of the first molecule")
        drug_name_sy2=st.text_input("Enter name of the second molecule")
        def get_smiles(drug_name):
            try:
                compound = pcp.get_compounds(drug_name, 'name')[0]
                return compound.isomeric_smiles
            except:
                return 0
        smiles_sy_1=get_smiles(drug_name_sy1)
        smiles_sy_2=get_smiles(drug_name_sy2)
       
    elif user_ch_3=="SMILES":
        smiles_sy_1=st.text_input("Enter first molecule's SMILES representation in capital letters")
        smiles_sy_2=st.text_input("Enter second molecule's SMILES representation in capital letters")
        
    if smiles_sy_1 and smiles_sy_2:
        cell=st.selectbox("Choose your desired cell line",options=cell_line)
    if smiles_sy_1 and smiles_sy_2:
        if st.button("Predict Synergy"):
            model_sy.load_state_dict(torch.load(f"{cell}_MODEL.pth", map_location = torch.device("cpu")))
            model_sy.eval()
            predicted_sy = predict_sy(model_sy, smiles_sy_1, smiles_sy_2)
            synergy_value = round(predicted_sy,2)
            bliss_score=str(synergy_value)
            st.write("Bliss score of the two molecules with desired cell line is")
            st.write(bliss_score)
            st.session_state.report_data['synergy'] = {
                'smiles_1': smiles_sy_1,
                'smiles_2': smiles_sy_2,
                'drug_name_1': drug_name_sy1 if user_ch_3 == "Name" else "SMILES Input 1",
                'drug_name_2': drug_name_sy2 if user_ch_3 == "Name" else "SMILES Input 2",
                'cell_line': cell,
                'bliss_score': bliss_score,
                'synergy_value': synergy_value,
                #'inference': give_insight.output if 'give_insight' in locals() else None,
                'inference': None,  # No AI inference needed
                'timestamp': datetime.datetime.now()
            }
            '''
            with st.spinner("Processing... Please wait."):
                inferencer = Agent(
                    role="Drug Discovery Scientist",
                    goal="From a given output of a prediction or experiment, provide valuable insights and inferences relating to the experiment",
                    backstory = "You give inferences and context from a given molecular property prediction",
                    verbose = True,
                    llm = llm,
                    #tool = [tool]
                )
                give_insight = Task(
                    description="""Write inference using bullet points for the following result: 
                                The Bliss Drug Synergy score for two drugs of SMILES {d1} and {d2} 
                                on a cell line {cl} is predicted to be {value}. Keep in mind that
                                a bliss score less than -1 is considered antagonistic, between -1 and 1 is considered additive
                                ,and a score higher than 1 is considered synergistic.
                                Provide inferences and context only, no supporting references""",
                    expected_output="Well-written, fact-based paragraph in bullet points from credible resources providing context to the given inputs",
                    agent=inferencer,
                )
                crew = Crew(
                    agents=[inferencer],
                    tasks=[give_insight],
                    verbose=False
                )
                result = crew.kickoff(inputs={"d1":smiles_sy_1 , "d2": smiles_sy_2,"cl": cell, "value":bliss_score})
                st.subheader("Inference")
                st.markdown(give_insight.output)
                # Update inference in report data
                if 'report_data' in st.session_state and st.session_state.report_data['synergy']:
                    st.session_state.report_data['synergy']['inference'] = give_insight.output
'''
if tab == "📊 Generate Report":
    st.subheader("Generate Comprehensive Report")
    st.markdown("""
    Generate a comprehensive report containing all your analysis results from ADME, Binding Affinity, and Drug Synergy predictions.
    
    **Note:** You need to run analyses in other tabs first to generate data for the report.
    """)
    
    # Initialize report data
    generate_report_data()
    
    # Check if there's any data to report
    has_data = any([
        st.session_state.report_data['adme'],
        st.session_state.report_data['binding'],
        st.session_state.report_data['synergy']
    ])
    
    if not has_data:
        st.warning("⚠️ No analysis data found. Please run analyses in other tabs first.")
        st.info("💡 Navigate to ADME Analysis, Binding Affinity, or Drug Synergy tabs to generate data for your report.")
    else:
        if st.button("Generate Report", type="primary"):
            # Function to generate report text for download
            def generate_report_text():
                report_content = []
                
                # Header
                report_content.append("DRUGXPLORER - COMPREHENSIVE DRUG DISCOVERY REPORT")
                report_content.append("=" * 60)
                report_content.append(f"Generated on: {datetime.datetime.now().strftime('%B %d, %Y at %H:%M:%S')}")
                report_content.append("\n")
                
                # ADME Analysis Section
                if st.session_state.report_data['adme']:
                    adme_data = st.session_state.report_data['adme']
                    report_content.append("ADME ANALYSIS RESULTS")
                    report_content.append("-" * 30)
                    report_content.append(f"Drug Name: {adme_data['drug_name']}")
                    report_content.append(f"SMILES: {adme_data['smiles']}")
                    report_content.append(f"Analysis Date: {adme_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    report_content.append("\nMolecular Properties:")
                    
                    for prop in adme_data['properties']:
                        report_content.append(f"  {prop[0]}: {prop[1]} ({prop[2]})")
                    
                    report_content.append(f"\nLipinski's Rule of Five: {'PASS' if adme_data['lipinski_pass'] else 'FAIL'}")
                    report_content.append(f"Bioavailability Score: {'PASS' if adme_data['bioavailability_score'] else 'FAIL'}")
                    report_content.append(f"BBB Permeability: {'GOOD' if adme_data['bbb_permeability'] else 'LIMITED'}")
                    
                    report_content.append("\nCYP Enzyme Inhibition Predictions:")
                    for enzyme, prediction in adme_data['cyp_predictions'].items():
                        status = "Inhibitor" if prediction == 1 else "Non-inhibitor"
                        report_content.append(f"  {enzyme}: {status}")
                    
                    if adme_data['inference']:
                        report_content.append(f"\nAI-Generated Insights:\n{adme_data['inference']}")
                    report_content.append("\n")
                
                # Binding Affinity Section
                if st.session_state.report_data['binding']:
                    binding_data = st.session_state.report_data['binding']
                    report_content.append("BINDING AFFINITY RESULTS")
                    report_content.append("-" * 30)
                    report_content.append(f"Drug Name: {binding_data['drug_name']}")
                    report_content.append(f"SMILES: {binding_data['smiles']}")
                    report_content.append(f"Target Protein: {binding_data['protein']} ({binding_data['protein_group']})")
                    report_content.append(f"Analysis Date: {binding_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    report_content.append(f"Binding Affinity (ΔG): {binding_data['binding_affinity']}")
                    report_content.append(f"Equilibrium Constant (Kd): {binding_data['equilibrium_constant']}")
                    
                    # Binding Strength
                    kd_val = binding_data['kd_value']
                    if kd_val <= 0.001:
                        strength = "Very Strong Binding"
                    elif kd_val <= 0.1:
                        strength = "Strong Binding"
                    elif kd_val <= 1:
                        strength = "Moderate Binding"
                    elif kd_val <= 10:
                        strength = "Weak Binding"
                    else:
                        strength = "Very Weak Binding"
                    
                    report_content.append(f"Binding Strength: {strength}")
                    
                    if binding_data['inference']:
                        report_content.append(f"\nAI-Generated Insights:\n{binding_data['inference']}")
                    report_content.append("\n")
                
                # Drug Synergy Section
                if st.session_state.report_data['synergy']:
                    synergy_data = st.session_state.report_data['synergy']
                    report_content.append("DRUG SYNERGY RESULTS")
                    report_content.append("-" * 30)
                    report_content.append(f"Drug 1: {synergy_data['drug_name_1']}")
                    report_content.append(f"SMILES 1: {synergy_data['smiles_1']}")
                    report_content.append(f"Drug 2: {synergy_data['drug_name_2']}")
                    report_content.append(f"SMILES 2: {synergy_data['smiles_2']}")
                    report_content.append(f"Cell Line: {synergy_data['cell_line']}")
                    report_content.append(f"Analysis Date: {synergy_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    report_content.append(f"Bliss Synergy Score: {synergy_data['bliss_score']}")
                    
                    # Synergy Interpretation
                    synergy_val = synergy_data['synergy_value']
                    if synergy_val > 1:
                        interpretation = "SYNERGISTIC - Drugs work significantly better together"
                    elif synergy_val >= -1:
                        interpretation = "ADDITIVE - Drugs work as expected without interaction"
                    else:
                        interpretation = "ANTAGONISTIC - Drug combination reduces effectiveness"
                    
                    report_content.append(f"Synergy Type: {interpretation}")
                    
                    if synergy_data['inference']:
                        report_content.append(f"\nAI-Generated Insights:\n{synergy_data['inference']}")
                    report_content.append("\n")
                
                # Footer
                report_content.append("REPORT SUMMARY")
                report_content.append("-" * 30)
                report_content.append(f"Report generated by DrugXplorer on {datetime.datetime.now().strftime('%B %d, %Y')}")
                report_content.append("\nAnalyses Included:")
                report_content.append(f"- ADME Analysis: {'YES' if st.session_state.report_data['adme'] else 'NO'}")
                report_content.append(f"- Binding Affinity: {'YES' if st.session_state.report_data['binding'] else 'NO'}")
                report_content.append(f"- Drug Synergy: {'YES' if st.session_state.report_data['synergy'] else 'NO'}")
                report_content.append("\nDrugXplorer - AI-Powered Drug Discovery Platform")
                
                return "\n".join(report_content)
            
            # Report Header with Logo
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Try to display logo if it exists
                try:
                    st.image("assets/logo.png", width=200)
                except:
                    st.markdown("### 🧬")  # Fallback if logo doesn't exist
                
                st.markdown("# Drug:blue[X]plorer")
                st.markdown("### Comprehensive Drug Discovery Report")
                st.markdown(f"**Generated on:** {datetime.datetime.now().strftime('%B %d, %Y at %H:%M:%S')}")
            
            st.markdown("---")
            
            # ADME Analysis Section
            if st.session_state.report_data['adme']:
                adme_data = st.session_state.report_data['adme']
                st.markdown("## 🔬 ADME Analysis Results")
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    if adme_data['mol_img']:
                        st.image(adme_data['mol_img'], caption="Molecular Structure", width=250)
                
                with col2:
                    st.markdown(f"**Drug Name:** {adme_data['drug_name']}")
                    st.markdown(f"**SMILES:** `{adme_data['smiles']}`")
                    st.markdown(f"**Analysis Date:** {adme_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Properties Table
                st.markdown("### Molecular Properties")
                df_adme = pd.DataFrame(adme_data['properties'], columns=["Property", "Value", "Interpretation"])
                st.table(df_adme)
                
                # Summary
                st.markdown("### Summary")
                if adme_data['lipinski_pass']:
                    st.success("✅ Passes Lipinski's Rule of Five (drug-like)")
                else:
                    st.error("❌ Violates Lipinski's Rule of Five")
                
                if adme_data['bioavailability_score']:
                    st.success("✅ Meets Veber's bioavailability criteria")
                else:
                    st.error("❌ May have poor oral bioavailability")
                
                if adme_data['bbb_permeability']:
                    st.success("✅ Good potential for Blood-Brain Barrier permeability")
                else:
                    st.warning("⚠️ Limited BBB permeability")
                
                # CYP Predictions
                st.markdown("### CYP Enzyme Inhibition Predictions")
                cyp_results = []
                for enzyme, prediction in adme_data['cyp_predictions'].items():
                    status = "Inhibitor" if prediction == 1 else "Non-inhibitor"
                    cyp_results.append([enzyme, status])
                
                df_cyp = pd.DataFrame(cyp_results, columns=["CYP Enzyme", "Prediction"])
                st.table(df_cyp)
                
                if adme_data['inference']:
                    st.markdown("### AI-Generated Insights")
                    st.markdown(adme_data['inference'])
                
                st.markdown("---")
            
            # Binding Affinity Section
            if st.session_state.report_data['binding']:
                binding_data = st.session_state.report_data['binding']
                st.markdown("## ⚛️ Binding Affinity Results")
                
                st.markdown(f"**Drug Name:** {binding_data['drug_name']}")
                st.markdown(f"**SMILES:** `{binding_data['smiles']}`")
                st.markdown(f"**Target Protein:** {binding_data['protein']} ({binding_data['protein_group']})")
                st.markdown(f"**Analysis Date:** {binding_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Binding Affinity (ΔG)", binding_data['binding_affinity'])
                with col2:
                    st.metric("Equilibrium Constant (Kd)", binding_data['equilibrium_constant'])
                
                # Binding Strength Interpretation
                kd_val = binding_data['kd_value']
                if kd_val <= 0.001:
                    strength = "Very Strong Binding"
                    color = "success"
                elif kd_val <= 0.1:
                    strength = "Strong Binding"
                    color = "success"
                elif kd_val <= 1:
                    strength = "Moderate Binding"
                    color = "info"
                elif kd_val <= 10:
                    strength = "Weak Binding"
                    color = "warning"
                else:
                    strength = "Very Weak Binding"
                    color = "error"
                
                if color == "success":
                    st.success(f"**Binding Strength:** {strength}")
                elif color == "info":
                    st.info(f"**Binding Strength:** {strength}")
                elif color == "warning":
                    st.warning(f"**Binding Strength:** {strength}")
                else:
                    st.error(f"**Binding Strength:** {strength}")
                
                if binding_data['inference']:
                    st.markdown("### AI-Generated Insights")
                    st.markdown(binding_data['inference'])
                
                st.markdown("---")
            
            # Drug Synergy Section
            if st.session_state.report_data['synergy']:
                synergy_data = st.session_state.report_data['synergy']
                st.markdown("## 🧪 Drug Synergy Results")
                
                st.markdown(f"**Drug 1:** {synergy_data['drug_name_1']}")
                st.markdown(f"**SMILES 1:** `{synergy_data['smiles_1']}`")
                st.markdown(f"**Drug 2:** {synergy_data['drug_name_2']}")
                st.markdown(f"**SMILES 2:** `{synergy_data['smiles_2']}`")
                st.markdown(f"**Cell Line:** {synergy_data['cell_line']}")
                st.markdown(f"**Analysis Date:** {synergy_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                st.metric("Bliss Synergy Score", synergy_data['bliss_score'])
                
                # Synergy Interpretation
                synergy_val = synergy_data['synergy_value']
                if synergy_val > 1:
                    st.success("✅ **SYNERGISTIC** - Drugs work significantly better together")
                elif synergy_val >= -1:
                    st.info("ℹ️ **ADDITIVE** - Drugs work as expected without interaction")
                else:
                    st.error("❌ **ANTAGONISTIC** - Drug combination reduces effectiveness")
                
                if synergy_data['inference']:
                    st.markdown("### AI-Generated Insights")
                    st.markdown(synergy_data['inference'])
                
                st.markdown("---")
            
            # Report Footer
            st.markdown("## Report Summary")
            st.markdown(f"""
            This comprehensive report was generated by **DrugXplorer** on {datetime.datetime.now().strftime('%B %d, %Y')}.
            
            **Analyses Included:**
            - ADME Analysis: {'✅' if st.session_state.report_data['adme'] else '❌'}
            - Binding Affinity: {'✅' if st.session_state.report_data['binding'] else '❌'}
            - Drug Synergy: {'✅' if st.session_state.report_data['synergy'] else '❌'}
            
            *DrugXplorer - AI-Powered Drug Discovery Platform*
            """)
            
            # Enhanced Download Section
            st.markdown("### 💾 Download Report")
            
            # Generate report content for download
            report_text = generate_report_text()
            
            # Create columns for different download options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Text file download
                st.download_button(
                    label="📄 Download as TXT",
                    data=report_text,
                    file_name=f"DrugXplorer_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    help="Download the report as a plain text file"
                )
            
            with col2:
                # CSV download for tabular data
                if st.session_state.report_data['adme'] or st.session_state.report_data['binding'] or st.session_state.report_data['synergy']:
                    # Create summary CSV
                    csv_data = []
                    
                    if st.session_state.report_data['adme']:
                        adme_data = st.session_state.report_data['adme']
                        csv_data.append({
                            'Analysis_Type': 'ADME',
                            'Drug_Name': adme_data['drug_name'],
                            'SMILES': adme_data['smiles'],
                            'Timestamp': adme_data['timestamp'],
                            'Lipinski_Pass': adme_data['lipinski_pass'],
                            'Bioavailability_Score': adme_data['bioavailability_score'],
                            'BBB_Permeability': adme_data['bbb_permeability']
                        })
                    
                    if st.session_state.report_data['binding']:
                        binding_data = st.session_state.report_data['binding']
                        csv_data.append({
                            'Analysis_Type': 'Binding_Affinity',
                            'Drug_Name': binding_data['drug_name'],
                            'SMILES': binding_data['smiles'],
                            'Target_Protein': binding_data['protein'],
                            'Binding_Affinity': binding_data['binding_affinity'],
                            'Kd_Value': binding_data['kd_value'],
                            'Timestamp': binding_data['timestamp']
                        })
                    
                    if st.session_state.report_data['synergy']:
                        synergy_data = st.session_state.report_data['synergy']
                        csv_data.append({
                            'Analysis_Type': 'Drug_Synergy',
                            'Drug_1': synergy_data['drug_name_1'],
                            'Drug_2': synergy_data['drug_name_2'],
                            'Cell_Line': synergy_data['cell_line'],
                            'Bliss_Score': synergy_data['bliss_score'],
                            'Synergy_Value': synergy_data['synergy_value'],
                            'Timestamp': synergy_data['timestamp']
                        })
                    
                    df_summary = pd.DataFrame(csv_data)
                    csv_string = df_summary.to_csv(index=False)
                    
                    st.download_button(
                        label="📊 Download as CSV",
                        data=csv_string,
                        file_name=f"DrugXplorer_Data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download summary data as CSV file"
                    )
            
            with col3:
                # JSON download for structured data
                import json
                
                json_data = {
                    "report_metadata": {
                        "generated_on": datetime.datetime.now().isoformat(),
                        "platform": "DrugXplorer",
                        "version": "1.0"
                    },
                    "analyses": {}
                }
                
                if st.session_state.report_data['adme']:
                    adme_data = st.session_state.report_data['adme'].copy()
                    adme_data['timestamp'] = adme_data['timestamp'].isoformat()
                    json_data["analyses"]["adme"] = adme_data
                
                if st.session_state.report_data['binding']:
                    binding_data = st.session_state.report_data['binding'].copy()
                    binding_data['timestamp'] = binding_data['timestamp'].isoformat()
                    json_data["analyses"]["binding_affinity"] = binding_data
                
                if st.session_state.report_data['synergy']:
                    synergy_data = st.session_state.report_data['synergy'].copy()
                    synergy_data['timestamp'] = synergy_data['timestamp'].isoformat()
                    json_data["analyses"]["drug_synergy"] = synergy_data
                
                json_string = json.dumps(json_data, indent=2, default=str)
                
                st.download_button(
                    label="🔧 Download as JSON",
                    data=json_string,
                    file_name=f"DrugXplorer_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="Download structured data as JSON file"
                )
            
            # Additional info
            st.info("💡 **Tip:** You can also save this page as PDF using your browser's print function (Ctrl+P → Save as PDF)")

##bg image



# Simple Animated Dark Blue Gradient Background
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(-45deg, #0c1629, #1e3a5f, #0f2847, #1a365d, #2563eb, #1e40af);
        background-size: 600% 600%;
        animation: gradientShift 12s ease-in-out infinite;
    }
    
    @keyframes gradientShift {
        0% {
            background-position: 0% 50%;
        }
        25% {
            background-position: 100% 0%;
        }
        50% {
            background-position: 100% 100%;
        }
        75% {
            background-position: 0% 100%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
</style>
""", unsafe_allow_html=True)
