from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.py_linac.lattice_modifications import Replace_BaseRF_Gap_and_Quads_to_Overlapping_Nodes
from orbit.py_linac.linac_parsers import SNS_LinacLatticeFactory
from orbit.py_linac.overlapping_fields import SNS_EngeFunctionFactory


# Create SNS linac lattice.
xml_filename = "inputs/sns_linac.xml"
sequences = [
    "MEBT",
    "DTL1",
    "DTL2",
    "DTL3",
    "DTL4",
    "DTL5",
    "DTL6",
    "CCL1",
    "CCL2",
    "CCL3",
    "CCL4",
    "SCLMed",
    "SCLHigh",
    "HEBT1",
    "HEBT2",
]
sns_linac_factory = SNS_LinacLatticeFactory()
lattice = sns_linac_factory.getLinacAccLattice(sequences, xml_filename)

for gap_node in lattice.getRF_Gaps():
    print("name={}, EzFile={}".format(gap_node.getName(), gap_node.getParam("EzFile")))

# Replace hard-edge quads with soft-edge quads; replace zero-length RF gap models
# with field-on-axis RF gap models. Can be used for any sequences, no limitations.
fields_dir = "inputs/sns_rf_fields/"
z_step = 0.002
sequences = [seq for seq in sequences if seq not in ["HEBT1", "HEBT2"]]
Replace_BaseRF_Gap_and_Quads_to_Overlapping_Nodes(
    lattice, z_step, fields_dir, sequences, [], SNS_EngeFunctionFactory
)
