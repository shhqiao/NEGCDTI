from PIL import Image
import os
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os
from rdkit import Chem


def smile2feature(data_root, file_data):
    with open(file_data, "r") as f:
        data_list = f.read().strip().split("\n")

    """Exclude data contains '.' in the SMILES format."""  # The '.' represents multiple chemical molecules
    # data_list = [d for d in data_list if '.' not in d.strip().split()[0]]

    smile_features = []
    file_name = data_root + "/" + "drugs" + "_smile_features.txt"
    with open(file_name, "w") as w:
        for i, data in enumerate(data_list):
            if i % 50 == 0:
                print('/'.join(map(str, [i + 1, len(data_list)])))

            fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
            factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
            smile = data.strip()
            mol = Chem.MolFromSmiles(smile)
            feats = factory.GetFeaturesForMol(mol)

            line = ""
            for f in feats:
                s = str(f.GetFamily())
                s += " " + str(f.GetType())
                s += " " + str(f.GetAtomIds())
                # s += " " + str(f.GetId())
                line += s + " "
            line += "\n"
            w.write(line)


if __name__ == '__main__':
    dataset_name = "KIBA"
    data_root = "Data/" + dataset_name
    train_file = data_root + "/" + "drugs" + "_train.txt"


    smile2feature(data_root, train_file)

