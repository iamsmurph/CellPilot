import numpy as np 
import pandas as pd

def drfp(reaction_smiles, bond_radius=3, nBits=2048):
    """
    https://github.com/reymond-group/drfp

    Builds reaction representation as a binary DRFP fingerprints.
    :param reaction_smiles: list of reaction smiles
    :type reaction_smiles: list
    :return: array of shape [len(reaction_smiles), nBits] with drfp featurised reactions

    """
    try:
        from drfp import DrfpEncoder
    except ImportError as e:
        raise ImportError(
            "Optional dependency 'drfp' is not installed. Install it (e.g. `uv add drfp`) or use another featurizer."
        ) from e

    fps = DrfpEncoder.encode(
        reaction_smiles, n_folded_length=nBits, radius=bond_radius
    )
    print(np.array(fps, dtype=np.float64).shape, "drfp vectors shape")

    return np.array(fps, dtype=np.float64)


def one_hot(df):
    """
    Builds reaction representation as a bit vector which indicates whether
    a certain condition, reagent, reactant etc. is present in the reaction.

    :param df: pandas DataFrame with columns representing different
    parameters of the reaction (e.g. reactants, reagents, conditions).
    :type df: pandas DataFrame
    :return: array of shape [len(reaction_smiles), sum(unique values for different columns in df)]
     with one-hot encoding of reactions
    """
    df_ohe = pd.get_dummies(df)
    return df_ohe.to_numpy(dtype=np.float64)

def rxnfp(reaction_smiles):
    """
    https://rxn4chemistry.github.io/rxnfp/

    Builds reaction representation as a continuous RXNFP fingerprints.
    :param reaction_smiles: list of reaction smiles
    :type reaction_smiles: list
    :return: array of shape [len(reaction_smiles), 256] with rxnfp featurised reactions

    """
    try:
        from rxnfp.transformer_fingerprints import (
            RXNBERTFingerprintGenerator,
            get_default_model_and_tokenizer,
        )
    except ImportError as e:
        raise ImportError(
            "Optional dependency 'rxnfp' is not installed. Install it (e.g. `uv add rxnfp`) or use another featurizer."
        ) from e

    rxn_model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(rxn_model, tokenizer)
    rxnfps = [rxnfp_generator.convert(smile) for smile in reaction_smiles]
    return np.array(rxnfps, dtype=np.float64)

